"""
pipeline/main.py

Wake word -> VAD -> ASR -> NLU/rules -> command JSON.
"""

import asyncio
import json
import logging
import os

import numpy as np
import sherpa_onnx
try:
    import webrtcvad
except ImportError:
    webrtcvad = None

from .asr_client import call_asr
from .cleaner import start_cleaner
from .command_dispatcher import CommandDispatcher
from .nlu_client import call_nlu
from .speaker import Speaker
from .text_normalizer import (
    is_wake_phrase,
    normalize_asr_text,
    parse_command_rule,
)

log = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.8.181")
KWS_MODEL_DIR = os.environ.get("KWS_MODEL_DIR", os.path.join(_PROJECT_ROOT, "models", "kws"))
WAKE_KEYWORD = os.environ.get("WAKE_KEYWORD", "n ǐ h ǎo h uā h uā @你好花花")
WAKE_BACKEND = os.environ.get("WAKE_BACKEND", "asr" if os.name == "nt" else "kws").lower()
WAKE_TEXT = os.environ.get("WAKE_TEXT", "你好花花,你好，花花,花花")
WAKE_AUDIO = os.environ.get("WAKE_AUDIO", os.path.join(_PROJECT_ROOT, "audio", "xuanxinghuida.mp3"))

VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000  # 480
VAD_MODE = os.environ.get("VAD_MODE", "silence").lower()
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
VAD_SILENCE_RMS = float(os.environ.get("VAD_SILENCE_RMS", "300"))
VAD_SILENCE_MULTIPLIER = float(os.environ.get("VAD_SILENCE_MULTIPLIER", "2.5"))
SILENCE_TIMEOUT_MS = int(os.environ.get("VAD_SILENCE_TIMEOUT_MS", "1200"))
MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "240"))
COMMAND_LISTEN_TIMEOUT_MS = int(os.environ.get("COMMAND_LISTEN_TIMEOUT_MS", "8000"))
UTTERANCE_PAD_MS = int(os.environ.get("UTTERANCE_PAD_MS", "240"))
SILENCE_TIMEOUT_FRAMES = max(1, SILENCE_TIMEOUT_MS // VAD_FRAME_MS)
MIN_SPEECH_FRAMES = max(1, MIN_SPEECH_MS // VAD_FRAME_MS)
COMMAND_LISTEN_TIMEOUT_FRAMES = max(1, COMMAND_LISTEN_TIMEOUT_MS // VAD_FRAME_MS)
UTTERANCE_PAD_SAMPLES = VAD_SAMPLE_RATE * UTTERANCE_PAD_MS // 1000
UTTERANCE_PAD_BYTES = np.zeros(UTTERANCE_PAD_SAMPLES, dtype=np.int16).tobytes()


def _make_kws():
    return sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(KWS_MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(KWS_MODEL_DIR, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=os.path.join(KWS_MODEL_DIR, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=os.path.join(KWS_MODEL_DIR, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        keywords_file=_write_keywords_file(),
        num_threads=2,
    )


def _write_keywords_file() -> str:
    path = os.path.join(KWS_MODEL_DIR, "keywords.txt")
    keywords = [k.strip() for k in WAKE_KEYWORD.split(",") if k.strip()]
    with open(path, "w", encoding="utf-8") as f:
        for kw in keywords:
            f.write(kw + "\n")
    return path


def _to_16k_mono(frame) -> np.ndarray:
    raw = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
    mono = raw.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return mono[::3]


class VoicePipeline:
    def __init__(self, speaker: Speaker = None):
        self.wake_backend = WAKE_BACKEND
        self.wake_texts = [t.strip() for t in WAKE_TEXT.split(",") if t.strip()]
        self.kws = None
        self.kws_stream = None
        if self.wake_backend == "kws":
            self.kws = _make_kws()
            self.kws_stream = self.kws.create_stream()
        self.vad_mode = VAD_MODE
        if self.vad_mode == "webrtc" and webrtcvad is None:
            log.warning("webrtcvad 不可用，自动切换到 silence VAD")
            self.vad_mode = "silence"
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if self.vad_mode == "webrtc" else None
        self._noise_rms = VAD_SILENCE_RMS
        self.speaker = speaker
        self.dispatcher = CommandDispatcher(speaker=speaker)

        self._state = "waiting"
        self._pcm_buf = np.array([], dtype=np.int16)
        self._wake_metadata = {}
        self._reset_speech_capture()
        log.info(
            "VAD 裁切: mode=%s silence_rms=%.1f multiplier=%.2f silence_timeout=%sms min_speech=%sms",
            self.vad_mode,
            VAD_SILENCE_RMS,
            VAD_SILENCE_MULTIPLIER,
            SILENCE_TIMEOUT_MS,
            MIN_SPEECH_MS,
        )

    async def on_audio_frame(self, frame):
        pcm = _to_16k_mono(frame)
        await self.push_pcm(pcm)

    async def push_pcm(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if self._state == "waiting" and self.wake_backend == "hardware":
            return
        self._pcm_buf = np.concatenate([self._pcm_buf, pcm])
        if self._state == "waiting":
            if self.wake_backend == "kws":
                await self._run_kws_wakeword()
            else:
                await self._run_asr_wakeword()
        else:
            await self._run_command_vad()

    async def trigger_wake(self, metadata: dict = None):
        wake_metadata = {"source": "hardware"}
        if metadata:
            wake_metadata.update(metadata)
            log.info("硬件唤醒事件: %s", json.dumps(wake_metadata, ensure_ascii=False))
        else:
            log.info("硬件唤醒事件")
        self._enter_listening(wake_metadata)

    async def _run_kws_wakeword(self):
        chunk = self._pcm_buf.astype(np.float32) / 32768.0
        self._pcm_buf = np.array([], dtype=np.int16)
        self.kws_stream.accept_waveform(VAD_SAMPLE_RATE, chunk)
        self.kws.decode_stream(self.kws_stream)
        result = self.kws_stream.result
        if result.keyword:
            log.info("唤醒词 [%s] 检测到，开始监听...", result.keyword.strip())
            self.kws_stream = self.kws.create_stream()
            self._enter_listening({"source": "kws", "keyword": result.keyword.strip()})

    async def _run_asr_wakeword(self):
        while len(self._pcm_buf) >= VAD_FRAME_SAMPLES:
            chunk = self._pcm_buf[:VAD_FRAME_SAMPLES]
            self._pcm_buf = self._pcm_buf[VAD_FRAME_SAMPLES:]
            chunk_bytes = chunk.astype(np.int16).tobytes()

            is_speech = self._is_speech(chunk, chunk_bytes)
            self._speech_buf += chunk_bytes

            if is_speech:
                self._speech_frame_count += 1
                self._silence_count = 0
            else:
                self._silence_count += 1

            if self._silence_count >= SILENCE_TIMEOUT_FRAMES:
                if self._speech_frame_count >= MIN_SPEECH_FRAMES:
                    await self._process_wake_utterance(self._with_padding(self._speech_buf))
                self._reset_speech_capture()
                break

    async def _run_command_vad(self):
        while len(self._pcm_buf) >= VAD_FRAME_SAMPLES:
            chunk = self._pcm_buf[:VAD_FRAME_SAMPLES]
            self._pcm_buf = self._pcm_buf[VAD_FRAME_SAMPLES:]
            chunk_bytes = chunk.astype(np.int16).tobytes()
            self._listen_frame_count += 1

            is_speech = self._is_speech(chunk, chunk_bytes)

            if is_speech:
                self._speech_buf += chunk_bytes
                self._speech_frame_count += 1
                self._silence_count = 0
            elif self._speech_frame_count > 0:
                self._speech_buf += chunk_bytes
                self._silence_count += 1
            elif self._listen_frame_count >= COMMAND_LISTEN_TIMEOUT_FRAMES:
                log.info("命令监听超时，回到等待唤醒")
                self._state = "waiting"
                self._wake_metadata = {}
                self._reset_speech_capture()
                break

            if self._silence_count >= SILENCE_TIMEOUT_FRAMES:
                if self._speech_frame_count >= MIN_SPEECH_FRAMES:
                    await self._process_utterance(self._with_padding(self._speech_buf))
                else:
                    log.debug("语句太短，丢弃")
                self._state = "waiting"
                self._wake_metadata = {}
                self._reset_speech_capture()
                break

    async def _process_wake_utterance(self, pcm_bytes: bytes):
        text = await call_asr(pcm_bytes)
        normalized = normalize_asr_text(text)
        if text:
            log.info("唤醒检测 ASR: %s -> %s", text, normalized)
        if normalized and is_wake_phrase(normalized, self.wake_texts):
            log.info("唤醒词 [你好花花] 检测到，开始监听...")
            self._enter_listening(
                {
                    "source": "asr",
                    "keyword": normalized,
                    "asr_text": text,
                    "normalized_text": normalized,
                }
            )

    async def _process_utterance(self, pcm_bytes: bytes):
        wake_metadata = dict(self._wake_metadata)
        try:
            log.info("识别中...")
            text = await call_asr(pcm_bytes)
            log.info("ASR: %s", text)
            if not text:
                return

            normalized = normalize_asr_text(text)
            if normalized != text:
                log.info("ASR 归一化: %s -> %s", text, normalized)

            result = parse_command_rule(text)
            if result:
                log.info("规则命中: %s", json.dumps(result, ensure_ascii=False))
            else:
                result = await call_nlu(normalized)
            log.info("指令 JSON: %s", json.dumps(result, ensure_ascii=False))
            dispatch_result = await self.dispatcher.dispatch(result, text, normalized, wake_metadata)
            log.info("指令分发结果: %s", json.dumps(dispatch_result, ensure_ascii=False))
        finally:
            self._wake_metadata = {}

    def _enter_listening(self, wake_metadata: dict = None):
        self._state = "listening"
        self._wake_metadata = wake_metadata or {}
        self._reset_speech_capture()
        log.info("已进入命令监听，请说指令")
        if self.speaker and WAKE_AUDIO:
            asyncio.create_task(self.speaker.play_file(WAKE_AUDIO))

    def _reset_speech_capture(self):
        self._speech_buf = b""
        self._silence_count = 0
        self._speech_frame_count = 0
        self._listen_frame_count = 0

    def _with_padding(self, pcm_bytes: bytes) -> bytes:
        if not UTTERANCE_PAD_BYTES:
            return pcm_bytes
        return UTTERANCE_PAD_BYTES + pcm_bytes + UTTERANCE_PAD_BYTES

    def _is_speech(self, chunk: np.ndarray, chunk_bytes: bytes) -> bool:
        if self.vad_mode == "webrtc":
            return self.vad.is_speech(chunk_bytes, VAD_SAMPLE_RATE)

        rms = _frame_rms(chunk)
        threshold = max(VAD_SILENCE_RMS, self._noise_rms * VAD_SILENCE_MULTIPLIER)
        is_speech = rms >= threshold
        if not is_speech:
            self._noise_rms = 0.995 * self._noise_rms + 0.005 * rms
        return is_speech

    def close(self):
        pass


def _frame_rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


async def run_webrtc():
    from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    speaker = Speaker(conn)
    pipe = VoicePipeline(speaker=speaker)
    connected = False

    try:
        await conn.connect()
        connected = True
        conn.audio.switchAudioChannel(True)
        conn.audio.add_track_callback(pipe.on_audio_frame)
        asyncio.create_task(start_cleaner())
        log.info("等待唤醒词 %s...", WAKE_KEYWORD)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()
        if connected:
            await conn.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run_webrtc())
