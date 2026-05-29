"""
pipeline/main.py

Wake word -> VAD -> ASR -> NLU -> command JSON.
"""

import asyncio
import json
import logging
import os
import time

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
    compact_text,
    is_wake_phrase,
    normalize_asr_text,
    parse_command_rule,
)

log = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.8.181")
ROBOT_WEBRTC_METHOD = os.environ.get("UNITREE_WEBRTC_METHOD", "LocalSTA")
ROBOT_SERIAL_NUMBER = os.environ.get("UNITREE_ROBOT_SERIAL_NUMBER") or None
ROBOT_AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY") or None
ROBOT_USERNAME = os.environ.get("UNITREE_USERNAME") or None
ROBOT_PASSWORD = os.environ.get("UNITREE_PASSWORD") or None
ROBOT_REGION = os.environ.get("UNITREE_REGION", "global")
ROBOT_DEVICE_TYPE = os.environ.get("UNITREE_DEVICE_TYPE", "Go2")
WEBRTC_LEVEL_LOG_INTERVAL = float(os.environ.get("WEBRTC_LEVEL_LOG_INTERVAL", os.environ.get("MIC_LEVEL_LOG_INTERVAL", "3")))
WEBRTC_AUDIO_GAIN = float(os.environ.get("WEBRTC_AUDIO_GAIN", "1.0"))
WEBRTC_AUDIO_DENOISE = os.environ.get("WEBRTC_AUDIO_DENOISE", os.environ.get("AUDIO_DENOISE", "0")) not in {"0", "false", "False", "no"}
WEBRTC_NOISE_GATE_RMS = float(os.environ.get("WEBRTC_NOISE_GATE_RMS", "80"))
WEBRTC_NOISE_GATE_ATTENUATION = float(os.environ.get("WEBRTC_NOISE_GATE_ATTENUATION", "0.2"))
WEBRTC_TARGET_PEAK = float(os.environ.get("WEBRTC_TARGET_PEAK", "12000"))
WEBRTC_CONNECT_RETRIES = max(1, int(os.environ.get("UNITREE_WEBRTC_CONNECT_RETRIES", "3")))
WEBRTC_RETRY_DELAY_MS = max(0, int(os.environ.get("UNITREE_WEBRTC_RETRY_DELAY_MS", "5000")))
KWS_MODEL_DIR = os.environ.get("KWS_MODEL_DIR", os.path.join(_PROJECT_ROOT, "models", "kws"))
WAKE_KEYWORD = os.environ.get("WAKE_KEYWORD", "n ǐ h ǎo h uā h uā @你好花花")
WAKE_BACKEND = os.environ.get("WAKE_BACKEND", "asr" if os.name == "nt" else "kws").lower()
WAKE_TEXT = os.environ.get("WAKE_TEXT", "你好花花,你好，花花,花花")
WAKE_ALIASES = os.environ.get(
    "WAKE_ALIASES",
    "你好曼波,曼波,慢播,快播,那波,南波,慢波,曼播,你好慢播,你好快播,你好那波,你好南波",
)
WAKE_AUDIO = os.environ.get("WAKE_AUDIO", os.path.join(_PROJECT_ROOT, "audio", "xuanxinghuida.mp3"))

WAKE_FEEDBACK_ENABLED = os.environ.get("WAKE_FEEDBACK_ENABLED", "0") not in {"0", "false", "False", "no", ""}
COMMAND_RULES_ENABLED = os.environ.get("COMMAND_RULES_ENABLED", "0") not in {"0", "false", "False", "no", ""}
COMMAND_RULES_FAST_PATH = os.environ.get("COMMAND_RULES_FAST_PATH", "1") not in {"0", "false", "False", "no", ""}
COMMAND_FEEDBACK_SUPPRESS_MS = int(os.environ.get("COMMAND_FEEDBACK_SUPPRESS_MS", "1800"))

VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000  # 480
VAD_MODE = os.environ.get("VAD_MODE", "silence").lower()
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
VAD_SILENCE_RMS = float(os.environ.get("VAD_SILENCE_RMS", "300"))
VAD_SILENCE_MULTIPLIER = float(os.environ.get("VAD_SILENCE_MULTIPLIER", "2.5"))
COMMAND_VAD_SILENCE_RMS = float(os.environ.get("COMMAND_VAD_SILENCE_RMS", "360"))
COMMAND_VAD_SILENCE_MULTIPLIER = float(os.environ.get("COMMAND_VAD_SILENCE_MULTIPLIER", "1.15"))
VAD_DEBUG = os.environ.get("VAD_DEBUG", "0") not in {"0", "false", "False", "no", ""}
VAD_DEBUG_INTERVAL = float(os.environ.get("VAD_DEBUG_INTERVAL", "1.0"))
SILENCE_TIMEOUT_MS = int(os.environ.get("VAD_SILENCE_TIMEOUT_MS", "1200"))
COMMAND_SILENCE_TIMEOUT_MS = int(os.environ.get("COMMAND_VAD_SILENCE_TIMEOUT_MS", "360"))
MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "240"))
COMMAND_LISTEN_TIMEOUT_MS = int(os.environ.get("COMMAND_LISTEN_TIMEOUT_MS", "8000"))
UTTERANCE_PAD_MS = int(os.environ.get("UTTERANCE_PAD_MS", "80"))
UTTERANCE_TRIM_ENABLED = os.environ.get("UTTERANCE_TRIM_ENABLED", "1") not in {"0", "false", "False", "no", ""}
UTTERANCE_TRIM_PAD_MS = int(os.environ.get("UTTERANCE_TRIM_PAD_MS", "90"))
SILENCE_TIMEOUT_FRAMES = max(1, SILENCE_TIMEOUT_MS // VAD_FRAME_MS)
COMMAND_SILENCE_TIMEOUT_FRAMES = max(1, COMMAND_SILENCE_TIMEOUT_MS // VAD_FRAME_MS)
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
    source_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
    source_channels = _frame_channel_count(frame)

    if raw.size and raw.size % source_channels == 0:
        mono = raw.reshape(-1, source_channels).mean(axis=1)
    else:
        mono = raw.astype(np.float32, copy=False)

    if mono.size in {320, 640, 960}:
        source_rate = int(mono.size * 50)

    mono = _condition_webrtc_pcm(mono)
    if source_rate != VAD_SAMPLE_RATE and mono.size:
        target_len = max(1, int(round(mono.size * VAD_SAMPLE_RATE / source_rate)))
        source_x = np.arange(mono.size, dtype=np.float32)
        target_x = np.linspace(0, mono.size - 1, target_len, dtype=np.float32)
        mono = np.interp(target_x, source_x, mono).astype(np.float32)

    return np.clip(mono, -32768, 32767).astype(np.int16)


def _frame_channel_count(frame) -> int:
    layout = getattr(frame, "layout", None)
    channels = getattr(layout, "channels", None)
    try:
        count = len(channels) if channels is not None else 0
    except TypeError:
        count = int(channels or 0)
    return max(1, count)


def _condition_webrtc_pcm(mono: np.ndarray) -> np.ndarray:
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    if not x.size:
        return x

    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        limiter_gain = min(1.0, WEBRTC_TARGET_PEAK / peak)
        x *= limiter_gain
    x *= WEBRTC_AUDIO_GAIN

    if WEBRTC_AUDIO_DENOISE:
        rms = _frame_rms(x)
        if rms < WEBRTC_NOISE_GATE_RMS:
            x *= WEBRTC_NOISE_GATE_ATTENUATION
    return x


def _resolve_webrtc_method(method_name: str, enum_cls):
    normalized = (method_name or "LocalSTA").replace("_", "").replace("-", "").lower()
    aliases = {
        "localsta": "LocalSTA",
        "sta": "LocalSTA",
        "lan": "LocalSTA",
        "localap": "LocalAP",
        "ap": "LocalAP",
        "remote": "Remote",
        "cloud": "Remote",
    }
    enum_name = aliases.get(normalized)
    if not enum_name:
        valid = ", ".join(item.name for item in enum_cls)
        raise RuntimeError(f"Unsupported UNITREE_WEBRTC_METHOD={method_name!r}; use one of: {valid}")
    return getattr(enum_cls, enum_name)


def _make_go2_webrtc_connection():
    from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod

    method = _resolve_webrtc_method(ROBOT_WEBRTC_METHOD, WebRTCConnectionMethod)
    kwargs = {
        "serialNumber": ROBOT_SERIAL_NUMBER,
        "ip": ROBOT_IP,
        "username": ROBOT_USERNAME,
        "password": ROBOT_PASSWORD,
        "aes_128_key": ROBOT_AES_128_KEY,
        "region": ROBOT_REGION,
        "device_type": ROBOT_DEVICE_TYPE,
    }
    if method.name in {"LocalAP", "Remote"}:
        kwargs["ip"] = None
    log.info(
        "GO2 WebRTC: method=%s ip=%s serial=%s region=%s device_type=%s aes_key=%s",
        method.name,
        kwargs.get("ip") or "-",
        ROBOT_SERIAL_NUMBER or "-",
        ROBOT_REGION,
        ROBOT_DEVICE_TYPE,
        "set" if ROBOT_AES_128_KEY else "unset",
    )
    return UnitreeWebRTCConnection(method, **kwargs)


def _unique_phrases(phrases):
    seen = set()
    out = []
    for phrase in phrases:
        normalized = normalize_asr_text(phrase).strip()
        key = compact_text(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _strip_leading_wake_aliases(text: str, wake_phrases) -> str:
    candidate = normalize_asr_text(text or "")
    for _ in range(3):
        stripped = _strip_boundary_punctuation(candidate)
        changed = False
        for phrase in wake_phrases:
            phrase_key = compact_text(normalize_asr_text(phrase))
            if not phrase_key:
                continue
            after = _slice_after_leading_compact_phrase(stripped, phrase_key)
            if after is None:
                continue
            if len(after) < len(stripped):
                candidate = after
                changed = True
                break
        if not changed:
            return stripped
    return _strip_boundary_punctuation(candidate)


def _slice_after_leading_compact_phrase(text: str, phrase_compact: str):
    compact_chars = []
    original_ends = []
    for index, ch in enumerate(text):
        if not ch.lower().isalnum():
            continue
        compact_chars.append(ch.lower())
        original_ends.append(index + 1)

    compact = "".join(compact_chars)
    if not compact.startswith(phrase_compact):
        return None
    end_index = original_ends[len(phrase_compact) - 1]
    return _strip_boundary_punctuation(text[end_index:])


def _strip_boundary_punctuation(text: str) -> str:
    return (text or "").strip(" \t\r\n,.;:!?\uFF0C\u3002\uFF01\uFF1F\uFF1B\uFF1A")


class VoicePipeline:
    def __init__(self, speaker: Speaker = None, metadata_provider=None):
        self.wake_backend = WAKE_BACKEND
        self.wake_texts = _unique_phrases(
            [t.strip() for t in WAKE_TEXT.split(",") if t.strip()]
            + [t.strip() for t in WAKE_ALIASES.split(",") if t.strip()]
        )
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
        self.metadata_provider = metadata_provider
        self.dispatcher = CommandDispatcher(speaker=speaker)
        self._audio_stats = {
            "frames": 0,
            "samples": 0,
            "rms": 0.0,
            "peak": 0,
            "source_rate": 0,
            "source_channels": 0,
            "last_log": time.monotonic(),
        }
        self._last_vad_debug_log = 0.0

        self._state = "waiting"
        self._pcm_buf = np.array([], dtype=np.int16)
        self._wake_metadata = {}
        self._feedback_suppress_until = 0.0
        self._reset_speech_capture()
        log.info("Wake feedback enabled: %s", WAKE_FEEDBACK_ENABLED)
        log.info(
            "VAD 裁切: mode=%s silence_rms=%.1f multiplier=%.2f wake_timeout=%sms command_timeout=%sms min_speech=%sms",
            self.vad_mode,
            VAD_SILENCE_RMS,
            VAD_SILENCE_MULTIPLIER,
            SILENCE_TIMEOUT_MS,
            COMMAND_SILENCE_TIMEOUT_MS,
            MIN_SPEECH_MS,
        )

    async def on_audio_frame(self, frame):
        pcm = _to_16k_mono(frame)
        self._log_audio_level(frame, pcm)
        await self.push_pcm(pcm)

    async def push_pcm(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if self._is_feedback_suppressed():
            self._pcm_buf = np.array([], dtype=np.int16)
            self._reset_speech_capture()
            return
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

            if self._listen_frame_count >= COMMAND_LISTEN_TIMEOUT_FRAMES:
                if self._speech_frame_count >= MIN_SPEECH_FRAMES:
                    log.info(
                        "Command listen timeout with speech buffered: speech=%sms, running ASR/NLU",
                        self._speech_frame_count * VAD_FRAME_MS,
                    )
                    await self._process_utterance(self._with_padding(self._speech_buf))
                else:
                    log.info("Command listen timeout without speech, back to wake waiting")
                self._state = "waiting"
                self._wake_metadata = {}
                self._reset_speech_capture()
                break

            if self._silence_count >= COMMAND_SILENCE_TIMEOUT_FRAMES:
                if self._speech_frame_count >= MIN_SPEECH_FRAMES:
                    log.info(
                        "Command speech ended: speech=%sms silence=%sms, running ASR/NLU",
                        self._speech_frame_count * VAD_FRAME_MS,
                        COMMAND_SILENCE_TIMEOUT_MS,
                    )
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
            command_text = self._strip_wake_phrase(normalized)
            if command_text:
                log.info("Wake phrase and command in one utterance: %s", command_text)
                self._wake_metadata = self._with_external_metadata({
                    "source": "asr",
                    "keyword": normalized,
                    "asr_text": text,
                    "normalized_text": normalized,
                    "inline_command": True,
                })
                await self._process_command_text(command_text, text, command_text)
                self._state = "waiting"
                self._wake_metadata = {}
                self._reset_speech_capture()
                return
            log.info("唤醒词 [你好花花] 检测到，开始监听...")
            self._enter_listening({
                "source": "asr",
                "keyword": normalized,
                "asr_text": text,
                "normalized_text": normalized,
            })

    async def _process_utterance(self, pcm_bytes: bytes):
        try:
            log.info("识别中...")
            text = await call_asr(pcm_bytes)
            log.info("ASR: %s", text)
            if not text:
                await self.dispatcher.play_unavailable()
                return

            normalized = normalize_asr_text(text)
            if normalized != text:
                log.info("ASR 归一化: %s -> %s", text, normalized)

            command_text = normalized
            if is_wake_phrase(normalized, getattr(self, "wake_texts", [])):
                stripped = self._strip_wake_phrase(normalized)
                if stripped:
                    command_text = stripped
                    log.info("Command text after wake-prefix stripping: %s -> %s", normalized, command_text)
                else:
                    log.info("Command utterance only contains wake phrase, ignored: %s", normalized)
                    await self.dispatcher.play_unavailable()
                    return

            result = await self._parse_command_with_nlu(command_text)
            log.info("指令 JSON: %s", json.dumps(result, ensure_ascii=False))
            dispatch_result = await self.dispatcher.dispatch(
                result,
                text,
                command_text,
                self._with_external_metadata(dict(self._wake_metadata)),
            )
            log.info("指令分发结果: %s", json.dumps(dispatch_result, ensure_ascii=False))
            self._suppress_feedback_audio()
        finally:
            self._wake_metadata = {}

    async def _process_command_text(self, command_text: str, asr_text: str, normalized_text: str):
        wake_metadata = self._with_external_metadata(dict(self._wake_metadata))
        result = await self._parse_command_with_nlu(command_text)
        log.info("Command JSON: %s", json.dumps(result, ensure_ascii=False))
        dispatch_result = await self.dispatcher.dispatch(result, asr_text, normalized_text, wake_metadata)
        log.info("Command dispatch result: %s", json.dumps(dispatch_result, ensure_ascii=False))
        self._suppress_feedback_audio()

    async def _parse_command_with_nlu(self, text: str) -> dict:
        if COMMAND_RULES_FAST_PATH:
            fallback = parse_command_rule(text)
            if fallback:
                log.info("Rule fast-path matched before NLU: %s", json.dumps(fallback, ensure_ascii=False))
                return fallback

        try:
            result = await call_nlu(text)
            log.info("NLU model result: %s", json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            log.exception("NLU failed, using fallback for text: %s", text)
            fallback = parse_command_rule(text)
            if fallback:
                log.info("Rule fallback matched after NLU failure: %s", json.dumps(fallback, ensure_ascii=False))
                return fallback
            return {
                "intent": "unknown",
                "slots": {},
                "source": "nlu_error",
                "error": str(exc),
                "raw": text,
            }
        if COMMAND_RULES_ENABLED and (not result or result.get("intent") == "unknown"):
            fallback = parse_command_rule(text)
            if fallback:
                log.info("Rule fallback matched: %s", json.dumps(fallback, ensure_ascii=False))
                return fallback
        return result

    def _strip_wake_phrase(self, text: str) -> str:
        normalized = normalize_asr_text(text)
        best = ""
        for phrase in self.wake_texts:
            phrase_normalized = normalize_asr_text(phrase)
            phrase_compact = compact_text(phrase_normalized)
            if not phrase_compact:
                continue
            candidate = self._slice_after_compact_phrase(normalized, phrase_compact)
            if candidate is None:
                continue
            if len(candidate) > len(best):
                best = candidate
        return normalize_asr_text(_strip_leading_wake_aliases(best, self.wake_texts))

    @staticmethod
    def _slice_after_compact_phrase(text: str, phrase_compact: str):
        compact_chars = []
        original_ends = []
        for index, ch in enumerate(text):
            if not ch.lower().isalnum():
                continue
            compact_chars.append(ch.lower())
            original_ends.append(index + 1)

        compact = "".join(compact_chars)
        phrase_index = compact.find(phrase_compact)
        if phrase_index < 0:
            return None
        end_index = original_ends[phrase_index + len(phrase_compact) - 1]
        return text[end_index:].strip(" ，,。.!！？?;；:：")

    def _enter_listening(self, wake_metadata: dict = None):
        self._state = "listening"
        self._wake_metadata = self._with_external_metadata(wake_metadata or {})
        self._reset_speech_capture()
        log.info("已进入命令监听，请说指令")
        if WAKE_FEEDBACK_ENABLED and WAKE_AUDIO:
            asyncio.create_task(self.dispatcher.play_audio(WAKE_AUDIO, success=True))

    def _with_external_metadata(self, metadata: dict) -> dict:
        out = dict(metadata or {})
        metadata_provider = getattr(self, "metadata_provider", None)
        if not metadata_provider:
            return out
        try:
            extra = metadata_provider() or {}
        except Exception:
            log.exception("Failed to read external voice metadata")
            return out
        if extra:
            out.update(extra)
        return out

    def _suppress_feedback_audio(self):
        if COMMAND_FEEDBACK_SUPPRESS_MS <= 0:
            return
        self._feedback_suppress_until = time.monotonic() + COMMAND_FEEDBACK_SUPPRESS_MS / 1000.0
        self._pcm_buf = np.array([], dtype=np.int16)
        self._reset_speech_capture()
        log.info("Suppressing microphone input for %sms after feedback playback", COMMAND_FEEDBACK_SUPPRESS_MS)

    def _is_feedback_suppressed(self) -> bool:
        return time.monotonic() < getattr(self, "_feedback_suppress_until", 0.0)

    def _reset_speech_capture(self):
        self._speech_buf = b""
        self._silence_count = 0
        self._speech_frame_count = 0
        self._listen_frame_count = 0

    def _with_padding(self, pcm_bytes: bytes) -> bytes:
        pcm_bytes = self._trim_utterance(pcm_bytes)
        if not UTTERANCE_PAD_BYTES:
            return pcm_bytes
        return UTTERANCE_PAD_BYTES + pcm_bytes + UTTERANCE_PAD_BYTES

    def _trim_utterance(self, pcm_bytes: bytes) -> bytes:
        if not UTTERANCE_TRIM_ENABLED or not pcm_bytes:
            return pcm_bytes
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        if pcm.size < VAD_FRAME_SAMPLES:
            return pcm_bytes

        frames = pcm.size // VAD_FRAME_SAMPLES
        trimmed = pcm[:frames * VAD_FRAME_SAMPLES].reshape(frames, VAD_FRAME_SAMPLES)
        rms = np.sqrt(np.mean(trimmed.astype(np.float32) ** 2, axis=1))
        threshold = max(COMMAND_VAD_SILENCE_RMS * 0.6, self._noise_rms * COMMAND_VAD_SILENCE_MULTIPLIER)
        speech = np.flatnonzero(rms >= threshold)
        if speech.size == 0:
            return pcm_bytes

        pad_frames = max(1, UTTERANCE_TRIM_PAD_MS // VAD_FRAME_MS)
        start_frame = max(0, int(speech[0]) - pad_frames)
        end_frame = min(frames, int(speech[-1]) + pad_frames + 1)
        start = start_frame * VAD_FRAME_SAMPLES
        end = end_frame * VAD_FRAME_SAMPLES
        if start == 0 and end >= pcm.size:
            return pcm_bytes
        return pcm[start:end].astype(np.int16, copy=False).tobytes()

    def _is_speech(self, chunk: np.ndarray, chunk_bytes: bytes) -> bool:
        if self.vad_mode == "webrtc":
            is_speech = self.vad.is_speech(chunk_bytes, VAD_SAMPLE_RATE)
            self._log_vad_decision(
                rms=_frame_rms(chunk),
                threshold=None,
                is_speech=is_speech,
                mode="webrtc",
            )
            return is_speech

        rms = _frame_rms(chunk)
        if self._state == "listening":
            threshold = max(COMMAND_VAD_SILENCE_RMS, self._noise_rms * COMMAND_VAD_SILENCE_MULTIPLIER)
        else:
            threshold = max(VAD_SILENCE_RMS, self._noise_rms * VAD_SILENCE_MULTIPLIER)
        is_speech = rms >= threshold
        self._log_vad_decision(rms=rms, threshold=threshold, is_speech=is_speech, mode="silence")
        if not is_speech:
            self._noise_rms = 0.995 * self._noise_rms + 0.005 * rms
        return is_speech

    def _log_vad_decision(self, rms: float, threshold, is_speech: bool, mode: str):
        if not VAD_DEBUG:
            return
        now = time.monotonic()
        if now - self._last_vad_debug_log < VAD_DEBUG_INTERVAL:
            return
        self._last_vad_debug_log = now
        threshold_text = "-" if threshold is None else f"{threshold:.1f}"
        log.info(
            "VAD debug: state=%s mode=%s rms=%.1f threshold=%s noise=%.1f speech=%s speech_ms=%s silence_ms=%s listen_ms=%s",
            self._state,
            mode,
            rms,
            threshold_text,
            self._noise_rms,
            is_speech,
            self._speech_frame_count * VAD_FRAME_MS,
            self._silence_count * VAD_FRAME_MS,
            self._listen_frame_count * VAD_FRAME_MS,
        )

    def _log_audio_level(self, frame, pcm: np.ndarray):
        if WEBRTC_LEVEL_LOG_INTERVAL <= 0:
            return

        stats = self._audio_stats
        stats["frames"] += 1
        stats["samples"] += int(pcm.size)
        if pcm.size:
            x = pcm.astype(np.float32)
            stats["rms"] = float(np.sqrt(np.mean(x * x)))
            stats["peak"] = int(np.max(np.abs(pcm)))
        stats["source_rate"] = int(getattr(frame, "sample_rate", 0) or 0)
        layout = getattr(frame, "layout", None)
        channels = getattr(layout, "channels", None)
        try:
            stats["source_channels"] = len(channels) if channels is not None else 0
        except TypeError:
            stats["source_channels"] = int(channels or 0)

        now = time.monotonic()
        if now - stats["last_log"] < WEBRTC_LEVEL_LOG_INTERVAL:
            return
        elapsed = max(0.001, now - stats["last_log"])
        frame_rate = stats["frames"] / elapsed
        sample_rate = stats["samples"] / elapsed
        dbfs = _dbfs(stats["rms"])
        log.info(
            "GO2 WebRTC mic level: rms=%.1f peak=%d dbfs=%.1f frames=%.1f/s pcm_rate=%.0f/s src_rate=%s src_channels=%s state=%s",
            stats["rms"],
            stats["peak"],
            dbfs,
            frame_rate,
            sample_rate,
            stats["source_rate"] or "-",
            stats["source_channels"] or "-",
            self._state,
        )
        stats["frames"] = 0
        stats["samples"] = 0
        stats["last_log"] = now

    def close(self):
        try:
            from .asr_client import close_asr_session
            from .nlu_client import close_nlu_session
        except Exception:
            log.debug("Failed to import pipeline client closers", exc_info=True)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(close_asr_session())
        loop.create_task(close_nlu_session())


def _dbfs(rms: float) -> float:
    if rms <= 0:
        return -120.0
    return max(-120.0, 20.0 * np.log10(rms / 32768.0))


def _frame_rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


async def run_webrtc():
    from unitree_webrtc_connect import DataChannelTimeoutError, NoSdpAnswerError, RobotBusyError

    conn = None
    pipe = None
    connected = False

    try:
        for attempt in range(1, WEBRTC_CONNECT_RETRIES + 1):
            conn = _make_go2_webrtc_connection()
            try:
                log.info("WebRTC connect attempt %s/%s", attempt, WEBRTC_CONNECT_RETRIES)
                await conn.connect()
                connected = True
                break
            except RobotBusyError:
                raise
            except (DataChannelTimeoutError, NoSdpAnswerError, TimeoutError, OSError) as exc:
                await _safe_disconnect_webrtc(conn)
                if attempt >= WEBRTC_CONNECT_RETRIES:
                    raise
                delay_s = WEBRTC_RETRY_DELAY_MS / 1000.0
                log.warning(
                    "WebRTC connect attempt %s/%s failed: %s; retrying in %.1fs",
                    attempt,
                    WEBRTC_CONNECT_RETRIES,
                    exc,
                    delay_s,
                )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)

        speaker = Speaker(conn)
        pipe = VoicePipeline(speaker=speaker)
        conn.audio.add_track_callback(pipe.on_audio_frame)
        conn.audio.switchAudioChannel(True)
        asyncio.create_task(start_cleaner())
        log.info("等待唤醒词 %s...", WAKE_KEYWORD)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if pipe:
            pipe.close()
        if connected and conn:
            await _safe_disconnect_webrtc(conn)


async def _safe_disconnect_webrtc(conn):
    try:
        await conn.disconnect()
    except Exception:
        log.exception("WebRTC disconnect failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run_webrtc())
