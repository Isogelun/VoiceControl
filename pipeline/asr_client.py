"""
ASR HTTP client.

The pipeline sends raw 16 kHz mono int16 PCM to the local ASR service as a WAV
upload. The client keeps one aiohttp session per process to avoid rebuilding
connections for every utterance.
"""

import io
import logging
import os
import struct
import time

import aiohttp

log = logging.getLogger(__name__)

ASR_URL = os.environ.get("ASR_URL", "http://localhost:8000/asr")
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "90"))
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
_ASR_SESSION = None


def _pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = PCM_SAMPLE_RATE,
    channels: int = PCM_CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> bytes:
    data_size = len(pcm_bytes)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * sample_width))
    buf.write(struct.pack("<H", channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


def _pcm_duration_ms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = len(pcm_bytes) / PCM_SAMPLE_WIDTH / PCM_CHANNELS
    return samples / PCM_SAMPLE_RATE * 1000


async def call_asr(pcm_bytes: bytes) -> str:
    wav_bytes = _pcm_to_wav(pcm_bytes)
    audio_ms = _pcm_duration_ms(pcm_bytes)
    started = time.perf_counter()

    try:
        form = aiohttp.FormData()
        form.add_field("audio", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("language", "zh")
        form.add_field("use_itn", "true")

        session = await _get_asr_session()
        async with session.post(ASR_URL, data=form) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.error("ASR service returned %d: %s", resp.status, body[:300])
                return ""

            result = await resp.json()
            text = result.get("text", "")
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.info(
                "ASR result: %s (service %sms, HTTP %.1fms, audio %.0fms)",
                text,
                result.get("total_ms", "?"),
                elapsed_ms,
                audio_ms,
            )
            return text
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.error(
            "ASR call failed: %s: %s (HTTP %.1fms, timeout %.1fs, audio %.0fms)",
            type(exc).__name__,
            exc,
            elapsed_ms,
            ASR_TIMEOUT,
            audio_ms,
        )
        return ""


async def _get_asr_session():
    global _ASR_SESSION
    if _ASR_SESSION is None or _ASR_SESSION.closed:
        timeout = aiohttp.ClientTimeout(total=ASR_TIMEOUT)
        _ASR_SESSION = aiohttp.ClientSession(timeout=timeout)
    return _ASR_SESSION


async def close_asr_session():
    global _ASR_SESSION
    if _ASR_SESSION is not None and not _ASR_SESSION.closed:
        await _ASR_SESSION.close()
    _ASR_SESSION = None
