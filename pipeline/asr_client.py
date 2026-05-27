"""
ASR HTTP client.

The pipeline sends raw 16 kHz mono int16 PCM to the local ASR service as a WAV
upload. Qwen3-ASR can be much slower than the old ASR model on CPU, so the
timeout is intentionally configurable and higher than before.
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

        timeout = aiohttp.ClientTimeout(total=ASR_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(ASR_URL, data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.error("ASR 服务返回 %d: %s", resp.status, body[:300])
                    return ""

                result = await resp.json()
                text = result.get("text", "")
                elapsed_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "ASR 结果: %s (服务耗时 %sms, HTTP %.1fms, 音频 %.0fms)",
                    text,
                    result.get("total_ms", "?"),
                    elapsed_ms,
                    audio_ms,
                )
                return text
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.error(
            "ASR 调用失败: %s: %s (HTTP %.1fms, timeout %.1fs, 音频 %.0fms)",
            type(exc).__name__,
            exc,
            elapsed_ms,
            ASR_TIMEOUT,
            audio_ms,
        )
        return ""
