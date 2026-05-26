"""
pipeline/asr_client.py

ASR HTTP 客户端 — 调用 asr/server.py 提供的 HTTP 服务。

环境变量:
    ASR_URL   ASR 服务地址，默认 http://localhost:8000/asr
"""

import io
import os
import struct
import logging

import aiohttp

log = logging.getLogger(__name__)

ASR_URL = os.environ.get("ASR_URL", "http://localhost:8000/asr")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """将原始 PCM 字节转为 WAV 格式（加 44 字节头）"""
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


async def call_asr(pcm_bytes: bytes) -> str:
    """调用 ASR HTTP 服务识别语音"""
    wav_bytes = _pcm_to_wav(pcm_bytes)
    try:
        form = aiohttp.FormData()
        form.add_field("audio", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("language", "zh")
        form.add_field("use_itn", "true")

        async with aiohttp.ClientSession() as session:
            async with session.post(ASR_URL, data=form, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.error("ASR 服务返回 %d", resp.status)
                    return ""
                result = await resp.json()
                text = result.get("text", "")
                log.info("ASR 结果: %s (耗时 %sms)", text, result.get("total_ms", "?"))
                return text
    except Exception as e:
        log.error("ASR 调用失败: %s", e)
        return ""
