"""
NLU HTTP client.
"""

import asyncio
import logging
import os

import aiohttp

log = logging.getLogger(__name__)

NLU_URL = os.environ.get("NLU_URL", "http://localhost:8001/nlu")
NLU_TIMEOUT = float(os.environ.get("NLU_TIMEOUT", "10"))
# 额外重试次数（总尝试次数 = NLU_RETRIES + 1）。失败时返回 unknown，由上层规则兜底接管。
NLU_RETRIES = max(0, int(os.environ.get("NLU_RETRIES", "1")))
NLU_RETRY_DELAY_MS = max(0, int(os.environ.get("NLU_RETRY_DELAY_MS", "150")))
_NLU_SESSION = None


async def call_nlu(text: str) -> dict:
    attempts = NLU_RETRIES + 1

    for attempt in range(1, attempts + 1):
        try:
            session = await _get_nlu_session()
            async with session.post(NLU_URL, json={"text": text}) as resp:
                if resp.status != 200:
                    log.error("NLU service returned %d (attempt %d/%d)", resp.status, attempt, attempts)
                else:
                    result = await resp.json()
                    log.info("NLU result: intent=%s slots=%s", result.get("intent"), result.get("slots"))
                    return result
        except Exception as exc:
            log.error("NLU call failed (attempt %d/%d): %s", attempt, attempts, exc)

        if attempt < attempts and NLU_RETRY_DELAY_MS > 0:
            await asyncio.sleep(NLU_RETRY_DELAY_MS / 1000.0)

    return {"intent": "unknown", "slots": {}, "raw": text}


async def _get_nlu_session():
    global _NLU_SESSION
    if _NLU_SESSION is None or _NLU_SESSION.closed:
        timeout = aiohttp.ClientTimeout(total=NLU_TIMEOUT)
        _NLU_SESSION = aiohttp.ClientSession(timeout=timeout)
    return _NLU_SESSION


async def close_nlu_session():
    global _NLU_SESSION
    if _NLU_SESSION is not None and not _NLU_SESSION.closed:
        await _NLU_SESSION.close()
    _NLU_SESSION = None
