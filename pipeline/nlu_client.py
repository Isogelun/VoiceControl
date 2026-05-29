"""
NLU HTTP client.
"""

import logging
import os

import aiohttp

log = logging.getLogger(__name__)

NLU_URL = os.environ.get("NLU_URL", "http://localhost:8001/nlu")
_NLU_SESSION = None


async def call_nlu(text: str) -> dict:
    try:
        session = await _get_nlu_session()
        async with session.post(NLU_URL, json={"text": text}) as resp:
            if resp.status != 200:
                log.error("NLU service returned %d", resp.status)
                return {"intent": "unknown", "slots": {}, "raw": text}
            result = await resp.json()
            log.info("NLU result: intent=%s slots=%s", result.get("intent"), result.get("slots"))
            return result
    except Exception as exc:
        log.error("NLU call failed: %s", exc)
        return {"intent": "unknown", "slots": {}, "raw": text}


async def _get_nlu_session():
    global _NLU_SESSION
    if _NLU_SESSION is None or _NLU_SESSION.closed:
        timeout = aiohttp.ClientTimeout(total=10)
        _NLU_SESSION = aiohttp.ClientSession(timeout=timeout)
    return _NLU_SESSION


async def close_nlu_session():
    global _NLU_SESSION
    if _NLU_SESSION is not None and not _NLU_SESSION.closed:
        await _NLU_SESSION.close()
    _NLU_SESSION = None
