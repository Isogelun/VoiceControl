"""
pipeline/nlu_client.py

NLU HTTP 客户端 — 调用 nlu/server.py 提供的 HTTP 服务。

环境变量:
    NLU_URL   NLU 服务地址，默认 http://localhost:8001/nlu
"""

import os
import logging

import aiohttp

log = logging.getLogger(__name__)

NLU_URL = os.environ.get("NLU_URL", "http://localhost:8001/nlu")


async def call_nlu(text: str) -> dict:
    """调用 NLU HTTP 服务进行意图识别和槽位填充"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                NLU_URL,
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.error("NLU 服务返回 %d", resp.status)
                    return {"intent": "unknown", "slots": {}, "raw": text}
                result = await resp.json()
                log.info("NLU 结果: intent=%s slots=%s", result.get("intent"), result.get("slots"))
                return result
    except Exception as e:
        log.error("NLU 调用失败: %s", e)
        return {"intent": "unknown", "slots": {}, "raw": text}
