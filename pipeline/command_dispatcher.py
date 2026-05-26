"""
Command dispatch and feedback.

The dispatcher is intentionally conservative: it persists every parsed command,
optionally forwards it to a later robot/action service, and plays success or
failure feedback without letting feedback errors crash the voice pipeline.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMAND_OUTPUT_DIR = Path(os.environ.get("COMMAND_OUTPUT_DIR", _PROJECT_ROOT / "output"))
COMMAND_SERVICE_URL = os.environ.get("COMMAND_SERVICE_URL", "").strip()
COMMAND_SERVICE_TIMEOUT = float(os.environ.get("COMMAND_SERVICE_TIMEOUT", "5"))
COMMAND_SUCCESS_AUDIO = os.environ.get(
    "COMMAND_SUCCESS_AUDIO",
    str(_PROJECT_ROOT / "audio" / "xuanxinghuida.mp3"),
)
COMMAND_FAILED_AUDIO = os.environ.get(
    "COMMAND_FAILED_AUDIO",
    str(_PROJECT_ROOT / "audio" / "command_failed.wav"),
)


class CommandDispatcher:
    def __init__(self, speaker=None):
        self.speaker = speaker

    async def dispatch(
        self,
        command: dict,
        asr_text: str,
        normalized_text: str,
        wake_metadata: dict = None,
    ) -> dict:
        envelope = self._make_envelope(command, asr_text, normalized_text, wake_metadata)

        if not self._is_actionable(command):
            envelope["status"] = "rejected"
            envelope["reason"] = "unknown_intent"
            self._persist(envelope)
            log.warning("指令无法解析: %s", json.dumps(envelope, ensure_ascii=False))
            await self.play_failure()
            return envelope

        self._persist(envelope)

        if COMMAND_SERVICE_URL:
            try:
                service_result = await self._post_to_service(envelope)
            except Exception as exc:
                envelope["status"] = "failed"
                envelope["reason"] = "service_error"
                envelope["error"] = str(exc)
                self._persist(envelope)
                log.exception("指令服务调用失败")
                await self.play_failure()
                return envelope

            envelope["service_result"] = service_result
            if not self._service_ok(service_result):
                envelope["status"] = "failed"
                envelope["reason"] = "service_rejected"
                self._persist(envelope)
                log.warning("指令服务拒绝/无法完成: %s", json.dumps(service_result, ensure_ascii=False))
                await self.play_failure()
                return envelope

            envelope["status"] = "completed"
            self._persist(envelope)
            await self.play_success()
            return envelope

        envelope["status"] = "accepted"
        envelope["reason"] = "no_command_service_configured"
        self._persist(envelope)
        log.info("未配置 COMMAND_SERVICE_URL，指令已落盘等待后续服务接入")
        await self.play_success()
        return envelope

    async def play_success(self):
        await self._play_feedback(COMMAND_SUCCESS_AUDIO, success=True)

    async def play_failure(self):
        await self._play_feedback(COMMAND_FAILED_AUDIO, success=False)

    def _make_envelope(
        self,
        command: dict,
        asr_text: str,
        normalized_text: str,
        wake_metadata: dict = None,
    ) -> dict:
        now = time.time()
        return {
            "id": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "status": "pending",
            "wake": wake_metadata or {},
            "asr_text": asr_text,
            "normalized_text": normalized_text,
            "command": command,
        }

    def _persist(self, envelope: dict):
        output_dir = Path(COMMAND_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        latest = output_dir / "latest_command.json"
        history = output_dir / f"{envelope['created_at'].replace(':', '-')}_{envelope['id']}.json"
        data = json.dumps(envelope, ensure_ascii=False, indent=2)
        latest.write_text(data, encoding="utf-8")
        history.write_text(data, encoding="utf-8")

    def _is_actionable(self, command: dict) -> bool:
        intent = (command or {}).get("intent")
        return bool(intent and intent != "unknown")

    async def _post_to_service(self, envelope: dict) -> dict:
        timeout = aiohttp.ClientTimeout(total=COMMAND_SERVICE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(COMMAND_SERVICE_URL, json=envelope) as resp:
                text = await resp.text()
                result = {
                    "http_status": resp.status,
                    "body": text,
                }
                try:
                    result["json"] = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    pass
                return result

    def _service_ok(self, result: dict) -> bool:
        status = int(result.get("http_status") or 0)
        if status < 200 or status >= 300:
            return False
        payload = result.get("json")
        if not isinstance(payload, dict):
            return True
        if payload.get("ok") is False or payload.get("success") is False:
            return False
        if str(payload.get("status", "")).lower() in {"failed", "error", "rejected", "unsupported"}:
            return False
        return True

    async def _play_feedback(self, path: str, success: bool):
        if path and os.path.isfile(path):
            if self.speaker:
                try:
                    await self.speaker.play_file(path)
                    return
                except Exception:
                    log.exception("机器人音频反馈播放失败: %s", path)
            try:
                await _play_local_file(path)
                return
            except Exception:
                log.exception("本地音频反馈播放失败: %s", path)

        log.warning("反馈音频不存在或无法播放: %s", path)
        if not self.speaker:
            await _play_local_beep(success=success)


async def _play_local_file(path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _play_local_file_sync, path)


def _play_local_file_sync(path: str):
    import sounddevice as sd
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    sd.play(data, sample_rate, blocking=True)


async def _play_local_beep(success: bool):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _play_local_beep_sync, success)


def _play_local_beep_sync(success: bool):
    import numpy as np
    import sounddevice as sd

    sample_rate = 16000
    if success:
        freqs = [880]
        duration = 0.18
    else:
        freqs = [220, 180]
        duration = 0.16

    chunks = []
    for freq in freqs:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        chunks.append((0.18 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
        chunks.append(np.zeros(int(sample_rate * 0.06), dtype=np.float32))
    sd.play(np.concatenate(chunks), sample_rate, blocking=True)
