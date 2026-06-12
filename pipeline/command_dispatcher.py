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
import re
import time
import uuid
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMAND_OUTPUT_DIR = Path(os.environ.get("COMMAND_OUTPUT_DIR", _PROJECT_ROOT / "output"))
COMMAND_SERVICE_URL = os.environ.get("COMMAND_SERVICE_URL", "").strip()
COMMAND_SERVICE_TIMEOUT = float(os.environ.get("COMMAND_SERVICE_TIMEOUT", "5"))
COMMAND_FAST_RESPONSE = os.environ.get("COMMAND_FAST_RESPONSE", "0") not in {"0", "false", "False", "no", ""}
COMMAND_FEEDBACK_ASYNC = os.environ.get("COMMAND_FEEDBACK_ASYNC", "1") not in {"0", "false", "False", "no", ""}
MOVE_STEP_TIMEOUT_MS = int(os.environ.get("MOVE_STEP_TIMEOUT_MS", "1200"))
MOVE_DEFAULT_TIMEOUT_MS = int(os.environ.get("MOVE_DEFAULT_TIMEOUT_MS", "1200"))
AUTO_STAND_BEFORE_MOVE = os.environ.get("AUTO_STAND_BEFORE_MOVE", "1") not in {"0", "false", "False", "no", ""}
MOVE_PREPARE_DELAY_MS = int(os.environ.get("MOVE_PREPARE_DELAY_MS", "1200"))
MOVE_LINEAR_SPEED = float(os.environ.get("MOVE_LINEAR_SPEED", "0.2"))
MOVE_YAW_SPEED = float(os.environ.get("MOVE_YAW_SPEED", "0.5"))
MOVE_PRIME_TIMEOUT_MS = int(os.environ.get("MOVE_PRIME_TIMEOUT_MS", "0"))
MOVE_POST_MOVE_DELAY_MS = int(os.environ.get("MOVE_POST_MOVE_DELAY_MS", "0"))
MOVE_NATIVE_ENABLED = os.environ.get("MOVE_NATIVE_ENABLED", "1") not in {"0", "false", "False", "no", ""}
MOVE_NATIVE_DEFAULT_STEPS = max(1, int(os.environ.get("MOVE_NATIVE_DEFAULT_STEPS", "3")))
MOVE_NATIVE_MIN_STEPS = max(1, int(os.environ.get("MOVE_NATIVE_MIN_STEPS", "1")))
MOVE_NATIVE_TIMEOUT_MS = max(1, int(os.environ.get("MOVE_NATIVE_TIMEOUT_MS", "1000")))
MOVE_NATIVE_LINEAR_SPEED = float(os.environ.get("MOVE_NATIVE_LINEAR_SPEED", "1"))
MOVE_NATIVE_YAW_SPEED = float(os.environ.get("MOVE_NATIVE_YAW_SPEED", "1"))
MOVE_FAST_RESPONSE = os.environ.get("MOVE_FAST_RESPONSE", "0") not in {"0", "false", "False", "no", ""}
MOVE_FAST_NATIVE_FIRST = os.environ.get("MOVE_FAST_NATIVE_FIRST", "1") not in {"0", "false", "False", "no", ""}
MOVE_FAST_FOLLOWUP_MOVE = os.environ.get("MOVE_FAST_FOLLOWUP_MOVE", "1") not in {"0", "false", "False", "no", ""}
MOVE_FAST_FOLLOWUP_DELAY_MS = max(0, int(os.environ.get("MOVE_FAST_FOLLOWUP_DELAY_MS", "80")))
MOVE_FAST_AUTO_STAND = os.environ.get("MOVE_FAST_AUTO_STAND", "0") not in {"0", "false", "False", "no", ""}
MOVE_STOP_AFTER_TIMEOUT = os.environ.get("MOVE_STOP_AFTER_TIMEOUT", "1") not in {"0", "false", "False", "no", ""}
COMMAND_SUCCESS_AUDIO = os.environ.get(
    "COMMAND_SUCCESS_AUDIO",
    str(_PROJECT_ROOT / "audio" / "xuanxinghuida.mp3"),
)
COMMAND_FAILED_AUDIO = os.environ.get(
    "COMMAND_FAILED_AUDIO",
    str(_PROJECT_ROOT / "audio" / "command_failed.wav"),
)
COMMAND_UNAVAILABLE_AUDIO = os.environ.get("COMMAND_UNAVAILABLE_AUDIO", COMMAND_FAILED_AUDIO)
COMMAND_ACTION_AUDIO = os.environ.get("COMMAND_ACTION_AUDIO", "").strip()
SUPPORTED_INTENTS = {
    "stop",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "stand_up",
    "sit_down",
    "lie_down",
    "greet",
    "shake_body",
    "stretch",
}
MOTION_COMMAND_TYPES = {
    "Move": "move",
    "MoveForward": "move_forward",
    "MoveBackward": "move_backward",
    "MoveLeft": "move_left",
    "MoveRight": "move_right",
    "TurnLeft": "turn_left",
    "TurnRight": "turn_right",
    "StandUp": "stand_up",
    "StandDown": "stand_down",
    "Sit": "sit",
    "LieDown": "lie_down",
    "Stop": "stop",
    "StopMove": "stop",
    "Greet": "greet",
    "ShakeBody": "shake_body",
    "Stretch": "stretch",
}
INTENT_COMMAND_TYPES = {
    "move_forward": "move_forward",
    "move_backward": "move_backward",
    "move_left": "move_left",
    "move_right": "move_right",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
    "stand_up": "stand_up",
    "sit_down": "stand_down",
    "lie_down": "lie_down",
    "stop": "stop",
    "greet": "hello",
    "shake_body": "wiggle_hips",
    "stretch": "stretch",
}
MOVE_COMMAND_TYPES = {
    "move",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
}
DIRECTIONAL_MOVE_TYPES = MOVE_COMMAND_TYPES - {"move"}
NATIVE_MOVE_PAYLOAD_KEY = "_native_move_payload"
ACTION_AUDIO_MAP = {}
BACKGROUND_POST_TASKS = set()
BACKGROUND_FEEDBACK_TASKS = set()


class CommandDispatcher:
    def __init__(self, speaker=None):
        self.speaker = speaker
        self._service_session = None

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
            await self.play_unavailable()
            return envelope

        self._persist(envelope)

        if COMMAND_SERVICE_URL:
            try:
                service_result = await self._post_to_service(envelope)
            except Exception as exc:
                envelope["status"] = "failed"
                envelope["reason"] = "service_error"
                envelope["error"] = _describe_error(exc)
                self._persist(envelope)
                log.exception("指令服务调用失败")
                await self.play_unavailable()
                return envelope

            envelope["service_result"] = service_result
            if not self._service_ok(service_result):
                envelope["status"] = "failed"
                envelope["reason"] = "service_rejected"
                self._persist(envelope)
                log.warning("指令服务拒绝/无法完成: %s", json.dumps(service_result, ensure_ascii=False))
                await self.play_unavailable()
                return envelope

            envelope["status"] = "completed"
            self._persist(envelope)
            await self.play_success(command)
            return envelope

        envelope["status"] = "accepted"
        envelope["reason"] = "no_command_service_configured"
        self._persist(envelope)
        log.info("未配置 COMMAND_SERVICE_URL，指令已落盘等待后续服务接入")
        await self.play_success(command)
        return envelope

    async def play_success(self, command: dict = None):
        await self._maybe_play_feedback(self._select_success_audio(command), success=True)

    async def play_failure(self):
        await self._maybe_play_feedback(COMMAND_FAILED_AUDIO, success=False)

    async def play_unavailable(self):
        await self._maybe_play_feedback(COMMAND_UNAVAILABLE_AUDIO, success=False)

    async def play_audio(self, path: str, success: bool = True):
        await self._maybe_play_feedback(path, success=success)

    def _select_success_audio(self, command: dict = None) -> str:
        audio_map = _action_audio_map()
        if isinstance(command, dict):
            intent = str(command.get("intent") or "").strip()
            command_type = self._extract_motion_command_type(command)
            for key in (intent, command_type):
                if key and audio_map.get(key):
                    return audio_map[key]
        return audio_map.get("default") or COMMAND_SUCCESS_AUDIO

    def _make_envelope(
        self,
        command: dict,
        asr_text: str,
        normalized_text: str,
        wake_metadata: dict = None,
    ) -> dict:
        now = time.time()
        wake = wake_metadata or {}
        microphone = _extract_microphone_metadata(wake)
        return {
            "id": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "status": "pending",
            "wake": wake,
            "microphone": microphone,
            "audio_direction": _audio_direction_metadata(microphone),
            "asr_text": asr_text,
            "normalized_text": normalized_text,
            "command": command,
        }

    def _persist(self, envelope: dict):
        # 落盘失败（磁盘满/权限/路径非法）不应中断指令分发主流程。
        try:
            output_dir = Path(COMMAND_OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            latest = output_dir / "latest_command.json"
            history = output_dir / f"{envelope['created_at'].replace(':', '-')}_{envelope['id']}.json"
            data = json.dumps(envelope, ensure_ascii=False, indent=2)
            latest.write_text(data, encoding="utf-8")
            history.write_text(data, encoding="utf-8")
        except Exception:
            log.exception("指令落盘失败: id=%s", envelope.get("id") if isinstance(envelope, dict) else "?")

    def _is_actionable(self, command: dict) -> bool:
        if not isinstance(command, dict):
            return False
        intent = (command or {}).get("intent")
        if not intent or intent == "unknown" or intent not in SUPPORTED_INTENTS:
            return False

        slots = command.get("slots")
        if slots is not None and not isinstance(slots, dict):
            return False

        model_command = command.get("command")
        if model_command is None:
            return True
        if not isinstance(model_command, dict):
            return False
        if model_command.get("type") != "cmd":
            return False
        payload = model_command.get("payload")
        if not isinstance(payload, dict):
            return False
        command_type = payload.get("command_type")
        if not isinstance(command_type, str) or not command_type.strip():
            return False
        payload_json = payload.get("payload_json")
        return payload_json is None or isinstance(payload_json, dict)

    async def _post_to_service(self, envelope: dict) -> dict:
        payload = self._make_service_payload(envelope)
        if payload.get("command_type") in MOVE_COMMAND_TYPES:
            return await self._post_move_sequence(payload)

        if COMMAND_FAST_RESPONSE:
            return self._queue_fast_payload(payload)

        return await self._post_payload(payload)

    async def _post_move_sequence(self, payload: dict) -> dict:
        """按 test.py 的动作顺序发送：stand_up -> move -> 可选原生方向动作。"""
        native_payload = payload.get(NATIVE_MOVE_PAYLOAD_KEY) if isinstance(payload, dict) else None
        move_payload = self._public_payload(payload)
        if MOVE_FAST_RESPONSE:
            return self._queue_fast_move_sequence(move_payload, native_payload)

        timeout_ms = self._move_timeout_ms(move_payload)
        sequence = []

        # 1. 先站立
        if AUTO_STAND_BEFORE_MOVE:
            prepare_result = await self._post_payload({"command_type": "stand_up", "params": {}})
            sequence.append(prepare_result)
            if not self._service_ok(prepare_result):
                return {
                    "sequence": sequence,
                    "http_status": prepare_result.get("http_status"),
                    "request_json": payload,
                    "json": prepare_result.get("json"),
                    "body": prepare_result.get("body", ""),
                }
            if MOVE_PREPARE_DELAY_MS > 0:
                await asyncio.sleep(MOVE_PREPARE_DELAY_MS / 1000.0)

        # 2. 发一次 move 命令（服务端自己按 timeout_ms 控制时长）
        log.info("Posting move command (once, timeout=%sms): %s", timeout_ms,
                 json.dumps(move_payload, ensure_ascii=False))
        move_result = await self._post_payload(move_payload)
        sequence.append(dict(move_result))
        if not self._service_ok(move_result):
            move_result["sequence"] = sequence
            return move_result

        # 3. 等待 move 完成；如果有方向动作，继续补发原生 move_forward/move_backward 等命令。
        post_move_delay_ms = timeout_ms
        if isinstance(native_payload, dict) and native_payload and MOVE_POST_MOVE_DELAY_MS > 0:
            post_move_delay_ms = MOVE_POST_MOVE_DELAY_MS
        await asyncio.sleep(post_move_delay_ms / 1000.0)
        if isinstance(native_payload, dict) and native_payload:
            native_result = await self._post_payload(native_payload)
            sequence.append(dict(native_result))
            if not self._service_ok(native_result):
                native_result["sequence"] = sequence
                return native_result
            native_timeout_ms = self._move_timeout_ms(native_payload)
            if native_timeout_ms > 0:
                await asyncio.sleep(native_timeout_ms / 1000.0)
            native_result["sequence"] = sequence
            return native_result

        # 4. 停
        if MOVE_STOP_AFTER_TIMEOUT:
            stop_result = await self._post_payload({"command_type": "stop", "params": {}})
            sequence.append(dict(stop_result))
            if not self._service_ok(stop_result):
                stop_result["sequence"] = sequence
                return stop_result

        move_result["sequence"] = sequence
        return move_result

    def _queue_fast_move_sequence(self, move_payload: dict, native_payload: dict = None) -> dict:
        """Queue motion commands without waiting for blocking motion-service responses."""
        initial_payload = native_payload if native_payload and MOVE_FAST_NATIVE_FIRST else move_payload
        followups = []
        if native_payload and initial_payload is native_payload and MOVE_FAST_FOLLOWUP_MOVE:
            followups.append((MOVE_FAST_FOLLOWUP_DELAY_MS, move_payload))
        elif native_payload and initial_payload is move_payload:
            followups.append((MOVE_FAST_FOLLOWUP_DELAY_MS, native_payload))

        if MOVE_FAST_AUTO_STAND and AUTO_STAND_BEFORE_MOVE:
            followups.insert(0, (0, {"command_type": "stand_up", "params": {}}))

        sequence = [{"queued": True, "request_json": initial_payload}]
        sequence.extend({"queued": True, "delay_ms": delay_ms, "request_json": payload} for delay_ms, payload in followups)
        task = asyncio.create_task(self._post_payload_sequence_background(initial_payload, followups))
        BACKGROUND_POST_TASKS.add(task)
        task.add_done_callback(BACKGROUND_POST_TASKS.discard)
        log.info("Queued fast motion sequence: %s", json.dumps(sequence, ensure_ascii=False))
        return {
            "http_status": 202,
            "queued": True,
            "request_json": initial_payload,
            "sequence": sequence,
        }

    def _queue_fast_payload(self, payload: dict) -> dict:
        task = asyncio.create_task(self._post_payload_background(payload))
        BACKGROUND_POST_TASKS.add(task)
        task.add_done_callback(BACKGROUND_POST_TASKS.discard)
        log.info("Queued fast command payload: %s", json.dumps(payload, ensure_ascii=False))
        return {
            "http_status": 202,
            "queued": True,
            "request_json": payload,
        }

    async def _post_payload_background(self, payload: dict):
        try:
            result = await self._post_payload(payload)
            log.info("Fast command background result: %s", json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            log.exception("Fast command background post failed: %s", _describe_error(exc))

    async def _post_payload_sequence_background(self, initial_payload: dict, followups: list):
        posted = []
        try:
            initial_result = await self._post_payload(initial_payload)
            posted.append(initial_result)
            for delay_ms, payload in followups:
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
                posted.append(await self._post_payload(payload))
            log.info("Fast motion background results: %s", json.dumps(posted, ensure_ascii=False))
        except Exception:
            log.exception("Fast motion background post failed")

    def _public_payload(self, payload: dict) -> dict:
        return {
            key: value
            for key, value in (payload or {}).items()
            if not str(key).startswith("_")
        }

    def _move_timeout_ms(self, payload: dict) -> int:
        try:
            params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
            return int(params.get("timeout_ms") or payload.get("timeout_ms") or MOVE_DEFAULT_TIMEOUT_MS)
        except (TypeError, ValueError):
            return MOVE_DEFAULT_TIMEOUT_MS

    async def _post_payload(self, payload: dict) -> dict:
        session = await self._get_service_session()
        started = time.perf_counter()
        log.info("Posting motion command to %s: %s", COMMAND_SERVICE_URL, json.dumps(payload, ensure_ascii=False))
        async with session.post(COMMAND_SERVICE_URL, json=payload) as resp:
            text = await resp.text()
            elapsed_ms = (time.perf_counter() - started) * 1000
            result = {
                "http_status": resp.status,
                "body": text,
                "request_json": payload,
                "elapsed_ms": round(elapsed_ms, 1),
            }
            try:
                result["json"] = json.loads(text) if text else {}
            except json.JSONDecodeError:
                pass
            log.info("Motion service response: status=%s elapsed=%.1fms", resp.status, elapsed_ms)
            return result

    async def _get_service_session(self):
        if getattr(self, "_service_session", None) is None or self._service_session.closed:
            timeout = aiohttp.ClientTimeout(total=COMMAND_SERVICE_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=4, keepalive_timeout=30)
            self._service_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._service_session

    async def close(self):
        if getattr(self, "_service_session", None) is not None and not self._service_session.closed:
            await self._service_session.close()
        self._service_session = None

    def _make_service_payload(self, envelope: dict) -> dict:
        command = envelope.get("command") if isinstance(envelope, dict) else {}
        if not isinstance(command, dict):
            return {}

        command_type = self._extract_motion_command_type(command)
        if command_type in MOVE_COMMAND_TYPES:
            return self._build_move_payload(command, envelope)
        return {"command_type": command_type, "params": {}}

    def _build_move_payload(self, command: dict, envelope: dict) -> dict:
        original_command_type = self._extract_motion_command_type(command)
        slots = command.get("slots") if isinstance(command.get("slots"), dict) else {}

        # 确定速度向量
        vx, vy, wz = 0.0, 0.0, 0.0
        payload_json = self._extract_payload_json(command)
        if payload_json:
            vx = float(payload_json.get("vx", 0) or 0)
            vy = float(payload_json.get("vy", 0) or 0)
            if "vyaw" in payload_json:
                wz = float(payload_json.get("vyaw", 0) or 0)
            elif "wz" in payload_json:
                wz = float(payload_json.get("wz", 0) or 0)

        # 如果模型没给速度，用默认值填充
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
            direction = str(slots.get("direction") or "").lower()
            intent = command.get("intent", "")
            vx, vy, wz = self._resolve_default_velocity(original_command_type, intent, direction)

        # 计算持续时长
        steps = slots.get("steps")
        if steps is None:
            steps = _extract_steps_from_text(
                envelope.get("normalized_text")
                or command.get("normalized")
                or command.get("raw")
                or envelope.get("asr_text")
                or ""
            )
        try:
            steps = int(steps) if steps is not None else None
        except (TypeError, ValueError):
            steps = None
        timeout_ms = max(1, steps) * MOVE_STEP_TIMEOUT_MS if steps else MOVE_DEFAULT_TIMEOUT_MS
        # 服务器最大只支持 2000ms，超过会被截断
        timeout_ms = min(timeout_ms, 2000)
        if original_command_type in DIRECTIONAL_MOVE_TYPES and MOVE_PRIME_TIMEOUT_MS > 0:
            timeout_ms = MOVE_PRIME_TIMEOUT_MS

        move_payload = {
            "command_type": "move",
            "params": {
                "vx": vx,
                "vy": vy,
                "wz": wz,
                "timeout_ms": timeout_ms,
            },
        }
        native_payload = self._build_native_move_payload(original_command_type, steps)
        if native_payload:
            move_payload[NATIVE_MOVE_PAYLOAD_KEY] = native_payload
        return move_payload

    def _build_native_move_payload(self, command_type: str, steps: int = None) -> dict:
        if not MOVE_NATIVE_ENABLED or command_type not in DIRECTIONAL_MOVE_TYPES:
            return {}

        try:
            step_count = int(steps) if steps is not None else MOVE_NATIVE_DEFAULT_STEPS
        except (TypeError, ValueError):
            step_count = MOVE_NATIVE_DEFAULT_STEPS
        step_count = max(MOVE_NATIVE_MIN_STEPS, step_count)

        params = {
            "step": step_count,
            "timeout_ms": MOVE_NATIVE_TIMEOUT_MS,
        }
        linear_speed = abs(MOVE_NATIVE_LINEAR_SPEED)
        yaw_speed = abs(MOVE_NATIVE_YAW_SPEED)
        if command_type == "move_forward":
            params["vx"] = linear_speed
        elif command_type == "move_backward":
            params["vx"] = -linear_speed
        elif command_type == "move_left":
            params["vy"] = linear_speed
        elif command_type == "move_right":
            params["vy"] = -linear_speed
        elif command_type == "turn_left":
            params["wz"] = yaw_speed
        elif command_type == "turn_right":
            params["wz"] = -yaw_speed

        return {"command_type": command_type, "params": params}

    def _resolve_default_velocity(self, command_type: str, intent: str, direction: str) -> tuple:
        """根据意图/方向推断默认速度向量 (vx, vy, wz)"""
        motion = str(command_type or intent or "").lower()
        direction = str(direction or "").lower()
        if motion.endswith("forward") or direction == "forward":
            return (MOVE_LINEAR_SPEED, 0.0, 0.0)
        elif motion.endswith("backward") or direction == "backward":
            return (-MOVE_LINEAR_SPEED, 0.0, 0.0)
        elif motion.endswith("left") or direction == "left":
            if "turn" in motion:
                return (0.0, 0.0, MOVE_YAW_SPEED)
            return (0.0, MOVE_LINEAR_SPEED, 0.0)
        elif motion.endswith("right") or direction == "right":
            if "turn" in motion:
                return (0.0, 0.0, -MOVE_YAW_SPEED)
            return (0.0, -MOVE_LINEAR_SPEED, 0.0)
        return (0.0, 0.0, 0.0)

    def _extract_motion_command_type(self, command: dict) -> str:
        slots = command.get("slots") if isinstance(command.get("slots"), dict) else {}
        raw_type = slots.get("command_type")
        if not raw_type:
            model_command = command.get("command")
            if isinstance(model_command, dict):
                payload = model_command.get("payload")
                if isinstance(payload, dict):
                    raw_type = payload.get("command_type")
        if isinstance(raw_type, str) and raw_type.strip():
            normalized = self._normalize_command_type(raw_type)
            if normalized:
                return normalized

        intent = command.get("intent")
        return INTENT_COMMAND_TYPES.get(str(intent), str(intent or "unknown"))

    def _extract_payload_json(self, command: dict) -> dict:
        model_command = command.get("command")
        if isinstance(model_command, dict):
            payload = model_command.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("payload_json"), dict):
                return dict(payload["payload_json"])
        slots = command.get("slots")
        if isinstance(slots, dict):
            return {
                key: value
                for key, value in slots.items()
                if key not in {"command_type", "direction", "steps", "angle"}
            }
        return {}

    def _normalize_command_type(self, command_type: str) -> str:
        if command_type in MOTION_COMMAND_TYPES:
            return MOTION_COMMAND_TYPES[command_type]
        chars = []
        for index, ch in enumerate(command_type.strip()):
            if ch in {"-", " ", "."}:
                chars.append("_")
            elif ch == "_":
                chars.append(ch)
            elif ch.isupper() and index > 0 and command_type[index - 1].islower():
                chars.extend(["_", ch.lower()])
            else:
                chars.append(ch.lower())
        return "".join(chars).strip("_")

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


    async def _maybe_play_feedback(self, path: str, success: bool):
        if not COMMAND_FEEDBACK_ASYNC:
            await self._play_feedback(path, success=success)
            return
        task = asyncio.create_task(self._play_feedback(path, success=success))
        BACKGROUND_FEEDBACK_TASKS.add(task)
        task.add_done_callback(BACKGROUND_FEEDBACK_TASKS.discard)


def _describe_error(exc: Exception) -> str:
    text = str(exc)
    if text:
        return f"{type(exc).__name__}: {text}"
    return type(exc).__name__


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


def _action_audio_map() -> dict:
    if not COMMAND_ACTION_AUDIO:
        return {}
    if ACTION_AUDIO_MAP:
        return ACTION_AUDIO_MAP

    try:
        data = json.loads(COMMAND_ACTION_AUDIO)
    except json.JSONDecodeError:
        data = _parse_action_audio_pairs(COMMAND_ACTION_AUDIO)
    if not isinstance(data, dict):
        log.warning("COMMAND_ACTION_AUDIO must be a JSON object or key=path list: %s", COMMAND_ACTION_AUDIO)
        return {}

    for key, value in data.items():
        if not key or not value:
            continue
        ACTION_AUDIO_MAP[str(key).strip()] = _resolve_audio_path(str(value).strip())
    return ACTION_AUDIO_MAP


def _parse_action_audio_pairs(text: str) -> dict:
    pairs = {}
    for part in text.split(";"):
        if not part.strip() or "=" not in part:
            continue
        key, value = part.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def _resolve_audio_path(path: str) -> str:
    if not path:
        return path
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return str(_PROJECT_ROOT / expanded)


def _extract_microphone_metadata(wake: dict) -> dict:
    if not isinstance(wake, dict):
        return {}
    keys = {
        "doa_source",
        "angle",
        "raw_angle",
        "speech_detected",
        "angle_direction",
        "updated_at",
    }
    return {key: wake[key] for key in keys if key in wake}



def _audio_direction_metadata(mic: dict) -> dict:
    if not isinstance(mic, dict) or not mic:
        return {}
    return {
        "angle": mic.get("angle"),
        "raw_angle": mic.get("raw_angle"),
        "direction": mic.get("angle_direction"),
        "speech_detected": mic.get("speech_detected"),
        "source": mic.get("doa_source"),
        "updated_at": mic.get("updated_at"),
    }


def _extract_steps_from_text(text: str):
    if not text:
        return None
    match = re.search(r"(\d+)\s*步", text)
    if match:
        return int(match.group(1))
    numbers = {
        "一": 1,
        "两": 2,
        "二": 2,
        "俩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    for word, value in numbers.items():
        if f"{word}步" in text:
            return value
    return None
