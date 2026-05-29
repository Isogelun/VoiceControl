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
MOVE_STEP_TIMEOUT_MS = int(os.environ.get("MOVE_STEP_TIMEOUT_MS", "1200"))
MOVE_DEFAULT_TIMEOUT_MS = int(os.environ.get("MOVE_DEFAULT_TIMEOUT_MS", "1200"))
AUTO_STAND_BEFORE_MOVE = os.environ.get("AUTO_STAND_BEFORE_MOVE", "1") not in {"0", "false", "False", "no", ""}
MOVE_PREPARE_DELAY_MS = int(os.environ.get("MOVE_PREPARE_DELAY_MS", "1200"))
MOVE_REPEAT_COUNT = max(1, int(os.environ.get("MOVE_REPEAT_COUNT", "5")))
MOVE_REPEAT_INTERVAL_MS = max(0, int(os.environ.get("MOVE_REPEAT_INTERVAL_MS", "150")))
MOVE_LINEAR_SPEED = float(os.environ.get("MOVE_LINEAR_SPEED", "0.2"))
MOVE_YAW_SPEED = float(os.environ.get("MOVE_YAW_SPEED", "0.5"))
MOVE_CONTINUOUS = os.environ.get("MOVE_CONTINUOUS", "1") not in {"0", "false", "False", "no", ""}
MOVE_CONTINUOUS_FIELDS = [
    field.strip()
    for field in os.environ.get("MOVE_CONTINUOUS_FIELDS", "continous_move,continuous_move").split(",")
    if field.strip()
]
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
    "greet": "greet",
    "shake_body": "shake_body",
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
ACTION_AUDIO_MAP = {}


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
            await self.play_unavailable()
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
        await self._play_feedback(self._select_success_audio(command), success=True)

    async def play_failure(self):
        await self._play_feedback(COMMAND_FAILED_AUDIO, success=False)

    async def play_unavailable(self):
        await self._play_feedback(COMMAND_UNAVAILABLE_AUDIO, success=False)

    async def play_audio(self, path: str, success: bool = True):
        await self._play_feedback(path, success=success)

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
        output_dir = Path(COMMAND_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        latest = output_dir / "latest_command.json"
        history = output_dir / f"{envelope['created_at'].replace(':', '-')}_{envelope['id']}.json"
        data = json.dumps(envelope, ensure_ascii=False, indent=2)
        latest.write_text(data, encoding="utf-8")
        history.write_text(data, encoding="utf-8")

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

        return await self._post_payload(payload)

    async def _post_move_sequence(self, payload: dict) -> dict:
        sequence = []
        if AUTO_STAND_BEFORE_MOVE:
            prepare_payload = self._inherit_voice_metadata({"command_type": "stand_up"}, payload)
            prepare_result = await self._post_payload(prepare_payload)
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

        repeat_count = self._move_repeat_count(payload)
        move_result = {}
        for index in range(repeat_count):
            log.info("Move repeat %s/%s", index + 1, repeat_count)
            move_result = await self._post_payload(payload)
            sequence.append(dict(move_result))
            if not self._service_ok(move_result):
                move_result["sequence"] = sequence
                return move_result
            if index < repeat_count - 1 and MOVE_REPEAT_INTERVAL_MS > 0:
                await asyncio.sleep(MOVE_REPEAT_INTERVAL_MS / 1000.0)

        if MOVE_STOP_AFTER_TIMEOUT:
            stop_result = await self._post_payload(self._inherit_voice_metadata({"command_type": "stop"}, payload))
            sequence.append(dict(stop_result))
            if not self._service_ok(stop_result):
                stop_result["sequence"] = sequence
                return stop_result

        move_result["sequence"] = sequence
        return move_result

    def _move_repeat_count(self, payload: dict) -> int:
        try:
            timeout_ms = int(payload.get("timeout_ms") or MOVE_DEFAULT_TIMEOUT_MS)
        except (TypeError, ValueError):
            timeout_ms = MOVE_DEFAULT_TIMEOUT_MS
        if timeout_ms <= 0 or MOVE_REPEAT_INTERVAL_MS <= 0:
            return MOVE_REPEAT_COUNT
        return max(MOVE_REPEAT_COUNT, (timeout_ms + MOVE_REPEAT_INTERVAL_MS - 1) // MOVE_REPEAT_INTERVAL_MS)

    async def _post_payload(self, payload: dict) -> dict:
        timeout = aiohttp.ClientTimeout(total=COMMAND_SERVICE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            log.info("Posting motion command to %s: %s", COMMAND_SERVICE_URL, json.dumps(payload, ensure_ascii=False))
            async with session.post(COMMAND_SERVICE_URL, json=payload) as resp:
                text = await resp.text()
                result = {
                    "http_status": resp.status,
                    "body": text,
                    "request_json": payload,
                }
                try:
                    result["json"] = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    pass
                return result

    def _make_service_payload(self, envelope: dict) -> dict:
        command = envelope.get("command") if isinstance(envelope, dict) else {}
        if not isinstance(command, dict):
            return {}

        command_type = self._extract_motion_command_type(command)
        payload = {"command_type": command_type}

        payload_json = self._extract_payload_json(command)
        if payload_json:
            payload.update(payload_json)
        payload.update(_service_voice_metadata(envelope))
        if command_type in MOVE_COMMAND_TYPES:
            payload = self._normalize_move_payload(payload, command, envelope)
        return payload

    def _normalize_move_payload(self, payload: dict, command: dict, envelope: dict) -> dict:
        out = dict(payload)
        original_command_type = str(out.get("command_type") or "")
        out["command_type"] = "move"
        if "vyaw" in out and "wz" not in out:
            out["wz"] = out.pop("vyaw")
        out.setdefault("vx", 0)
        out.setdefault("vy", 0)
        out.setdefault("wz", 0)

        slots = command.get("slots") if isinstance(command.get("slots"), dict) else {}
        self._apply_default_move_velocity(out, original_command_type, command.get("intent"), slots.get("direction"))
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
        if "timeout_ms" not in out:
            out["timeout_ms"] = max(1, steps) * MOVE_STEP_TIMEOUT_MS if steps else MOVE_DEFAULT_TIMEOUT_MS
        if MOVE_CONTINUOUS:
            for field in MOVE_CONTINUOUS_FIELDS:
                out[field] = True
        out["payload_json"] = {
            "vx": out["vx"],
            "vy": out["vy"],
            "wz": out["wz"],
            "timeout_ms": out["timeout_ms"],
        }
        if MOVE_CONTINUOUS:
            for field in MOVE_CONTINUOUS_FIELDS:
                out["payload_json"][field] = True
        return out

    def _inherit_voice_metadata(self, payload: dict, source_payload: dict) -> dict:
        out = dict(payload)
        for key in ("voice_angle", "voice_raw_angle", "voice_direction", "voice_speech_detected", "voice_doa_source"):
            if key in source_payload:
                out[key] = source_payload[key]
        return out

    def _apply_default_move_velocity(self, out: dict, command_type: str, intent: str, direction: str):
        try:
            has_velocity = any(abs(float(out.get(key, 0) or 0)) > 1e-6 for key in ("vx", "vy", "wz"))
        except (TypeError, ValueError):
            has_velocity = False
        if has_velocity:
            return

        motion = str(command_type or intent or "").lower()
        direction = str(direction or "").lower()
        if motion.endswith("forward") or direction == "forward":
            out["vx"] = MOVE_LINEAR_SPEED
        elif motion.endswith("backward") or direction == "backward":
            out["vx"] = -MOVE_LINEAR_SPEED
        elif motion.endswith("left") or direction == "left":
            if "turn" in motion:
                out["wz"] = MOVE_YAW_SPEED
            else:
                out["vy"] = MOVE_LINEAR_SPEED
        elif motion.endswith("right") or direction == "right":
            if "turn" in motion:
                out["wz"] = -MOVE_YAW_SPEED
            else:
                out["vy"] = -MOVE_LINEAR_SPEED

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


def _service_voice_metadata(envelope: dict) -> dict:
    mic = envelope.get("microphone") if isinstance(envelope, dict) else {}
    if not isinstance(mic, dict) or not mic:
        return {}
    mapping = {
        "angle": "voice_angle",
        "raw_angle": "voice_raw_angle",
        "angle_direction": "voice_direction",
        "speech_detected": "voice_speech_detected",
        "doa_source": "voice_doa_source",
    }
    return {
        dst_key: mic[src_key]
        for src_key, dst_key in mapping.items()
        if src_key in mic and mic[src_key] is not None
    }


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
