"""
nlu/engine.py

ONNX NLU inference engine.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PREFIX = "指令解析: "
DEFAULT_MAX_INPUT = 64
DEFAULT_MAX_OUTPUT = 128
SESSION_EXPORT_CONFIGS: Dict[int, dict] = {}
STRUCTURED_STOP_CHARS = {"}", "]"}
STRUCTURED_EARLY_STOP = os.environ.get("NLU_STRUCTURED_EARLY_STOP", "0") not in {
    "0",
    "false",
    "False",
    "no",
}


def _load_export_config(model_dir: str) -> dict:
    path = Path(model_dir) / "export_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read NLU export config: %s", path, exc_info=True)
        return {}


def load_sessions(model_dir: str, use_gpu: bool = False, num_threads: int = None):
    """Load encoder/decoder ONNX sessions."""
    if num_threads is None:
        cores = os.cpu_count() or 4
        max_threads = int(os.environ.get("NLU_MAX_THREADS", "4"))
        num_threads = min(max(2, cores - 1), max_threads)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            logger.warning("CUDAExecutionProvider unavailable; NLU falling back to CPU")

    enc = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), sess_options=opts, providers=providers)
    dec = ort.InferenceSession(os.path.join(model_dir, "decoder.onnx"), sess_options=opts, providers=providers)
    export_config = _load_export_config(model_dir)
    SESSION_EXPORT_CONFIGS[id(enc)] = export_config
    SESSION_EXPORT_CONFIGS[id(dec)] = export_config
    logger.info("NLU model loaded (provider: %s)", enc.get_providers()[0])
    return enc, dec


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer from the exported model package."""
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def predict(enc_sess, dec_sess, tokenizer, text: str) -> str:
    """Run NLU inference and return raw model output text."""
    export_config = SESSION_EXPORT_CONFIGS.get(id(enc_sess), {})
    max_input = int(export_config.get("max_seq_len") or DEFAULT_MAX_INPUT)
    max_output = int(export_config.get("max_target_len") or DEFAULT_MAX_OUTPUT)
    decoder_start_token_id = int(
        export_config.get("decoder_start_token_id")
        if export_config.get("decoder_start_token_id") is not None
        else tokenizer.pad_token_id
    )
    eos_token_id = int(
        export_config.get("eos_token_id")
        if export_config.get("eos_token_id") is not None
        else tokenizer.eos_token_id
    )

    inputs = tokenizer(
        PREFIX + text,
        return_tensors="np",
        max_length=max_input,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

    enc_out = enc_sess.run(
        ["last_hidden_state"],
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]

    dec_ids = np.array([[decoder_start_token_id]], dtype=np.int64)
    enc_mask = inputs["attention_mask"].astype(np.int64)
    generated_ids = []

    for _ in range(max_output):
        logits = dec_sess.run(
            ["logits"],
            {
                "decoder_input_ids": dec_ids,
                "encoder_hidden_states": enc_out,
                "encoder_attention_mask": enc_mask,
            },
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == eos_token_id:
            break
        generated_ids.append(next_id)
        dec_ids = np.concatenate([dec_ids, [[next_id]]], axis=1)
        if STRUCTURED_EARLY_STOP and _should_stop_structured_decode(tokenizer, generated_ids):
            break

    return tokenizer.decode(dec_ids[0], skip_special_tokens=True)


def _should_stop_structured_decode(tokenizer, token_ids) -> bool:
    if not token_ids:
        return False
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if not text or text[-1] not in STRUCTURED_STOP_CHARS:
        return False
    return _loads_json(text) is not None


def parse_nlu_output(raw_output: str) -> dict:
    """
    Parse raw model output into the legacy pipeline shape:
    {"intent": str, "slots": dict, "raw": str}.

    Newer models may return command JSON:
    {"type":"cmd","payload":{"command_type":"Move","payload_json":{...}}}
    In that case we keep the original command under "command" and expose a
    compatible intent/slots pair for the existing dispatcher.
    """
    raw_output = (raw_output or "").strip()
    parsed = _loads_json(raw_output)
    if isinstance(parsed, dict):
        converted = _parse_json_output(parsed, raw_output)
        if converted is not None:
            return converted

    if "=" in raw_output:
        parts = [p.strip() for p in raw_output.replace(";", ",").split(",")]
        kv = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                kv[key.strip()] = value.strip()
        intent = kv.pop("intent", "unknown")
        return {"intent": intent, "slots": kv, "raw": raw_output}

    return {"intent": "unknown", "slots": {}, "raw": raw_output}


def _parse_json_output(parsed: dict, raw_output: str) -> Optional[dict]:
    if parsed.get("type") == "cmd":
        payload = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else {}
        command_type = str(payload.get("command_type") or "").strip()
        payload_json = payload.get("payload_json") if isinstance(payload.get("payload_json"), dict) else {}
        intent = _command_type_to_intent(command_type, payload_json)
        slots = _augment_slots(dict(payload_json), intent, command_type)
        if command_type:
            slots.setdefault("command_type", command_type)
        return {
            "intent": intent,
            "slots": slots,
            "raw": raw_output,
            "command": parsed,
            "source": "model",
        }

    if parsed.get("type") == "chat":
        return {
            "intent": "unknown",
            "slots": {},
            "raw": raw_output,
            "message": _dig(parsed, "payload", "message"),
            "command": parsed,
            "source": "chat",
        }

    if "intent" in parsed or "slots" in parsed:
        slots = parsed.get("slots")
        if not isinstance(slots, dict):
            if slots is None:
                slots = {k: v for k, v in parsed.items() if k != "intent"}
            else:
                slots = {"value": slots}
        return {
            "intent": parsed.get("intent", "unknown"),
            "slots": slots,
            "raw": raw_output,
        }

    return None


def _command_type_to_intent(command_type: str, payload_json: Optional[dict] = None) -> str:
    normalized = "".join(ch.lower() for ch in command_type if ch.isalnum())
    mapping = {
        "moveforward": "move_forward",
        "movebackward": "move_backward",
        "moveleft": "move_left",
        "moveright": "move_right",
        "turnleft": "turn_left",
        "turnright": "turn_right",
        "sit": "sit_down",
        "sitdown": "sit_down",
        "standdown": "sit_down",
        "stand": "stand_up",
        "standup": "stand_up",
        "risesit": "stand_up",
        "recoverystand": "stand_up",
        "balancestand": "stand_up",
        "liedown": "lie_down",
        "greet": "greet",
        "shakebody": "shake_body",
        "stretch": "stretch",
        "damp": "stop",
        "stop": "stop",
        "stopmove": "stop",
        "error": "unknown",
    }
    if normalized == "move":
        return _infer_move_intent(payload_json or {})
    return mapping.get(normalized, normalized or "unknown")


def _infer_move_intent(payload_json: dict) -> str:
    vx = float(payload_json.get("vx", 0.0) or 0.0)
    vy = float(payload_json.get("vy", 0.0) or 0.0)
    vyaw = float(payload_json.get("vyaw", 0.0) or 0.0)
    epsilon = 1e-6
    if abs(vyaw) > max(abs(vx), abs(vy), epsilon):
        return "turn_left" if vyaw > 0 else "turn_right"
    if abs(vy) > max(abs(vx), epsilon):
        return "move_left" if vy > 0 else "move_right"
    if abs(vx) > epsilon:
        return "move_forward" if vx > 0 else "move_backward"
    return "move"


def _augment_slots(slots: dict, intent: str, command_type: str) -> dict:
    if intent == "move_forward":
        slots.setdefault("direction", "forward")
    elif intent == "move_backward":
        slots.setdefault("direction", "backward")
    elif intent == "move_left":
        slots.setdefault("direction", "left")
    elif intent == "move_right":
        slots.setdefault("direction", "right")
    elif intent == "turn_left":
        slots.setdefault("direction", "left")
    elif intent == "turn_right":
        slots.setdefault("direction", "right")
    if command_type:
        slots.setdefault("command_type", command_type)
    return slots


def _loads_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _dig(data: dict, *keys: str) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
