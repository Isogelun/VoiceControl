"""
nlu/engine.py

Mengzi-T5 ONNX 推理引擎。
纯推理逻辑，不含 HTTP 服务和 CLI。
"""

import json
import logging
import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PREFIX = "指令解析: "
MAX_INPUT = 64
MAX_OUTPUT = 128


# ─── 模型加载 ───────────────────────────────────────────────────────────────────

def load_sessions(model_dir: str):
    """加载 encoder/decoder ONNX 模型，返回 (enc_sess, dec_sess)"""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), providers=providers)
    dec = ort.InferenceSession(os.path.join(model_dir, "decoder.onnx"), providers=providers)
    logger.info("NLU 模型已加载 (provider: %s)", enc.get_providers()[0])
    return enc, dec


def load_tokenizer(tokenizer_dir: str):
    """加载分词器"""
    return AutoTokenizer.from_pretrained(tokenizer_dir)


# ─── 推理 ───────────────────────────────────────────────────────────────────────

def predict(enc_sess, dec_sess, tokenizer, text: str) -> str:
    """执行 NLU 推理，返回模型原始输出文本"""
    inputs = tokenizer(
        PREFIX + text, return_tensors="np",
        max_length=MAX_INPUT, padding="max_length", truncation=True,
    )
    inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    # encoder
    enc_out = enc_sess.run(
        ["last_hidden_state"],
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]

    # greedy decode
    dec_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    enc_mask = inputs["attention_mask"].astype(np.int64)

    for _ in range(MAX_OUTPUT):
        logits = dec_sess.run(
            ["logits"],
            {"decoder_input_ids": dec_ids,
             "encoder_hidden_states": enc_out,
             "encoder_attention_mask": enc_mask},
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == tokenizer.eos_token_id:
            break
        dec_ids = np.concatenate([dec_ids, [[next_id]]], axis=1)

    return tokenizer.decode(dec_ids[0], skip_special_tokens=True)


# ─── 输出解析 ───────────────────────────────────────────────────────────────────

def parse_nlu_output(raw_output: str) -> dict:
    """
    将模型原始输出解析为结构化 dict。
    尝试 JSON → 键值对 → 原样返回。
    返回: {"intent": str, "slots": dict, "raw": str}
    """
    raw_output = raw_output.strip()
    # 尝试直接 JSON 解析
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return {
                "intent": parsed.get("intent", "unknown"),
                "slots": parsed.get("slots", parsed),
                "raw": raw_output,
            }
    except (json.JSONDecodeError, ValueError):
        pass
    # 尝试解析类似 "intent=move_forward; direction=前; steps=3" 格式
    if "=" in raw_output:
        parts = [p.strip() for p in raw_output.replace(";", ",").split(",")]
        kv = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip()
        intent = kv.pop("intent", "unknown")
        return {"intent": intent, "slots": kv, "raw": raw_output}
    # 无法解析，原样返回
    return {"intent": "unknown", "slots": {}, "raw": raw_output}
