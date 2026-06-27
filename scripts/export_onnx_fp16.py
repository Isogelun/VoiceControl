"""
Export Qwen3-ASR-0.6B safetensors to FP16 ONNX models.

Produces:
  encoder.onnx        mel [1,128,T] -> audio_features [1,seq,1024]
  decoder_init.onnx   input_ids, position_ids, audio_features, audio_offset -> logits, past_keys, past_values
  decoder_step.onnx   input_embeds [1,1,1024], position_ids, past_keys, past_values -> logits, past_keys, past_values
  embed_tokens.bin    [151936, 1024] float16
  config.json
  tokenizer.json      (copied from source)

Compatible with ORT 1.17 / opset 17 / Python 3.8 on Jetson.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# ── Config ───────────────────────────────────────────────────────────────────

ENC_LAYERS = 18
ENC_DIM = 896
ENC_HEADS = 14
ENC_FFN = 3584
CONV_CH = 480
OUTPUT_DIM = 1024
N_MELS = 128

DEC_LAYERS = 28
DEC_DIM = 1024
DEC_HEADS = 16
DEC_KV_HEADS = 8
HEAD_DIM = 128
DEC_FFN = 3072
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000
RMS_EPS = 1e-6
MROPE_SECTIONS = [24, 20, 20]  # interleaved

OPSET = 17


# ── Audio Encoder ────────────────────────────────────────────────────────────

class EncoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = ENC_HEADS
        self.head_dim = ENC_DIM // ENC_HEADS
        self.q_proj = nn.Linear(ENC_DIM, ENC_DIM)
        self.k_proj = nn.Linear(ENC_DIM, ENC_DIM)
        self.v_proj = nn.Linear(ENC_DIM, ENC_DIM)
        self.out_proj = nn.Linear(ENC_DIM, ENC_DIM)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, ENC_DIM)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(ENC_DIM)
        self.self_attn = EncoderAttention()
        self.final_layer_norm = nn.LayerNorm(ENC_DIM)
        self.fc1 = nn.Linear(ENC_DIM, ENC_FFN)
        self.fc2 = nn.Linear(ENC_FFN, ENC_DIM)

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class AudioEncoder(nn.Module):
    """Whisper-like audio encoder with 3x stride-2 conv downsampling + transformer.

    All 3 convs have stride=2: mel 128->64->32->16 freq, T->T/2->T/4->T/8 time.
    conv_out input: CONV_CH * (N_MELS/8) = 480*16 = 7680. downsample_factor=8.
    """

    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, CONV_CH, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(CONV_CH, CONV_CH, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(CONV_CH, CONV_CH, 3, stride=2, padding=1)
        self.conv_out = nn.Linear(CONV_CH * (N_MELS // 8), ENC_DIM, bias=False)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(ENC_LAYERS)])
        self.ln_post = nn.LayerNorm(ENC_DIM)
        self.proj1 = nn.Linear(ENC_DIM, ENC_DIM)
        self.proj2 = nn.Linear(ENC_DIM, OUTPUT_DIM)

    def forward(self, mel):
        # mel: [B, n_mels, T]
        x = mel.unsqueeze(1)  # [B, 1, n_mels, T]
        x = F.gelu(self.conv2d1(x))  # [B, 480, 64, T/2] if stride=2
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [B, 480, n_mels/8, T/8] = [B, 480, 16, T/8]
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)  # [B, T/8, 7680]
        x = self.conv_out(x)  # [B, T/8, 896]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # [B, T/8, 1024]
        return x


# ── RoPE helpers ─────────────────────────────────────────────────────────────

def build_rope_cache(seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """Pre-compute cos/sin for MROPE (interleaved, sections [24,20,20])."""
    freqs_list = []
    for section_size in MROPE_SECTIONS:
        dim = section_size * 2
        freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs_list.append(freq)
    # For simplicity in ONNX export, we pass position_ids and compute rope inside.
    # But ONNX doesn't support complex ops well, so we'll precompute and pass cos/sin.
    return freqs_list


# ── Decoder Modules ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DecoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = DEC_HEADS
        self.num_kv_heads = DEC_KV_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(DEC_DIM, DEC_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DEC_DIM, DEC_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DEC_DIM, DEC_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(DEC_HEADS * HEAD_DIM, DEC_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, x, cos, sin, past_key=None, past_value=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key is not None:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        new_k = k
        new_v = v

        # GQA: repeat kv heads
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_k, new_v


def apply_rope(x, cos, sin):
    """Apply rotary embeddings (interleaved format)."""
    # x: [B, heads, T, head_dim=128], cos/sin: [1, 1, T, 64]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x1 = x[..., 0::2]  # [B, heads, T, 64]
    x2 = x[..., 1::2]  # [B, heads, T, 64]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class DecoderMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DEC_DIM, DEC_FFN, bias=False)
        self.up_proj = nn.Linear(DEC_DIM, DEC_FFN, bias=False)
        self.down_proj = nn.Linear(DEC_FFN, DEC_DIM, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(DEC_DIM)
        self.self_attn = DecoderAttention()
        self.post_attention_layernorm = RMSNorm(DEC_DIM)
        self.mlp = DecoderMLP()

    def forward(self, x, cos, sin, past_key=None, past_value=None):
        residual = x
        x = self.input_layernorm(x)
        x, new_k, new_v = self.self_attn(x, cos, sin, past_key, past_value)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, new_k, new_v


def compute_rope_cos_sin(position_ids, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """
    Compute cos/sin for MROPE with interleaved sections.
    position_ids: [B, T] int64
    Returns cos, sin: [1, 1, T, 64] matching input dtype context.
    """
    sections = MROPE_SECTIONS  # [24, 20, 20]
    pos = position_ids[0].float()  # always compute in FP32

    all_cos = []
    all_sin = []
    for section_size in sections:
        dim = section_size * 2
        freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = pos.unsqueeze(1) * freq.unsqueeze(0)
        all_cos.append(angles.cos())
        all_sin.append(angles.sin())

    cos = torch.cat(all_cos, dim=-1)  # [T, 64]
    sin = torch.cat(all_sin, dim=-1)  # [T, 64]

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


# ── Exportable Wrappers ─────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """Wraps AudioEncoder for ONNX export. FP32 I/O, FP16 compute."""
    def __init__(self, encoder, use_fp16=True):
        super().__init__()
        self.encoder = encoder.half() if use_fp16 else encoder
        self.use_fp16 = use_fp16

    def forward(self, mel):
        # mel: [B, 128, T] float32
        x = mel.half() if self.use_fp16 else mel
        out = self.encoder(x)
        return out.float() if self.use_fp16 else out


class DecoderInitWrapper(nn.Module):
    """First decoder pass. FP32 I/O, FP16 compute."""
    def __init__(self, embed_tokens, layers, norm, lm_head, use_fp16=True):
        super().__init__()
        if use_fp16:
            self.embed_tokens = embed_tokens.half()
            self.layers = nn.ModuleList([l.half() for l in layers])
            self.norm = norm.half()
            self.lm_head = lm_head.half()
        else:
            self.embed_tokens = embed_tokens
            self.layers = layers
            self.norm = norm
            self.lm_head = lm_head
        self.use_fp16 = use_fp16

    def forward(self, input_ids, position_ids, audio_features, audio_offset):
        # input_ids: [B, seq_len], position_ids: [B, seq_len]
        # audio_features: [B, audio_len, 1024] float32, audio_offset: [B] int64

        x = self.embed_tokens(input_ids)  # [B, seq_len, 1024] fp16/fp32

        # Inject audio features (avoid in-place op for tracing)
        off = audio_offset[0]
        audio_len = audio_features.shape[1]
        af = audio_features.half() if self.use_fp16 else audio_features
        pre = x[:, :off, :]
        post = x[:, off + audio_len:, :]
        x = torch.cat([pre, af, post], dim=1)

        cos, sin = compute_rope_cos_sin(position_ids)

        all_keys = []
        all_values = []
        for layer in self.layers:
            x, k, v = layer(x, cos, sin)
            all_keys.append(k)
            all_values.append(v)

        x = self.norm(x)
        logits = self.lm_head(x)

        past_keys = torch.stack(all_keys, dim=0)
        past_values = torch.stack(all_values, dim=0)

        if self.use_fp16:
            return logits.float(), past_keys.float(), past_values.float()
        return logits, past_keys, past_values


class DecoderStepWrapper(nn.Module):
    """Subsequent decoder steps. FP32 I/O, FP16 compute."""
    def __init__(self, layers, norm, lm_head, use_fp16=True):
        super().__init__()
        if use_fp16:
            self.layers = nn.ModuleList([l.half() for l in layers])
            self.norm = norm.half()
            self.lm_head = lm_head.half()
        else:
            self.layers = layers
            self.norm = norm
            self.lm_head = lm_head
        self.use_fp16 = use_fp16

    def forward(self, input_embeds, position_ids, past_keys, past_values):
        # All float32 I/O
        cos, sin = compute_rope_cos_sin(position_ids)

        x = input_embeds.half() if self.use_fp16 else input_embeds
        pk = past_keys.half() if self.use_fp16 else past_keys
        pv = past_values.half() if self.use_fp16 else past_values

        new_keys = []
        new_values = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, pk[i], pv[i])
            new_keys.append(k)
            new_values.append(v)

        x = self.norm(x)
        logits = self.lm_head(x)

        new_past_keys = torch.stack(new_keys, dim=0)
        new_past_values = torch.stack(new_values, dim=0)

        if self.use_fp16:
            return logits.float(), new_past_keys.float(), new_past_values.float()
        return logits, new_past_keys, new_past_values


# ── Weight loading ───────────────────────────────────────────────────────────

def load_encoder(weights):
    enc = AudioEncoder()
    prefix = "thinker.audio_tower."

    enc.conv2d1.weight.data = weights[prefix + "conv2d1.weight"].float()
    enc.conv2d1.bias.data = weights[prefix + "conv2d1.bias"].float()
    enc.conv2d2.weight.data = weights[prefix + "conv2d2.weight"].float()
    enc.conv2d2.bias.data = weights[prefix + "conv2d2.bias"].float()
    enc.conv2d3.weight.data = weights[prefix + "conv2d3.weight"].float()
    enc.conv2d3.bias.data = weights[prefix + "conv2d3.bias"].float()

    enc.conv_out.weight.data = weights[prefix + "conv_out.weight"].float()
    if (prefix + "conv_out.bias") in weights:
        enc.conv_out.bias = nn.Parameter(weights[prefix + "conv_out.bias"].float())

    for i in range(ENC_LAYERS):
        lp = f"{prefix}layers.{i}."
        layer = enc.layers[i]
        layer.self_attn_layer_norm.weight.data = weights[lp + "self_attn_layer_norm.weight"].float()
        layer.self_attn_layer_norm.bias.data = weights[lp + "self_attn_layer_norm.bias"].float()
        layer.self_attn.q_proj.weight.data = weights[lp + "self_attn.q_proj.weight"].float()
        layer.self_attn.q_proj.bias.data = weights[lp + "self_attn.q_proj.bias"].float()
        layer.self_attn.k_proj.weight.data = weights[lp + "self_attn.k_proj.weight"].float()
        layer.self_attn.k_proj.bias.data = weights[lp + "self_attn.k_proj.bias"].float()
        layer.self_attn.v_proj.weight.data = weights[lp + "self_attn.v_proj.weight"].float()
        layer.self_attn.v_proj.bias.data = weights[lp + "self_attn.v_proj.bias"].float()
        layer.self_attn.out_proj.weight.data = weights[lp + "self_attn.out_proj.weight"].float()
        layer.self_attn.out_proj.bias.data = weights[lp + "self_attn.out_proj.bias"].float()
        layer.final_layer_norm.weight.data = weights[lp + "final_layer_norm.weight"].float()
        layer.final_layer_norm.bias.data = weights[lp + "final_layer_norm.bias"].float()
        layer.fc1.weight.data = weights[lp + "fc1.weight"].float()
        layer.fc1.bias.data = weights[lp + "fc1.bias"].float()
        layer.fc2.weight.data = weights[lp + "fc2.weight"].float()
        layer.fc2.bias.data = weights[lp + "fc2.bias"].float()

    enc.ln_post.weight.data = weights[prefix + "ln_post.weight"].float()
    enc.ln_post.bias.data = weights[prefix + "ln_post.bias"].float()
    enc.proj1.weight.data = weights[prefix + "proj1.weight"].float()
    enc.proj1.bias.data = weights[prefix + "proj1.bias"].float()
    enc.proj2.weight.data = weights[prefix + "proj2.weight"].float()
    enc.proj2.bias.data = weights[prefix + "proj2.bias"].float()

    return enc


def load_decoder_parts(weights):
    embed = nn.Embedding(VOCAB_SIZE, DEC_DIM)
    embed.weight.data = weights["thinker.model.embed_tokens.weight"].float()

    layers = nn.ModuleList()
    for i in range(DEC_LAYERS):
        lp = f"thinker.model.layers.{i}."
        layer = DecoderLayer()
        layer.input_layernorm.weight.data = weights[lp + "input_layernorm.weight"].float()
        layer.post_attention_layernorm.weight.data = weights[lp + "post_attention_layernorm.weight"].float()
        layer.self_attn.q_proj.weight.data = weights[lp + "self_attn.q_proj.weight"].float()
        layer.self_attn.k_proj.weight.data = weights[lp + "self_attn.k_proj.weight"].float()
        layer.self_attn.v_proj.weight.data = weights[lp + "self_attn.v_proj.weight"].float()
        layer.self_attn.o_proj.weight.data = weights[lp + "self_attn.o_proj.weight"].float()
        layer.self_attn.q_norm.weight.data = weights[lp + "self_attn.q_norm.weight"].float()
        layer.self_attn.k_norm.weight.data = weights[lp + "self_attn.k_norm.weight"].float()
        layer.mlp.gate_proj.weight.data = weights[lp + "mlp.gate_proj.weight"].float()
        layer.mlp.up_proj.weight.data = weights[lp + "mlp.up_proj.weight"].float()
        layer.mlp.down_proj.weight.data = weights[lp + "mlp.down_proj.weight"].float()
        layers.append(layer)

    norm = RMSNorm(DEC_DIM)
    norm.weight.data = weights["thinker.model.norm.weight"].float()

    # tie_word_embeddings=true, so lm_head uses embed_tokens weights
    lm_head = nn.Linear(DEC_DIM, VOCAB_SIZE, bias=False)
    lm_head.weight.data = embed.weight.data.clone()

    return embed, layers, norm, lm_head


# ── ONNX export ──────────────────────────────────────────────────────────────

def _export(model, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        model, args, path,
        opset_version=OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def export_encoder(encoder, output_path, use_fp16=True):
    print("Exporting encoder...")
    wrapper = EncoderWrapper(encoder, use_fp16=use_fp16)
    wrapper.eval()

    mel = torch.randn(1, N_MELS, 100)  # always FP32 I/O

    _export(
        wrapper, (mel,), output_path,
        input_names=["mel"],
        output_names=["audio_features"],
        dynamic_axes={
            "mel": {2: "time"},
            "audio_features": {1: "seq_len"},
        },
    )
    print(f"  -> {output_path}")


def export_decoder_init(embed, layers, norm, lm_head, output_path, use_fp16=True):
    print("Exporting decoder_init...")
    wrapper = DecoderInitWrapper(embed, layers, norm, lm_head, use_fp16=use_fp16)
    wrapper.eval()

    seq_len = 20
    audio_len = 5
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    audio_features = torch.randn(1, audio_len, OUTPUT_DIM)  # FP32 I/O
    audio_offset = torch.tensor([3], dtype=torch.long)

    _export(
        wrapper,
        (input_ids, position_ids, audio_features, audio_offset),
        output_path,
        input_names=["input_ids", "position_ids", "audio_features", "audio_offset"],
        output_names=["logits", "past_keys", "past_values"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "position_ids": {1: "seq_len"},
            "audio_features": {1: "audio_len"},
            "logits": {1: "seq_len"},
            "past_keys": {3: "seq_len"},
            "past_values": {3: "seq_len"},
        },
    )
    print(f"  -> {output_path}")


def export_decoder_step(layers, norm, lm_head, output_path, use_fp16=True):
    print("Exporting decoder_step...")
    wrapper = DecoderStepWrapper(layers, norm, lm_head, use_fp16=use_fp16)
    wrapper.eval()

    input_embeds = torch.randn(1, 1, DEC_DIM)  # FP32 I/O
    position_ids = torch.tensor([[20]], dtype=torch.long)
    past_keys = torch.randn(DEC_LAYERS, 1, DEC_KV_HEADS, 20, HEAD_DIM)
    past_values = torch.randn(DEC_LAYERS, 1, DEC_KV_HEADS, 20, HEAD_DIM)

    _export(
        wrapper,
        (input_embeds, position_ids, past_keys, past_values),
        output_path,
        input_names=["input_embeds", "position_ids", "past_keys", "past_values"],
        output_names=["logits", "past_keys_out", "past_values_out"],
        dynamic_axes={
            "past_keys": {3: "past_len"},
            "past_values": {3: "past_len"},
            "past_keys_out": {3: "total_len"},
            "past_values_out": {3: "total_len"},
        },
    )
    print(f"  -> {output_path}")


def export_embed_tokens(weights, output_path):
    """Save embedding table as FP16 binary."""
    print("Exporting embed_tokens.bin...")
    embed = weights["thinker.model.embed_tokens.weight"]
    arr = embed.cpu().to(torch.float32).numpy().astype(np.float16)
    arr.tofile(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")


def write_config(output_dir):
    config = {
        "model_type": "qwen3_asr",
        "encoder": {
            "num_layers": ENC_LAYERS,
            "hidden_size": ENC_DIM,
            "num_heads": ENC_HEADS,
            "ffn_dim": ENC_FFN,
            "conv_channels": CONV_CH,
            "output_dim": OUTPUT_DIM,
            "downsample_factor": 8,
            "num_mel_bins": N_MELS,
        },
        "decoder": {
            "num_layers": DEC_LAYERS,
            "hidden_size": DEC_DIM,
            "num_attention_heads": DEC_HEADS,
            "num_key_value_heads": DEC_KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": DEC_FFN,
            "vocab_size": VOCAB_SIZE,
            "rope_theta": ROPE_THETA,
            "rms_norm_eps": RMS_EPS,
            "tie_word_embeddings": True,
            "rope_scaling": {
                "mrope_section": MROPE_SECTIONS,
                "interleaved": True,
            },
        },
        "mel": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": N_MELS,
            "fmin": 0,
            "fmax": 8000,
        },
        "special_tokens": {
            "eos_token_ids": [151643, 151645],
            "pad_token_id": 151643,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_pad_token_id": 151676,
            "asr_text_token_id": 151704,
        },
        "embed_tokens_dtype": "float16",
    }
    path = os.path.join(output_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR-0.6B to FP16 ONNX")
    parser.add_argument("--model-dir", default="Qwen3-ASR-0.6B", help="Source model directory")
    parser.add_argument("--output-dir", default="models/asr", help="Output directory for ONNX models")
    parser.add_argument("--no-fp16", action="store_true", help="Keep FP32 (skip FP16 conversion)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    safetensors_path = os.path.join(args.model_dir, "model.safetensors")
    print(f"Loading weights from {safetensors_path}...")
    weights = load_file(safetensors_path)
    print(f"  Loaded {len(weights)} tensors")

    # ── Build models ──
    print("\nBuilding encoder...")
    encoder = load_encoder(weights)
    encoder.eval()

    print("Building decoder...")
    embed, dec_layers, dec_norm, lm_head = load_decoder_parts(weights)

    use_fp16 = not args.no_fp16
    print(f"\nExport precision: {'FP16' if use_fp16 else 'FP32'}")

    with torch.no_grad():
        for name, export_fn in [
            ("encoder", lambda p: export_encoder(encoder, p, use_fp16)),
            ("decoder_init", lambda p: export_decoder_init(embed, dec_layers, dec_norm, lm_head, p, use_fp16)),
            ("decoder_step", lambda p: export_decoder_step(dec_layers, dec_norm, lm_head, p, use_fp16)),
        ]:
            onnx_path = os.path.join(args.output_dir, name + ".onnx")
            print(f"\n=== {name} ===")
            export_fn(onnx_path)

    # ── Embed tokens ──
    print("\n=== Embeddings ===")
    export_embed_tokens(weights, os.path.join(args.output_dir, "embed_tokens.bin"))

    # ── Config ──
    print("\n=== Config ===")
    write_config(args.output_dir)

    # ── Tokenizer ──
    src_tokenizer = os.path.join(args.model_dir, "tokenizer.json")
    dst_tokenizer = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.isfile(src_tokenizer):
        shutil.copy2(src_tokenizer, dst_tokenizer)
        print(f"  -> {dst_tokenizer} (copied)")
    else:
        print(f"  WARNING: {src_tokenizer} not found, skipping tokenizer copy")

    print("\n=== Done ===")
    print(f"Output: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        fp = os.path.join(args.output_dir, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            if size > 1024 * 1024:
                print(f"  {f}: {size / (1024*1024):.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
