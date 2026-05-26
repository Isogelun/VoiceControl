"""
asr/engine.py

SenseVoiceSmall ONNX 推理引擎。
纯推理逻辑，不含 HTTP 服务和 CLI。
"""

import io
import os
import re
import time
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LFR_M, LFR_N = 7, 6

# language / text_norm IDs（来自模型 metadata）
LANG_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}
WITH_ITN = 14
WITHOUT_ITN = 15


# ─── 模型加载 ───────────────────────────────────────────────────────────────────

def _parse_metadata(sess):
    raw = sess.get_modelmeta().custom_metadata_map
    neg_mean = np.array([float(x) for x in raw["neg_mean"].split(",")], dtype=np.float32)
    inv_stddev = np.array([float(x) for x in raw["inv_stddev"].split(",")], dtype=np.float32)
    return neg_mean, inv_stddev


def load_session(model_dir: str, use_q8: bool = True, num_threads: int = None, use_gpu: bool = False):
    """加载 ONNX 模型，返回 (session, neg_mean, inv_stddev)"""
    import onnxruntime as ort
    if num_threads is None:
        cores = os.cpu_count() or 4
        num_threads = min(max(2, cores - 1), 4)
    model_file = "model_q8.onnx" if use_q8 else "model.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("使用 GPU 推理 (CUDA)")
    else:
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            logger.warning("CUDAExecutionProvider 不可用，回退 CPU")

    sess = ort.InferenceSession(
        os.path.join(model_dir, model_file),
        sess_options=opts,
        providers=providers,
    )
    neg_mean, inv_stddev = _parse_metadata(sess)
    return sess, neg_mean, inv_stddev


def load_tokens(model_dir: str) -> dict:
    """加载词表 tokens.txt"""
    tokens = {}
    with open(os.path.join(model_dir, "tokens.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]
    return tokens


# ─── 音频处理 ───────────────────────────────────────────────────────────────────

def load_audio(source) -> np.ndarray:
    """加载音频文件/字节，返回 16kHz float32 单声道 numpy 数组"""
    try:
        data, sr = sf.read(source, dtype="float32", always_2d=False)
    except Exception:
        import librosa
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        data, sr = librosa.load(source, sr=None, mono=True, dtype=np.float32)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data


# ─── 特征提取 ───────────────────────────────────────────────────────────────────

def _fbank(wav: np.ndarray) -> np.ndarray:
    sr, n_mels, n_fft = SAMPLE_RATE, 80, 512
    win_samples = int(sr * 0.025)
    hop_samples = int(sr * 0.010)
    n_frames = (len(wav) - win_samples) // hop_samples + 1
    idx = np.arange(win_samples)[None, :] + np.arange(n_frames)[:, None] * hop_samples
    frames = wav[idx] * np.hamming(win_samples)
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    # mel filterbank
    fmin, fmax = 20, sr // 2
    mel_pts = np.linspace(2595 * np.log10(1 + fmin / 700), 2595 * np.log10(1 + fmax / 700), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fb[m-1, bins[m-1]:bins[m]] = (np.arange(bins[m-1], bins[m]) - bins[m-1]) / max(bins[m] - bins[m-1], 1)
        fb[m-1, bins[m]:bins[m+1]] = (bins[m+1] - np.arange(bins[m], bins[m+1])) / max(bins[m+1] - bins[m], 1)
    return np.log(spec @ fb.T + 1e-10).astype(np.float32)


def _lfr(fbank: np.ndarray) -> np.ndarray:
    T = fbank.shape[0]
    if T == 0:
        raise ValueError("音频过短")
    if T < LFR_M:
        fbank = np.concatenate([fbank, np.tile(fbank[-1:], (LFR_M - T, 1))], axis=0)
        T = fbank.shape[0]
    lfr_len = (T - LFR_M) // LFR_N + 1
    idx = np.arange(lfr_len)[:, None] * LFR_N + np.arange(LFR_M)[None, :]
    return fbank[idx].reshape(lfr_len, -1)  # (T_lfr, 560)


def extract_features(wav: np.ndarray, neg_mean: np.ndarray, inv_stddev: np.ndarray):
    """音频波形 → 归一化 LFR 特征"""
    feat = _lfr(_fbank(wav))                        # (T, 560)
    feat = (feat + neg_mean[:feat.shape[1]]) * inv_stddev[:feat.shape[1]]
    return feat[None].astype(np.float32)            # (1, T, 560)


# ─── 解码 ───────────────────────────────────────────────────────────────────────

def _ctc_decode(logits: np.ndarray, tokens: dict) -> str:
    ids = np.argmax(logits[0], axis=-1)
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i > 2:  # skip blank(0), <s>(1), </s>(2)
            out.append(i)
        prev = i
    text = "".join(tokens.get(i, "") for i in out)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.replace("\u2581", " ").strip()


# ─── 推理入口 ───────────────────────────────────────────────────────────────────

def transcribe(sess, neg_mean, inv_stddev, tokens, wav: np.ndarray,
               language: str = "auto", use_itn: bool = True) -> dict:
    """
    执行 ASR 推理。
    返回: {"text": str, "feat_ms": float, "infer_ms": float, "total_ms": float}
    """
    t0 = time.perf_counter()
    lang_id = LANG_IDS.get(language, 0)
    text_norm_id = WITH_ITN if use_itn else WITHOUT_ITN

    tf = time.perf_counter()
    feat = extract_features(wav, neg_mean, inv_stddev)
    feat_ms = (time.perf_counter() - tf) * 1000

    ti = time.perf_counter()
    logits, = sess.run(None, {
        "x":         feat,
        "x_length":  np.array([feat.shape[1]], dtype=np.int32),
        "language":  np.array([lang_id], dtype=np.int32),
        "text_norm": np.array([text_norm_id], dtype=np.int32),
    })
    infer_ms = (time.perf_counter() - ti) * 1000

    return {
        "text": _ctc_decode(logits, tokens),
        "feat_ms": round(feat_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "segments": 1,
    }
