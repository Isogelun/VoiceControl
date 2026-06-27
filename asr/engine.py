"""
Qwen3-ASR ONNX inference engine.

Expected model directory layout:
- encoder.int4.onnx or encoder.onnx
- decoder_init.int4.onnx or decoder_init.onnx
- decoder_step.int4.onnx or decoder_step.onnx
- embed_tokens.bin
- tokenizer.json
- config.json
"""

import io
import json
import logging
import os
import re
import time

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _providers(use_gpu: bool, *, gpu_mem_limit: int = 0):
    import onnxruntime as ort

    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        logger.info("Using GPU inference (CUDA)")
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
        }
        if gpu_mem_limit > 0:
            cuda_opts["gpu_mem_limit"] = gpu_mem_limit
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    if use_gpu:
        logger.warning("CUDAExecutionProvider is unavailable, falling back to CPU")
    return ["CPUExecutionProvider"]


def _cpu_providers():
    return ["CPUExecutionProvider"]


def _ort_session(path: str, providers, num_threads: int, is_int4: bool = True):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if is_int4:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _model_suffix(model_dir: str) -> str:
    if os.path.isfile(os.path.join(model_dir, "encoder.int4.onnx")):
        return ".int4.onnx"
    if os.path.isfile(os.path.join(model_dir, "encoder.onnx")):
        return ".onnx"
    raise FileNotFoundError(f"Qwen3-ASR encoder not found in {model_dir}")


def _validate_model_dir(model_dir: str, suffix: str):
    required = [
        "config.json",
        "tokenizer.json",
        "embed_tokens.bin",
        "encoder" + suffix,
        "decoder_init" + suffix,
        "decoder_step" + suffix,
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise FileNotFoundError(f"Qwen3-ASR model directory is incomplete: {missing}")


class Qwen3ASREngine:
    def __init__(self, model_dir: str, num_threads: int = None, use_gpu: bool = False,
                 gpu_encoder_only: bool = False):
        if num_threads is None:
            cores = os.cpu_count() or 4
            max_threads = int(os.environ.get("QWEN_ASR_MAX_THREADS", "8"))
            num_threads = min(max(2, cores - 1), max_threads)

        self.model_dir = model_dir
        self.suffix = _model_suffix(model_dir)
        _validate_model_dir(model_dir, self.suffix)

        gpu_mem_limit = int(os.environ.get("ORT_GPU_MEM_LIMIT", "0"))
        encoder_providers = _providers(use_gpu, gpu_mem_limit=gpu_mem_limit)
        if use_gpu and gpu_encoder_only:
            decoder_providers = _cpu_providers()
            logger.info("GPU encoder-only mode: decoders will run on CPU")
        else:
            decoder_providers = encoder_providers

        self.config = self._load_config()
        self.special = self.config.get("special_tokens", {})
        self.max_new_tokens = int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "32"))
        self.fast_mel = os.environ.get("QWEN_ASR_FAST_MEL", "1") not in {"0", "false", "False", "no"}

        is_int4 = self.suffix == ".int4.onnx"
        self.encoder = _ort_session(
            os.path.join(model_dir, "encoder" + self.suffix), encoder_providers, num_threads, is_int4
        )
        self.decoder_init = _ort_session(
            os.path.join(model_dir, "decoder_init" + self.suffix), decoder_providers, num_threads, is_int4
        )
        self.decoder_step = _ort_session(
            os.path.join(model_dir, "decoder_step" + self.suffix), decoder_providers, num_threads, is_int4
        )
        self.tokenizer = self._load_tokenizer()
        self.embed_tokens = self._load_embeddings()
        self._init_mel_frontend()

        enc_prov = self.encoder.get_providers()[0] if self.encoder.get_providers() else "unknown"
        dec_prov = self.decoder_init.get_providers()[0] if self.decoder_init.get_providers() else "unknown"
        logger.info(
            "Qwen3-ASR model loaded (%s, encoder: %s, decoder: %s)",
            "int4" if self.suffix == ".int4.onnx" else "fp16/fp32",
            enc_prov, dec_prov,
        )

    def _load_config(self):
        with open(os.path.join(self.model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("model_type") != "qwen3_asr":
            raise RuntimeError(f"Unsupported ASR model_type: {config.get('model_type')}")
        return config

    def _load_tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))

    def _load_embeddings(self):
        decoder = self.config.get("decoder", {})
        vocab_size = int(decoder.get("vocab_size", 151936))
        hidden_size = int(decoder.get("hidden_size", 1024))
        return np.memmap(
            os.path.join(self.model_dir, "embed_tokens.bin"),
            dtype=np.float16,
            mode="r",
            shape=(vocab_size, hidden_size),
        )

    def _init_mel_frontend(self):
        mel_cfg = self.config.get("mel", {})
        self._n_fft = int(mel_cfg.get("n_fft", 400))
        self._hop_length = int(mel_cfg.get("hop_length", 160))
        self._n_mels = int(mel_cfg.get("n_mels", 128))
        fmin = float(mel_cfg.get("fmin", 0))
        fmax = float(mel_cfg.get("fmax", SAMPLE_RATE // 2))
        n = np.arange(self._n_fft, dtype=np.float32)
        # Match librosa/scipy's periodic Hann window for window="hann".
        self._window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / self._n_fft)).astype(np.float32)

        import librosa

        self._mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=self._n_fft,
            n_mels=self._n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=False,
            norm="slaney",
        ).astype(np.float32)

    def _token_ids(self, text: str):
        return self.tokenizer.encode(text).ids

    def _log_mel(self, wav: np.ndarray) -> np.ndarray:
        if self.fast_mel:
            return self._log_mel_fast(wav)
        return self._log_mel_librosa(wav)

    def _log_mel_fast(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            wav = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)

        pad = self._n_fft // 2
        padded = np.pad(wav, (pad, pad), mode="constant")
        if padded.size < self._n_fft:
            padded = np.pad(padded, (0, self._n_fft - padded.size), mode="constant")

        n_frames = 1 + (padded.size - self._n_fft) // self._hop_length
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_frames, self._n_fft),
            strides=(padded.strides[0] * self._hop_length, padded.strides[0]),
            writeable=False,
        )
        windowed = frames * self._window
        power = np.abs(np.fft.rfft(windowed, n=self._n_fft, axis=1)) ** 2
        mel = np.dot(self._mel_basis, power.T, out=None)
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel[None].astype(np.float32, copy=False)

    def _log_mel_librosa(self, wav: np.ndarray) -> np.ndarray:
        import librosa

        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            wav = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)

        mel_cfg = self.config.get("mel", {})
        n_fft = int(mel_cfg.get("n_fft", 400))
        hop_length = int(mel_cfg.get("hop_length", 160))
        n_mels = int(mel_cfg.get("n_mels", 128))
        fmin = float(mel_cfg.get("fmin", 0))
        fmax = float(mel_cfg.get("fmax", SAMPLE_RATE // 2))

        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            power=2.0,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=False,
            norm="slaney",
        )
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel[None].astype(np.float32)

    def _build_prompt(self, audio_len: int):
        audio_pad_id = int(self.special.get("audio_pad_token_id", 151676))
        prefix = self._token_ids("<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>")
        suffix = self._token_ids("<|audio_end|><|im_end|><|im_start|>assistant")
        audio_offset = len(prefix)
        ids = prefix + [audio_pad_id] * int(audio_len) + suffix
        return np.asarray([ids], dtype=np.int64), np.asarray([audio_offset], dtype=np.int64)

    def _clean_text(self, ids) -> str:
        text = self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)
        text = text.replace("<asr_text>", "")
        text = re.sub(r"<\|[^|]*\|>", "", text)
        return text.strip()

    def transcribe(self, wav: np.ndarray, language: str = "auto", use_itn: bool = True) -> dict:
        t0 = time.perf_counter()

        tf = time.perf_counter()
        mel = self._log_mel(wav)
        audio_features, = self.encoder.run(None, {"mel": mel})
        feat_ms = (time.perf_counter() - tf) * 1000

        ti = time.perf_counter()
        audio_len = int(audio_features.shape[1])
        input_ids, audio_offset = self._build_prompt(audio_len)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        logits, past_keys, past_values = self.decoder_init.run(
            None,
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "audio_features": audio_features.astype(np.float32, copy=False),
                "audio_offset": audio_offset,
            },
        )

        eos_ids = {int(i) for i in self.special.get("eos_token_ids", [151643, 151645])}
        out_ids = []
        next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        for _ in range(self.max_new_tokens):
            if next_id in eos_ids:
                break
            out_ids.append(next_id)

            input_embeds = np.asarray(self.embed_tokens[next_id], dtype=np.float32).reshape(1, 1, -1)
            step_position = np.asarray([[seq_len + len(out_ids) - 1]], dtype=np.int64)
            logits, past_keys, past_values = self.decoder_step.run(
                None,
                {
                    "input_embeds": input_embeds,
                    "position_ids": step_position,
                    "past_keys": past_keys,
                    "past_values": past_values,
                },
            )
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])

        infer_ms = (time.perf_counter() - ti) * 1000
        return {
            "text": self._clean_text(out_ids),
            "feat_ms": round(feat_ms, 1),
            "infer_ms": round(infer_ms, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "segments": 1,
        }


def load_session(model_dir: str, num_threads: int = None, use_gpu: bool = False,
                  gpu_encoder_only: bool = False):
    return Qwen3ASREngine(model_dir, num_threads=num_threads, use_gpu=use_gpu,
                          gpu_encoder_only=gpu_encoder_only)


def load_audio(source) -> np.ndarray:
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


def transcribe(engine: Qwen3ASREngine, wav: np.ndarray, language: str = "auto", use_itn: bool = True) -> dict:
    return engine.transcribe(wav, language=language, use_itn=use_itn)
