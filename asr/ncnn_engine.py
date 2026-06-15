"""
Sherpa-NCNN inference engine (ASR backend No.2).

Wraps `sherpa_ncnn.Recognizer` (ConvEmformer Transducer, int8/float16). Exposes
the same public surface as `asr.engine` so `run.py` can dispatch to either
backend transparently:
    load_session(model_dir, num_threads, use_gpu) -> engine
    transcribe(engine, wav, language, use_itn) -> {"text", "total_ms", ...}

Expected model directory layout (e.g. csukuangfj/sherpa-ncnn-conv-emformer-
transducer-2022-12-06):
    encoder.ncnn.param + encoder.ncnn.bin
    decoder.ncnn.param + decoder.ncnn.bin
    joiner.ncnn.param  + joiner.ncnn.bin
    tokens.txt

`language` / `use_itn` are accepted for signature parity with the Qwen3 engine
but have no effect on a sherpa-ncnn transducer (the model is fixed-language).
"""

import logging
import os
import time

import numpy as np

# Reuse the battle-tested WAV/loader from the Qwen3 engine so both backends
# agree on audio decoding and 16 kHz resampling.
from .engine import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)


def _find_component(model_dir: str, name: str):
    """Locate a ncnn component's .param + .bin inside model_dir.

    Supports the common sherpa-ncnn naming conventions, in priority order:
      1. <name>.ncnn.param          (decoder in most repos)
      2. <name>.ncnn.int8.param     (encoder/joiner int8 quantized)
      3. <name>.int8.ncnn.param
    The .bin is picked to match the chosen .param suffix so int8 param always
    pairs with the int8 bin.
    """
    candidates = [
        (".ncnn.param", ".ncnn.bin"),
        (".ncnn.int8.param", ".ncnn.int8.bin"),
        (".int8.ncnn.param", ".int8.ncnn.bin"),
        (".int8.param", ".int8.bin"),
        (".param", ".bin"),
    ]
    for param_suffix, bin_suffix in candidates:
        param = os.path.join(model_dir, f"{name}{param_suffix}")
        binf = os.path.join(model_dir, f"{name}{bin_suffix}")
        if os.path.isfile(param) and os.path.isfile(binf):
            return param, binf
    # Return the default-named pair (may not exist) so validation reports it.
    return os.path.join(model_dir, f"{name}.ncnn.param"), os.path.join(model_dir, f"{name}.ncnn.bin")


def _validate_model_dir(model_dir: str):
    missing = []
    for comp in ("encoder", "decoder", "joiner"):
        param, binf = _find_component(model_dir, comp)
        if not os.path.isfile(param):
            missing.append(f"{comp}*.ncnn.param")
        if not os.path.isfile(binf):
            missing.append(f"{comp}*.ncnn.bin")
    if not os.path.isfile(os.path.join(model_dir, "tokens.txt")):
        missing.append("tokens.txt")
    if missing:
        raise FileNotFoundError(f"Sherpa-NCNN model directory is incomplete: {missing}")


class SherpaNcnnEngine:
    """Thin wrapper around sherpa_ncnn.Recognizer.

    Non-streaming use: feed the whole utterance then call input_finished() and
    read `.text`. A fresh recognizer per request keeps state clean.
    """

    def __init__(self, model_dir: str, num_threads: int = None, use_gpu: bool = False):
        try:
            import sherpa_ncnn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-ncnn is not installed. Install it with `pip install sherpa-ncnn` "
                "when asr.engine=ncnn"
            ) from exc

        if num_threads is None:
            cores = os.cpu_count() or 4
            max_threads = int(os.environ.get("SHERPA_NCNN_MAX_THREADS", "4"))
            num_threads = min(max(1, cores - 1), max_threads)

        if use_gpu:
            # ncnn can use Vulkan, but the Python wheel exposes no GPU flag here.
            logger.warning("sherpa-ncnn has no GPU path in this wrapper; running on CPU")

        self.model_dir = model_dir
        self.num_threads = num_threads
        _validate_model_dir(model_dir)

        enc_param, enc_bin = _find_component(model_dir, "encoder")
        dec_param, dec_bin = _find_component(model_dir, "decoder")
        join_param, join_bin = _find_component(model_dir, "joiner")
        tokens = os.path.join(model_dir, "tokens.txt")

        # Build a template config dict; recreate the recognizer per request to
        # keep each utterance isolated (transducer streaming state).
        self._recognizer_config = {
            "tokens": tokens,
            "encoder_param": enc_param,
            "encoder_bin": enc_bin,
            "decoder_param": dec_param,
            "decoder_bin": dec_bin,
            "joiner_param": join_param,
            "joiner_bin": join_bin,
            "num_threads": num_threads,
        }

        # Warm up the recognizer once so the first real request doesn't pay
        # the ncnn init cost.
        t0 = time.perf_counter()
        _ = self._new_recognizer()
        logger.info(
            "Sherpa-NCNN model loaded from %s (warmup %.0fms, %d threads)",
            model_dir,
            (time.perf_counter() - t0) * 1000,
            num_threads,
        )

    def _new_recognizer(self):
        import sherpa_ncnn

        return sherpa_ncnn.Recognizer(**self._recognizer_config)

    def transcribe(self, wav: np.ndarray, language: str = "auto", use_itn: bool = True) -> dict:
        t0 = time.perf_counter()

        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return {"text": "", "total_ms": 0.0, "segments": 0}

        recognizer = self._new_recognizer()
        # sherpa_ncnn expects float32 samples in [-1, 1] at the recognizer's
        # native sample rate (16 kHz here). load_audio already resamples.
        recognizer.accept_waveform(SAMPLE_RATE, wav)
        recognizer.input_finished()
        text = (recognizer.text or "").strip()

        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "text": text,
            "total_ms": round(total_ms, 1),
            "segments": 1,
        }


def load_session(model_dir: str, num_threads: int = None, use_gpu: bool = False):
    return SherpaNcnnEngine(model_dir, num_threads=num_threads, use_gpu=use_gpu)


def transcribe(engine: SherpaNcnnEngine, wav: np.ndarray, language: str = "auto", use_itn: bool = True) -> dict:
    return engine.transcribe(wav, language=language, use_itn=use_itn)
