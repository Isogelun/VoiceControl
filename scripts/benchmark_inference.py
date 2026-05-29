#!/usr/bin/env python3
"""
Benchmark the local ASR/NLU inference path.
"""

import argparse
import json
import os
import sys
import time
from statistics import mean


def _configure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _bench_asr(args):
    from asr.engine import load_audio, load_session

    if args.asr_fast_mel is not None:
        os.environ["QWEN_ASR_FAST_MEL"] = "1" if args.asr_fast_mel else "0"
    if args.asr_max_tokens is not None:
        os.environ["QWEN_ASR_MAX_NEW_TOKENS"] = str(args.asr_max_tokens)

    started = time.perf_counter()
    engine = load_session(args.asr_model, num_threads=args.threads, use_gpu=args.gpu)
    load_ms = (time.perf_counter() - started) * 1000
    wav = load_audio(args.audio)

    results = []
    for _ in range(args.warmup):
        engine.transcribe(wav)
    for _ in range(args.runs):
        result = engine.transcribe(wav)
        results.append(result)

    return {
        "load_ms": round(load_ms, 1),
        "audio_ms": round(wav.shape[0] / 16000 * 1000, 1),
        "feat_ms": round(mean(item["feat_ms"] for item in results), 1),
        "infer_ms": round(mean(item["infer_ms"] for item in results), 1),
        "total_ms": round(mean(item["total_ms"] for item in results), 1),
        "text": results[-1]["text"] if results else "",
    }


def _bench_pipeline_fast_path(args):
    from asr.engine import load_audio, load_session
    import numpy as np
    from pipeline.main import VoicePipeline
    from pipeline.text_normalizer import normalize_asr_text, parse_command_rule

    if args.asr_fast_mel is not None:
        os.environ["QWEN_ASR_FAST_MEL"] = "1" if args.asr_fast_mel else "0"
    if args.asr_max_tokens is not None:
        os.environ["QWEN_ASR_MAX_NEW_TOKENS"] = str(args.asr_max_tokens)

    started = time.perf_counter()
    engine = load_session(args.asr_model, num_threads=args.threads, use_gpu=args.gpu)
    load_ms = (time.perf_counter() - started) * 1000
    wav = load_audio(args.audio)
    input_wav = wav
    if args.pipeline_trim:
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._noise_rms = float(args.pipeline_noise_rms)
        pcm = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
        input_wav = np.frombuffer(pipe._with_padding(pcm.tobytes()), dtype=np.int16).astype(np.float32) / 32768.0

    samples = []
    last = {}
    for index in range(args.warmup + args.runs):
        started = time.perf_counter()
        result = engine.transcribe(input_wav)
        normalized = normalize_asr_text(result.get("text", ""))
        command = parse_command_rule(normalized)
        total_ms = (time.perf_counter() - started) * 1000
        item = {
            "total_ms": total_ms,
            "asr_ms": result.get("total_ms", 0),
            "feat_ms": result.get("feat_ms", 0),
            "infer_ms": result.get("infer_ms", 0),
            "text": result.get("text", ""),
            "normalized": normalized,
            "command": command or {"intent": "unknown", "slots": {}, "source": "rule_miss"},
        }
        if index >= args.warmup:
            samples.append(item)
            last = item

    return {
        "load_ms": round(load_ms, 1),
        "audio_ms": round(wav.shape[0] / 16000 * 1000, 1),
        "asr_input_ms": round(input_wav.shape[0] / 16000 * 1000, 1),
        "avg_total_ms": round(mean(item["total_ms"] for item in samples), 1) if samples else 0,
        "avg_asr_ms": round(mean(item["asr_ms"] for item in samples), 1) if samples else 0,
        "last": {
            **last,
            "total_ms": round(last.get("total_ms", 0), 1),
        } if last else {},
    }


def _bench_nlu(args):
    from nlu.engine import load_sessions, load_tokenizer, parse_nlu_output, predict

    started = time.perf_counter()
    tokenizer = load_tokenizer(args.nlu_tokenizer)
    enc_sess, dec_sess = load_sessions(args.nlu_model, num_threads=args.threads, use_gpu=args.gpu)
    load_ms = (time.perf_counter() - started) * 1000

    outputs = []
    timings = []
    for text in args.text:
        for _ in range(args.warmup):
            predict(enc_sess, dec_sess, tokenizer, text)
        samples = []
        raw = ""
        for _ in range(args.runs):
            started = time.perf_counter()
            raw = predict(enc_sess, dec_sess, tokenizer, text)
            samples.append((time.perf_counter() - started) * 1000)
        timings.extend(samples)
        outputs.append(
            {
                "text": text,
                "ms": round(mean(samples), 1),
                "raw": raw,
                "parsed": parse_nlu_output(raw),
            }
        )

    return {
        "load_ms": round(load_ms, 1),
        "avg_ms": round(mean(timings), 1) if timings else 0,
        "outputs": outputs,
    }


def main():
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Benchmark ASR/NLU inference")
    parser.add_argument("--asr-model", default="models/asr")
    parser.add_argument("--nlu-model", default="models/nlu")
    parser.add_argument("--nlu-tokenizer", default="models/nlu/tokenizer")
    parser.add_argument("--audio", default="audio/mabo.mp3")
    parser.add_argument("--text", action="append", default=["向前走", "向左转", "停下"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-nlu", action="store_true")
    parser.add_argument("--pipeline-fast-path", action="store_true", help="Benchmark ASR + rule parser without NLU")
    parser.add_argument("--pipeline-trim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-noise-rms", type=float, default=10.0)
    parser.add_argument("--asr-max-tokens", type=int, default=None)
    mel = parser.add_mutually_exclusive_group()
    mel.add_argument("--asr-fast-mel", dest="asr_fast_mel", action="store_true", default=None)
    mel.add_argument("--asr-librosa-mel", dest="asr_fast_mel", action="store_false")
    args = parser.parse_args()

    report = {}
    if args.pipeline_fast_path:
        report["pipeline_fast_path"] = _bench_pipeline_fast_path(args)
    if not args.no_asr and not args.pipeline_fast_path:
        report["asr"] = _bench_asr(args)
    if not args.no_nlu and not args.pipeline_fast_path:
        report["nlu"] = _bench_nlu(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
