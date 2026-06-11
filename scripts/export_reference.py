"""
导出 Python 推理管线的中间结果，作为 Rust 重写的 ground truth。

用法:
    python scripts/export_reference.py [--out tests/fixtures/reference]

生成文件列表:
    mel/
        hann_window.npy          Hann 窗 [400]
        mel_basis.npy            Mel 滤波器组 [128, 201]
        power_frame0.npy         第 0 帧 power spectrum [201]
        full_mel.npy             完整 log-mel [1, 128, T]
        mel_config.json          mel 参数快照
    asr/
        prompt_prefix_ids.json   build_prompt prefix token ids
        prompt_suffix_ids.json   build_prompt suffix token ids
        audio_features_head.npy  encoder 输出前 8 列 [1, 8, 1024]
        init_logits_last.npy     decoder_init 最后位置 logits [151936]
        embed_token0.npy         embed_tokens[0] (f32) [1024]
        embed_token100.npy       embed_tokens[100] (f32) [1024]
        transcribe_result.json   最终端到端结果
    nlu/
        encode_input.json        tokenizer("指令解析: 向前走三步") → ids + mask
        hidden_head.npy          encoder hidden_states[:, :5, :5] 角落值
        predict_forward.json     predict("向前走三步") 结果
        predict_stop.json        predict("停止") 结果
        predict_sit.json         predict("坐下") 结果
    parse/
        cases.json               10 组 (raw_input, expected_output) 解析测试
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    print(f"  saved {path}  shape={arr.shape}  dtype={arr.dtype}")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  saved {path}")


def generate_test_audio(duration_s=1.0, sr=16000):
    """合成一段包含语音特征的测试音频 (440Hz + 白噪声)"""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = 0.02 * np.random.RandomState(42).randn(len(t)).astype(np.float32)
    return tone + noise


def export_mel(out_dir, test_wav):
    """导出 mel 频谱各阶段的参考值"""
    print("\n=== Mel 频谱参考值 ===")
    mel_dir = os.path.join(out_dir, "mel")

    from asr.engine import Qwen3ASREngine
    import librosa

    model_dir = os.path.join(PROJECT_ROOT, "models", "asr")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mel_cfg = config.get("mel", {})
    n_fft = int(mel_cfg.get("n_fft", 400))
    hop_length = int(mel_cfg.get("hop_length", 160))
    n_mels = int(mel_cfg.get("n_mels", 128))
    fmin = float(mel_cfg.get("fmin", 0))
    fmax = float(mel_cfg.get("fmax", 8000))
    sr = 16000

    save_json(os.path.join(mel_dir, "mel_config.json"), {
        "n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels,
        "fmin": fmin, "fmax": fmax, "sample_rate": sr,
    })

    # Hann 窗 (periodic)
    n = np.arange(n_fft, dtype=np.float32)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)).astype(np.float32)
    save_npy(os.path.join(mel_dir, "hann_window.npy"), window)

    # Mel 滤波器组
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels,
        fmin=fmin, fmax=fmax, htk=False, norm="slaney",
    ).astype(np.float32)
    save_npy(os.path.join(mel_dir, "mel_basis.npy"), mel_basis)

    # 完整 log_mel_fast 流程，逐步导出
    wav = test_wav.copy()
    save_npy(os.path.join(mel_dir, "input_wav.npy"), wav.astype(np.float32, copy=False))
    pad = n_fft // 2
    padded = np.pad(wav, (pad, pad), mode="constant")
    n_frames = 1 + (padded.size - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop_length, padded.strides[0]),
        writeable=False,
    )
    windowed = frames * window
    power = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) ** 2

    # 第 0 帧 power spectrum
    save_npy(os.path.join(mel_dir, "power_frame0.npy"), power[0].astype(np.float32))

    mel = np.dot(mel_basis, power.T)
    log_mel = np.log10(np.maximum(mel, 1e-10))
    save_npy(os.path.join(mel_dir, "log_mel_pre_norm.npy"), log_mel.astype(np.float32, copy=False))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    full_mel = log_mel[None].astype(np.float32, copy=False)
    save_npy(os.path.join(mel_dir, "full_mel.npy"), full_mel)

    return full_mel


def export_asr(out_dir, test_wav, mel):
    """导出 ASR 推理各阶段的参考值"""
    print("\n=== ASR 参考值 ===")
    asr_dir = os.path.join(out_dir, "asr")

    from asr.engine import Qwen3ASREngine
    model_dir = os.path.join(PROJECT_ROOT, "models", "asr")
    engine = Qwen3ASREngine(model_dir, num_threads=2, use_gpu=False)

    # Prompt 构建
    prefix_ids = engine._token_ids(
        "<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>"
    )
    suffix_ids = engine._token_ids(
        "<|audio_end|><|im_end|><|im_start|>assistant"
    )
    save_json(os.path.join(asr_dir, "prompt_prefix_ids.json"), prefix_ids)
    save_json(os.path.join(asr_dir, "prompt_suffix_ids.json"), suffix_ids)

    # Embedding 查找
    embed0 = np.asarray(engine.embed_tokens[0], dtype=np.float32)
    embed100 = np.asarray(engine.embed_tokens[100], dtype=np.float32)
    save_npy(os.path.join(asr_dir, "embed_token0.npy"), embed0)
    save_npy(os.path.join(asr_dir, "embed_token100.npy"), embed100)

    # Encoder
    audio_features, = engine.encoder.run(None, {"mel": mel})
    save_npy(
        os.path.join(asr_dir, "audio_features_head.npy"),
        audio_features[:, :min(8, audio_features.shape[1]), :].astype(np.float32),
    )

    # Decoder init
    audio_len = int(audio_features.shape[1])
    input_ids, audio_offset = engine._build_prompt(audio_len)
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

    logits, _, _ = engine.decoder_init.run(None, {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "audio_features": audio_features.astype(np.float32, copy=False),
        "audio_offset": audio_offset,
    })
    save_npy(
        os.path.join(asr_dir, "init_logits_last.npy"),
        logits[0, -1, :].astype(np.float32),
    )

    # 端到端
    result = engine.transcribe(test_wav)
    save_json(os.path.join(asr_dir, "transcribe_result.json"), result)
    print(f"  ASR result: {result['text']!r}  ({result['total_ms']}ms)")


def export_nlu(out_dir):
    """导出 NLU 推理各阶段的参考值"""
    print("\n=== NLU 参考值 ===")
    nlu_dir = os.path.join(out_dir, "nlu")

    from nlu.engine import load_sessions, load_tokenizer, predict, parse_nlu_output

    model_dir = os.path.join(PROJECT_ROOT, "models", "nlu")
    tokenizer_dir = os.path.join(model_dir, "tokenizer")

    tokenizer = load_tokenizer(tokenizer_dir)
    enc_sess, dec_sess = load_sessions(model_dir, num_threads=2)

    # Tokenizer encode
    inputs = tokenizer(
        "指令解析: 向前走三步",
        return_tensors="np", max_length=64,
        padding="max_length", truncation=True,
    )
    save_json(os.path.join(nlu_dir, "encode_input.json"), {
        "input_ids": inputs["input_ids"][0].astype(int).tolist(),
        "attention_mask": inputs["attention_mask"][0].astype(int).tolist(),
    })

    # Encoder hidden states (角落值用于 spot check)
    import onnxruntime as ort
    enc_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    hidden = enc_sess.run(
        ["last_hidden_state"],
        {"input_ids": enc_inputs["input_ids"],
         "attention_mask": enc_inputs["attention_mask"]},
    )[0]
    save_npy(
        os.path.join(nlu_dir, "hidden_head.npy"),
        hidden[:, :5, :5].astype(np.float32),
    )

    # 端到端 predict
    test_cases = [
        ("向前走三步", "predict_forward.json"),
        ("停止", "predict_stop.json"),
        ("坐下", "predict_sit.json"),
    ]
    for text, fname in test_cases:
        raw = predict(enc_sess, dec_sess, tokenizer, text)
        result = parse_nlu_output(raw)
        save_json(os.path.join(nlu_dir, fname), {
            "input": text,
            "raw_output": raw,
            "parsed": result,
        })
        print(f"  NLU '{text}' → intent={result['intent']}")


def export_parse_cases(out_dir):
    """导出 NLU 输出解析的测试用例"""
    print("\n=== 解析测试用例 ===")

    from nlu.engine import parse_nlu_output

    cases = [
        # 1. JSON cmd — MoveForward
        '{"type":"cmd","payload":{"command_type":"MoveForward","payload_json":{"vx":0.3}}}',
        # 2. JSON cmd — Move with vx/vy/vyaw (方向推断)
        '{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.3,"vyaw":0.0}}}',
        # 3. JSON cmd — Move with vyaw (转弯)
        '{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.0,"vyaw":0.5}}}',
        # 4. JSON cmd — Sit
        '{"type":"cmd","payload":{"command_type":"Sit","payload_json":{}}}',
        # 5. JSON cmd — StopMove
        '{"type":"cmd","payload":{"command_type":"StopMove","payload_json":{}}}',
        # 6. JSON chat
        '{"type":"chat","payload":{"message":"你好，我是曼波"}}',
        # 7. JSON intent/slots 格式
        '{"intent":"move_forward","slots":{"direction":"forward","steps":3}}',
        # 8. JSON intent, slots=None
        '{"intent":"stop"}',
        # 9. key=value 格式
        'intent=stop, direction=none',
        # 10. 纯文本 fallback
        '你好世界',
        # 11. 空字符串
        '',
        # 12. JSON cmd — RecoveryStand
        '{"type":"cmd","payload":{"command_type":"RecoveryStand","payload_json":{}}}',
        # 13. JSON cmd — Move 全零 (edge case)
        '{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.0,"vyaw":0.0}}}',
    ]

    results = []
    for raw in cases:
        parsed = parse_nlu_output(raw)
        results.append({"input": raw, "expected": parsed})
        print(f"  '{raw[:50]}...' → intent={parsed['intent']}")

    save_json(os.path.join(out_dir, "parse", "cases.json"), results)


def export_test_audio(out_dir):
    """导出测试音频文件"""
    print("\n=== 测试音频 ===")
    import soundfile as sf

    fixture_dir = os.path.join(out_dir, "..")

    # 16kHz mono 1s
    wav_1s = generate_test_audio(1.0, 16000)
    path_1s = os.path.join(fixture_dir, "test_1s.wav")
    sf.write(path_1s, wav_1s, 16000, subtype="FLOAT")
    print(f"  saved {path_1s}  samples={len(wav_1s)}")

    # 16kHz mono 0.5s silence
    wav_silence = np.zeros(8000, dtype=np.float32)
    path_silence = os.path.join(fixture_dir, "test_silence.wav")
    sf.write(path_silence, wav_silence, 16000, subtype="FLOAT")
    print(f"  saved {path_silence}  samples={len(wav_silence)}")

    # 8kHz stereo (for resample + channel merge test)
    rng = np.random.RandomState(123)
    wav_stereo = rng.randn(8000, 2).astype(np.float32) * 0.1
    path_stereo = os.path.join(fixture_dir, "test_8k_stereo.wav")
    sf.write(path_stereo, wav_stereo, 8000, subtype="FLOAT")
    print(f"  saved {path_stereo}  samples={wav_stereo.shape}")

    return wav_1s


def main():
    parser = argparse.ArgumentParser(description="导出 Rust 重写参考数据")
    parser.add_argument(
        "--out", default=os.path.join(PROJECT_ROOT, "tests", "fixtures", "reference"),
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # Step 1: 测试音频
    test_wav = export_test_audio(out_dir)

    # Step 2: Mel 频谱
    mel = export_mel(out_dir, test_wav)

    # Step 3: ASR
    export_asr(out_dir, test_wav, mel)

    # Step 4: NLU
    export_nlu(out_dir)

    # Step 5: 解析测试用例
    export_parse_cases(out_dir)

    print(f"\n完成! 所有参考数据已导出到 {out_dir}")


if __name__ == "__main__":
    main()
