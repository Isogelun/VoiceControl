"""
对比 Python 和 Rust 推理服务的输出。

用法:
    # 先分别启动 Python 和 Rust 服务 (不同端口)
    python -m asr.server --serve --port 8000 &
    python -m nlu.server --serve --port 8001 &
    ./voice-infer --asr-port 9000 --nlu-port 9001 &

    python scripts/compare_outputs.py \
        --py-asr http://localhost:8000 \
        --py-nlu http://localhost:8001 \
        --rs-asr http://localhost:9000 \
        --rs-nlu http://localhost:9001 \
        --audio tests/fixtures/test_1s.wav
"""

import argparse
import io
import json
import os
import sys
import time

import requests
import numpy as np


def compare_asr(py_url, rs_url, audio_path):
    """发送同一段音频到两个 ASR 服务，对比结果。"""
    print(f"\n{'='*60}")
    print(f"ASR 对比: {audio_path}")
    print(f"  Python: {py_url}/asr")
    print(f"  Rust:   {rs_url}/asr")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    files = {"audio": ("test.wav", io.BytesIO(audio_data), "audio/wav")}
    data = {"language": "auto", "use_itn": "true"}

    t0 = time.perf_counter()
    py_resp = requests.post(f"{py_url}/asr", files=files, data=data)
    py_ms = (time.perf_counter() - t0) * 1000
    py_result = py_resp.json()

    files = {"audio": ("test.wav", io.BytesIO(audio_data), "audio/wav")}
    t0 = time.perf_counter()
    rs_resp = requests.post(f"{rs_url}/asr", files=files, data=data)
    rs_ms = (time.perf_counter() - t0) * 1000
    rs_result = rs_resp.json()

    py_text = py_result.get("text", "")
    rs_text = rs_result.get("text", "")
    match = py_text == rs_text

    print(f"\n  Python text: {py_text!r}")
    print(f"  Rust   text: {rs_text!r}")
    print(f"  文本一致: {'YES' if match else 'NO <<<'}")
    print(f"\n  Python total_ms: {py_result.get('total_ms')}")
    print(f"  Rust   total_ms: {rs_result.get('total_ms')}")
    print(f"  Python 请求耗时: {py_ms:.1f}ms")
    print(f"  Rust   请求耗时: {rs_ms:.1f}ms")

    return match


def compare_nlu(py_url, rs_url, texts):
    """对比多组 NLU 输入。"""
    print(f"\n{'='*60}")
    print(f"NLU 对比")
    print(f"  Python: {py_url}/nlu")
    print(f"  Rust:   {rs_url}/nlu")

    all_match = True
    for text in texts:
        payload = {"text": text}
        py_resp = requests.post(f"{py_url}/nlu", json=payload)
        rs_resp = requests.post(f"{rs_url}/nlu", json=payload)
        py_r = py_resp.json()
        rs_r = rs_resp.json()

        py_intent = py_r.get("intent", "")
        rs_intent = rs_r.get("intent", "")
        match = py_intent == rs_intent

        status = "OK" if match else "MISMATCH <<<"
        print(f"\n  输入: {text!r}")
        print(f"    Python: intent={py_intent}, slots={py_r.get('slots', {})}")
        print(f"    Rust:   intent={rs_intent}, slots={rs_r.get('slots', {})}")
        print(f"    {status}")

        if not match:
            all_match = False

    return all_match


def check_health(url, name):
    """检查服务健康状态。"""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        ok = resp.status_code == 200
        print(f"  {name} {url}/health: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  {name} {url}/health: UNREACHABLE ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="对比 Python/Rust 推理输出")
    parser.add_argument("--py-asr", default="http://localhost:8000")
    parser.add_argument("--py-nlu", default="http://localhost:8001")
    parser.add_argument("--rs-asr", default="http://localhost:9000")
    parser.add_argument("--rs-nlu", default="http://localhost:9001")
    parser.add_argument("--audio", default="tests/fixtures/test_1s.wav",
                        help="测试音频路径 (可逗号分隔多个)")
    parser.add_argument("--nlu-texts", default="向前走三步,停止,坐下,左转,后退两步,打招呼,伸懒腰",
                        help="NLU 测试文本 (逗号分隔)")
    args = parser.parse_args()

    print("检查服务状态...")
    services_ok = True
    for url, name in [
        (args.py_asr, "Python ASR"),
        (args.py_nlu, "Python NLU"),
        (args.rs_asr, "Rust ASR"),
        (args.rs_nlu, "Rust NLU"),
    ]:
        if not check_health(url, name):
            services_ok = False

    if not services_ok:
        print("\n部分服务不可达，请确认已启动。")
        sys.exit(1)

    # ASR 对比
    audio_paths = [p.strip() for p in args.audio.split(",")]
    asr_ok = True
    for path in audio_paths:
        if not os.path.isfile(path):
            print(f"\n  跳过不存在的音频: {path}")
            continue
        if not compare_asr(args.py_asr, args.rs_asr, path):
            asr_ok = False

    # NLU 对比
    nlu_texts = [t.strip() for t in args.nlu_texts.split(",") if t.strip()]
    nlu_ok = compare_nlu(args.py_nlu, args.rs_nlu, nlu_texts)

    # 总结
    print(f"\n{'='*60}")
    print(f"ASR 对比: {'ALL MATCH' if asr_ok else 'HAS MISMATCH'}")
    print(f"NLU 对比: {'ALL MATCH' if nlu_ok else 'HAS MISMATCH'}")

    if asr_ok and nlu_ok:
        print("\n所有输出一致，Rust 重写验证通过!")
        sys.exit(0)
    else:
        print("\n存在不一致，需要排查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
