#!/usr/bin/env python3
"""
扬声器测试工具。

功能:
1. 列出所有可用输出设备
2. 在指定输出设备播放一段测试音

示例:
    python scripts/speaker_test.py --list
    python scripts/speaker_test.py
    python scripts/speaker_test.py --output-device 27 --frequency 880 --duration 3
"""

import argparse

import numpy as np
import sounddevice as sd


DEFAULT_SAMPLE_RATE = 16000


def list_output_devices():
    print("可用输出设备:")
    found = False
    for idx, dev in enumerate(sd.query_devices()):
        max_out = int(dev.get("max_output_channels", 0))
        if max_out <= 0:
            continue
        found = True
        print(
            f"  [{idx}] {dev['name']} | max_output_channels={max_out} "
            f"| default_samplerate={dev['default_samplerate']}"
        )
    if not found:
        print("  未找到可用输出设备")


def make_tone(sample_rate, frequency, duration, volume):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    return tone.astype(np.float32).reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="扬声器测试工具")
    parser.add_argument("--list", action="store_true", help="列出所有可用输出设备")
    parser.add_argument("--output-device", help="输出设备编号或名称")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frequency", type=float, default=660.0, help="测试音频率，单位 Hz")
    parser.add_argument("--duration", type=float, default=2.0, help="播放时长，单位秒")
    parser.add_argument("--volume", type=float, default=0.2, help="音量，范围建议 0.0~1.0")
    args = parser.parse_args()

    if args.list:
        list_output_devices()
        return

    output_device = args.output_device
    try:
        if output_device is not None:
            output_device = int(output_device)
    except ValueError:
        pass

    if output_device is not None:
        output_info = sd.query_devices(output_device, kind="output")
        print(
            f"使用输出设备: {output_info['name']} "
            f"(device={output_device}, max_output_channels={output_info['max_output_channels']})"
        )
    else:
        print("使用默认输出设备")

    tone = make_tone(
        sample_rate=args.sample_rate,
        frequency=args.frequency,
        duration=args.duration,
        volume=args.volume,
    )

    print(
        f"开始播放测试音: frequency={args.frequency}Hz duration={args.duration}s "
        f"volume={args.volume}"
    )
    sd.play(tone, samplerate=args.sample_rate, device=output_device, blocking=True)
    print("播放完成")


if __name__ == "__main__":
    main()
