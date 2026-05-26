#!/usr/bin/env python3
"""
录音并立即回放的小工具。

用途:
1. 验证指定输入设备是否真的能采到声音
2. 验证指定输出设备是否真的能播出声音
3. 可选将录音保存为 WAV，便于后续回放

示例:
    python scripts/record_playback.py --list
    python scripts/record_playback.py --input-device 4 --duration 5
    python scripts/record_playback.py --input-device 4 --mic-channel 1 --output-device 27 --duration 5 --save-wav test.wav
"""

import argparse
import wave

import numpy as np
import sounddevice as sd


DEFAULT_SAMPLE_RATE = 16000


def list_devices():
    print("输入/输出设备列表:")
    for idx, dev in enumerate(sd.query_devices()):
        max_in = int(dev.get("max_input_channels", 0))
        max_out = int(dev.get("max_output_channels", 0))
        if max_in <= 0 and max_out <= 0:
            continue
        print(
            f"  [{idx}] {dev['name']} | in={max_in} out={max_out} "
            f"| default_samplerate={dev['default_samplerate']}"
        )


def save_wav(path, audio, sample_rate):
    with wave.open(path, "wb") as wf:
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser(description="录音并回放的小工具")
    parser.add_argument("--list", action="store_true", help="列出设备")
    parser.add_argument("--input-device", help="输入设备编号或名称")
    parser.add_argument("--output-device", help="输出设备编号或名称")
    parser.add_argument("--duration", type=int, default=5, help="录音时长（秒）")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=1, help="录音通道数")
    parser.add_argument("--mic-channel", type=int, default=0, help="从多通道输入中选取哪一路回放")
    parser.add_argument("--save-wav", help="保存录音到 WAV 文件")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    input_device = args.input_device
    output_device = args.output_device
    try:
        if input_device is not None:
            input_device = int(input_device)
    except ValueError:
        pass
    try:
        if output_device is not None:
            output_device = int(output_device)
    except ValueError:
        pass

    if input_device is None:
        parser.error("需要通过 --input-device 指定输入设备，或使用 --list 查看设备")

    input_info = sd.query_devices(input_device, kind="input")
    channels = max(1, min(args.channels, int(input_info["max_input_channels"])))
    if args.mic_channel < 0 or args.mic_channel >= channels:
        parser.error(f"--mic-channel 必须在 0 到 {channels - 1} 之间")

    frames = int(args.duration * args.sample_rate)
    print(
        f"开始录音: input_device={input_device} ({input_info['name']}), "
        f"channels={channels}, mic_channel={args.mic_channel}, duration={args.duration}s"
    )
    audio = sd.rec(
        frames,
        samplerate=args.sample_rate,
        channels=channels,
        dtype="int16",
        device=input_device,
        blocking=True,
    )
    print("录音完成")

    if audio.ndim > 1:
        selected = audio[:, args.mic_channel]
    else:
        selected = audio

    mean_abs = float(np.mean(np.abs(selected))) if selected.size else 0.0
    peak = int(np.max(np.abs(selected))) if selected.size else 0
    print(f"录音电平: mean_abs={mean_abs:.1f} peak={peak}")

    playback = selected.reshape(-1, 1)
    if args.save_wav:
        save_wav(args.save_wav, playback, args.sample_rate)
        print(f"已保存 WAV: {args.save_wav}")

    if output_device is not None:
        output_info = sd.query_devices(output_device, kind="output")
        print(f"开始回放: output_device={output_device} ({output_info['name']})")
    else:
        print("开始回放: 使用默认输出设备")

    sd.play(
        playback,
        samplerate=args.sample_rate,
        device=output_device,
        blocking=True,
    )
    print("回放完成")


if __name__ == "__main__":
    main()
