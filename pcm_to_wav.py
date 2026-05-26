#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCM 音频播放工具
PCM Audio Playback Tool

用于在 Windows 上播放从讯飞六麦阵列录制的 PCM 音频
Used to play PCM audio recorded from the Xunfei 6-mic array on Windows
"""

import wave
import sys
import os

def pcm_to_wav(pcm_file: str, wav_file: str, sample_rate: int = 16000, channels: int = 1):
    """
    将 PCM 转换为 WAV 格式
    Convert PCM to WAV format
    """
    with open(pcm_file, 'rb') as pcm:
        pcm_data = pcm.read()
    
    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit / 16位采样
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    
    print(f"已转换: {pcm_file} -> {wav_file}")
    print(f"参数: {sample_rate}Hz, {channels}声道, 16-bit")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PCM 转 WAV 工具')
    parser.add_argument('pcm_file', nargs='?', default='audio.pcm', help='PCM 文件路径')
    parser.add_argument('-r', '--rate', type=int, default=16000, help='采样率 (默认: 16000)')
    parser.add_argument('-c', '--channels', type=int, default=6, help='声道数 (默认: 6，六麦阵列)')
    parser.add_argument('-o', '--output', default=None, help='输出 WAV 文件路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.pcm_file):
        print(f"错误: 文件不存在 - {args.pcm_file}")
        sys.exit(1)
    
    wav_file = args.output or args.pcm_file.replace('.pcm', '.wav')
    pcm_to_wav(args.pcm_file, wav_file, args.rate, args.channels)
    
    print(f"\n可以用任意播放器打开: {wav_file}")

if __name__ == '__main__':
    main()
