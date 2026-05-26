"""
pipeline — 语音指令处理管道

唤醒词 → VAD → ASR(HTTP) → NLU(HTTP) → 指令 JSON → 音频反馈
"""

from .main import VoicePipeline, VAD_SAMPLE_RATE, run_webrtc
from .onboard import run_onboard
