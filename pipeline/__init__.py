"""
pipeline — 语音指令处理管道

唤醒词 → VAD → ASR(HTTP) → NLU(HTTP) → 指令 JSON → 音频反馈
"""

from .main import VoicePipeline, VAD_SAMPLE_RATE, run_webrtc


def __getattr__(name):
    if name == "run_onboard":
        from .onboard import run_onboard

        return run_onboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
