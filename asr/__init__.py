"""
asr — SenseVoiceSmall 语音识别模块

用法:
    # 作为包导入
    from asr import load_session, load_tokens, load_audio, transcribe

    # 作为 HTTP 服务启动
    python -m asr.server --serve --port 8000
"""

from .engine import (
    load_session,
    load_tokens,
    load_audio,
    extract_features,
    transcribe,
    SAMPLE_RATE,
    LANG_IDS,
)
