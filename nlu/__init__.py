"""
nlu — Mengzi-T5 自然语言理解模块

用法:
    # 作为包导入
    from nlu import load_sessions, load_tokenizer, predict, parse_nlu_output

    # 作为 HTTP 服务启动
    python -m nlu.server --serve --port 8001
"""

from .engine import (
    load_sessions,
    load_tokenizer,
    predict,
    parse_nlu_output,
)
