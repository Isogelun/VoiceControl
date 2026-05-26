"""
nlu/server.py

NLU HTTP 服务 + CLI 入口。
可独立运行：python -m nlu.server --serve --port 8001
"""

import argparse
import json
import logging
import os

from .engine import load_sessions, load_tokenizer, predict, parse_nlu_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "nlu")
DEFAULT_TOKENIZER_DIR = os.path.join(_MODULE_DIR, "tokenizer")


def run_serve(enc_sess, dec_sess, tokenizer, host="0.0.0.0", port=8001):
    """启动 FastAPI HTTP 服务"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="NLU Service")

    class NLURequest(BaseModel):
        text: str

    @app.post("/nlu")
    async def nlu_endpoint(req: NLURequest):
        raw_output = predict(enc_sess, dec_sess, tokenizer, req.text)
        result = parse_nlu_output(raw_output)
        logger.info(f"NLU: '{req.text}' -> {result}")
        return JSONResponse(content=result)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    logger.info(f"NLU 服务启动: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="NLU 自然语言理解服务")
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logger.info("加载 NLU 模型: %s", args.model_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    enc_sess, dec_sess = load_sessions(args.model_dir)

    if args.serve:
        run_serve(enc_sess, dec_sess, tokenizer, args.host, args.port)
        return

    if args.texts:
        for t in args.texts:
            raw = predict(enc_sess, dec_sess, tokenizer, t)
            result = parse_nlu_output(raw)
            print(f"输入: {t}\n输出: {raw}\n解析: {json.dumps(result, ensure_ascii=False)}\n")
    else:
        print("=== 交互模式，输入 quit 退出 ===")
        while True:
            try:
                t = input("输入> ").strip()
            except EOFError:
                break
            if t.lower() in ("quit", "exit", "q") or not t:
                break
            raw = predict(enc_sess, dec_sess, tokenizer, t)
            result = parse_nlu_output(raw)
            print(f"输出> {raw}")
            print(f"解析> {json.dumps(result, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
