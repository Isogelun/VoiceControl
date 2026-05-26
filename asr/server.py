"""
asr/server.py

ASR HTTP 服务 + CLI 入口。
可独立运行：python -m asr.server --serve --port 8000
"""

import io
import argparse
import logging
import os

from .engine import load_session, load_tokens, load_audio, transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "asr")


def run_serve(sess, neg_mean, inv_stddev, tokens, host="0.0.0.0", port=8000):
    """启动 FastAPI HTTP 服务"""
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="ASR Service")

    @app.post("/asr")
    async def asr_endpoint(audio: UploadFile = File(...), language: str = Form("auto"),
                           use_itn: bool = Form(True)):
        data = await audio.read()
        wav = load_audio(io.BytesIO(data))
        result = transcribe(sess, neg_mean, inv_stddev, tokens, wav, language, use_itn)
        logger.info(f"total={result['total_ms']}ms | {result['text'][:60]}")
        return JSONResponse(content=result)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    logger.info(f"ASR 服务启动: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_loop(sess, neg_mean, inv_stddev, tokens, language="auto", use_itn=True):
    """交互模式：模型只加载一次，反复输入音频路径进行识别"""
    print("模型已就绪，输入音频文件路径开始识别（输入 quit 退出）")
    print("-" * 50)
    while True:
        try:
            path = input("音频路径> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not path or path.lower() == "quit":
            break
        try:
            wav = load_audio(path)
        except Exception as e:
            print(f"加载音频失败: {e}")
            continue
        result = transcribe(sess, neg_mean, inv_stddev, tokens, wav, language, use_itn)
        print(f"文本: {result['text']}")
        print(f"耗时: {result['total_ms']}ms (特征 {result['feat_ms']}ms + 推理 {result['infer_ms']}ms)")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="ASR 语音识别服务")
    parser.add_argument("audio_path", nargs="?")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--full", action="store_true", help="使用全精度模型")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 推理")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--loop", action="store_true", help="交互模式")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logger.info("加载 ASR 模型: %s", args.model_dir)
    sess, neg_mean, inv_stddev = load_session(
        args.model_dir, use_q8=not args.full,
        num_threads=args.threads, use_gpu=args.gpu,
    )
    tokens = load_tokens(args.model_dir)
    logger.info("ASR 模型加载完成")

    if args.serve:
        run_serve(sess, neg_mean, inv_stddev, tokens, args.host, args.port)
        return

    if args.loop:
        run_loop(sess, neg_mean, inv_stddev, tokens, args.language, not args.no_itn)
        return

    if not args.audio_path:
        parser.error("需要指定音频文件路径，或使用 --serve / --loop")

    wav = load_audio(args.audio_path)
    result = transcribe(sess, neg_mean, inv_stddev, tokens, wav, args.language, not args.no_itn)
    print(f"文本: {result['text']}")
    print(f"总耗时: {result['total_ms']}ms  特征: {result['feat_ms']}ms  推理: {result['infer_ms']}ms")


if __name__ == "__main__":
    main()
