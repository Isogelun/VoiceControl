"""
ASR HTTP service and CLI entry point for Sherpa-NCNN (backend No.2).

Endpoint contract is identical to `asr.server` so the pipeline client
(`pipeline/asr_client.py`) is unaware of which engine is running:
    POST /asr   multipart/form-data: audio=WAV(16k/mono/int16), language, use_itn
                -> 200 {"text", "total_ms", ...} | 500 {"text": "", "error": ...}
    GET  /health -> {"status": "ok", "model": "sherpa-ncnn"}
"""

import argparse
import io
import logging
import os

from .engine import load_audio
from .ncnn_engine import load_session, transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "asr_ncnn")


def run_serve(engine, host="0.0.0.0", port=8000):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="Sherpa-NCNN ASR Service")

    @app.post("/asr")
    async def asr_endpoint(
        audio: UploadFile = File(...),
        language: str = Form("auto"),
        use_itn: bool = Form(True),
    ):
        try:
            data = await audio.read()
            wav = load_audio(io.BytesIO(data))
            result = transcribe(engine, wav, language, use_itn)
            logger.info("total=%sms | %s", result["total_ms"], result["text"][:60])
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("ASR inference failed")
            return JSONResponse(status_code=500, content={"text": "", "error": str(exc)})

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "sherpa-ncnn"}

    logger.info("ASR service started: http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_loop(engine, language="auto", use_itn=True):
    print("Sherpa-NCNN model is ready. Enter an audio path, or quit to exit.")
    print("-" * 50)
    while True:
        try:
            path = input("audio path> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not path or path.lower() == "quit":
            break
        try:
            wav = load_audio(path)
        except Exception as exc:
            print(f"failed to load audio: {exc}")
            continue
        result = transcribe(engine, wav, language, use_itn)
        print(f"text: {result['text']}")
        print(f"time: {result['total_ms']}ms")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Sherpa-NCNN speech recognition service")
    parser.add_argument("audio_path", nargs="?")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Start HTTP service")
    parser.add_argument("--loop", action="store_true", help="Interactive CLI mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logger.info("Loading ASR model (sherpa-ncnn): %s", args.model_dir)
    engine = load_session(args.model_dir, num_threads=args.threads)
    logger.info("ASR model loaded")

    if args.serve:
        run_serve(engine, args.host, args.port)
        return

    if args.loop:
        run_loop(engine, args.language, not args.no_itn)
        return

    if not args.audio_path:
        parser.error("audio_path is required unless --serve or --loop is used")

    wav = load_audio(args.audio_path)
    result = transcribe(engine, wav, args.language, not args.no_itn)
    print(f"text: {result['text']}")
    print(f"total: {result['total_ms']}ms")


if __name__ == "__main__":
    main()
