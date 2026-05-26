"""
pipeline/cleaner.py

定时清理指令 JSON、临时音频、日志文件（保留最近 24 小时）。
每小时执行一次，作为 asyncio 后台任务运行。

环境变量:
    PIPELINE_OUTPUT_DIR   指令 JSON 输出目录，默认 ./output
    PIPELINE_TEMP_DIR     临时音频目录，默认 ./tmp
    PIPELINE_LOG_DIR      日志目录，默认 ./logs
    CLEAN_RETAIN_HOURS    保留时长（小时），默认 24
    CLEAN_INTERVAL_SECS   清理间隔（秒），默认 3600
"""

import asyncio
import logging
import os
import time

log = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "output")
TEMP_DIR = os.environ.get("PIPELINE_TEMP_DIR", "tmp")
LOG_DIR = os.environ.get("PIPELINE_LOG_DIR", "logs")
RETAIN_SECONDS = int(os.environ.get("CLEAN_RETAIN_HOURS", 24)) * 3600
INTERVAL = int(os.environ.get("CLEAN_INTERVAL_SECS", 3600))

_WATCH_DIRS = {
    OUTPUT_DIR: {".json"},
    TEMP_DIR: {".wav", ".pcm"},
    LOG_DIR: {".log"},
}


def _clean_once():
    now = time.time()
    removed = 0
    for directory, exts in _WATCH_DIRS.items():
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if not any(fname.endswith(e) for e in exts):
                continue
            fpath = os.path.join(directory, fname)
            try:
                if now - os.path.getmtime(fpath) > RETAIN_SECONDS:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
    if removed:
        log.info("清理完成，删除 %d 个文件", removed)


async def start_cleaner():
    """在 main() 中 asyncio.create_task(start_cleaner()) 启动"""
    while True:
        await asyncio.sleep(INTERVAL)
        _clean_once()
