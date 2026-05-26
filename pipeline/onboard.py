"""
pipeline/onboard.py

部署在 Go2 EDU (Jetson Orin) 上的版本。
直接读本机麦克风，无需 WebRTC 连接。

用法:
    python run.py --onboard
"""

import asyncio
import logging
import os
import numpy as np
import sounddevice as sd

from .main import VoicePipeline, VAD_SAMPLE_RATE
from .cleaner import start_cleaner
from .audio_preprocessor import AudioPreprocessor

log = logging.getLogger(__name__)

MIC_BLOCKSIZE = 480  # 30ms @ 16kHz
MIC_DEVICE = os.environ.get("MIC_DEVICE")
MIC_CHANNEL = int(os.environ.get("MIC_CHANNEL", "0"))
MIC_LEVEL_LOG_INTERVAL = float(os.environ.get("MIC_LEVEL_LOG_INTERVAL", "3"))


def _resolve_input_device():
    """优先使用显式配置设备，其次优先 pulse，再回退到默认输入设备"""
    devices = sd.query_devices()
    input_devices = [
        (idx, dev) for idx, dev in enumerate(devices)
        if dev.get("max_input_channels", 0) > 0
    ]
    if not input_devices:
        raise RuntimeError("未找到可用输入设备")

    if MIC_DEVICE:
        try:
            return int(MIC_DEVICE)
        except ValueError:
            return MIC_DEVICE

    for idx, dev in input_devices:
        if dev["name"].strip().lower() == "pulse":
            return idx

    default_input = sd.default.device[0]
    if isinstance(default_input, int) and default_input >= 0:
        return default_input

    return input_devices[0][0]


async def run_onboard():
    """本机麦克风模式"""
    pipeline = VoicePipeline()
    preprocessor = AudioPreprocessor.from_env(
        sample_rate=VAD_SAMPLE_RATE,
        frame_samples=MIC_BLOCKSIZE,
    )
    loop = asyncio.get_running_loop()
    stream_errors = []
    audio_queue = asyncio.Queue(maxsize=100)
    mic_stats = {
        "mean_abs": 0.0,
        "peak": 0,
        "processed_mean_abs": 0.0,
        "processed_peak": 0,
        "noise_threshold": 0.0,
        "last_log": 0.0,
    }

    def _enqueue_pcm(pcm: np.ndarray):
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        audio_queue.put_nowait(pcm)

    def _mic_callback(indata, frames, time, status):
        try:
            if status:
                log.warning("麦克风回调状态异常: %s", status)
            pcm = indata[:, MIC_CHANNEL] if getattr(indata, "ndim", 1) > 1 else indata
            pcm = np.asarray(pcm, dtype=np.int16).copy()
            mic_stats["mean_abs"] = float(np.mean(np.abs(pcm)))
            mic_stats["peak"] = int(np.max(np.abs(pcm))) if pcm.size else 0
            processed = preprocessor.process(pcm)
            mic_stats["processed_mean_abs"] = float(np.mean(np.abs(processed)))
            mic_stats["processed_peak"] = int(np.max(np.abs(processed))) if processed.size else 0
            mic_stats["noise_threshold"] = preprocessor.last_threshold
            loop.call_soon_threadsafe(_enqueue_pcm, processed)
        except Exception as exc:
            stream_errors.append(str(exc))

    try:
        try:
            input_device = _resolve_input_device()
            input_info = sd.query_devices(input_device, kind="input")
            if MIC_CHANNEL >= input_info["max_input_channels"]:
                raise RuntimeError(
                    f"输入设备 {input_device} 仅支持 {input_info['max_input_channels']} 路，"
                    f"当前 MIC_CHANNEL={MIC_CHANNEL} 超出范围"
                )
            log.info(
                "使用输入设备: %s (device=%s, max_input_channels=%s, default_samplerate=%s, mic_channel=%s)",
                input_info["name"],
                input_device,
                input_info["max_input_channels"],
                input_info["default_samplerate"],
                MIC_CHANNEL,
            )
        except Exception:
            log.exception("选择输入设备失败")
            raise

        with sd.InputStream(
            device=input_device,
            samplerate=VAD_SAMPLE_RATE,
            channels=MIC_CHANNEL + 1,
            dtype="int16",
            blocksize=MIC_BLOCKSIZE,
            callback=_mic_callback,
        ) as stream:
            log.info(
                "麦克风流已打开: samplerate=%s channels=%s blocksize=%s active=%s selected_channel=%s",
                stream.samplerate,
                stream.channels,
                stream.blocksize,
                stream.active,
                MIC_CHANNEL,
            )
            asyncio.create_task(start_cleaner())
            log.info("麦克风已启动，等待唤醒词...")
            log.info(
                "音频降噪: enabled=%s gain=%.2f calibrating_frames=%s",
                preprocessor.enabled,
                preprocessor.gain,
                preprocessor.calibration_frames,
            )
            while True:
                try:
                    pcm = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    await pipeline.push_pcm(pcm)
                except asyncio.TimeoutError:
                    pass

                now = loop.time()
                if now - mic_stats["last_log"] >= MIC_LEVEL_LOG_INTERVAL:
                    mic_stats["last_log"] = now
                    log.info(
                        "麦克风电平: raw_mean=%.1f raw_peak=%d processed_mean=%.1f processed_peak=%d threshold=%.1f channel=%s",
                        mic_stats["mean_abs"],
                        mic_stats["peak"],
                        mic_stats["processed_mean_abs"],
                        mic_stats["processed_peak"],
                        mic_stats["noise_threshold"],
                        MIC_CHANNEL,
                    )
                if stream_errors:
                    raise RuntimeError(f"麦克风回调异常: {stream_errors[-1]}")
                if stream.closed:
                    raise RuntimeError("麦克风流已关闭")
                if not stream.active:
                    raise RuntimeError("麦克风流未处于活动状态")
    except KeyboardInterrupt:
        log.info("收到退出信号，停止麦克风监听")
    except Exception:
        log.exception("Onboard 麦克风监听异常退出")
        raise
    finally:
        log.info("Onboard 麦克风监听结束")
        pipeline.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run_onboard())
