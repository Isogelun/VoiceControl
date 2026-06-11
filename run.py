#!/usr/bin/env python3
"""
VoiceControl 统一入口

用法:
    python run.py                      # 默认：启动全部（ASR + NLU + 本机麦克风）
    python run.py --webrtc             # WebRTC 模式（ASR + NLU + Go2 音频）
    python run.py --serve-asr          # 仅启动 ASR HTTP 服务
    python run.py --serve-nlu          # 仅启动 NLU HTTP 服务
    python run.py --pipeline-only      # 仅启动 Pipeline（ASR/NLU 服务已在其他地方运行）
"""

import argparse
import asyncio
import atexit
import faulthandler
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import multiprocessing
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run")
faulthandler.enable(all_threads=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASR_MODEL = os.path.join(PROJECT_ROOT, "models", "asr")
DEFAULT_NLU_MODEL = os.path.join(PROJECT_ROOT, "models", "nlu")
DEFAULT_NLU_TOKENIZER = os.path.join(DEFAULT_NLU_MODEL, "tokenizer")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_SERVICE_TIMEOUT = float(os.environ.get("SERVICE_START_TIMEOUT", "120"))
CHILD_PROCS = []
_CLEANING_UP = False
RUST_BINARY_NAMES = ("voice-infer.exe", "voice-infer")

CONFIG_ENV_MAP = {
    ("robot", "ip"): "UNITREE_ROBOT_IP",
    ("robot", "connection_method"): "UNITREE_WEBRTC_METHOD",
    ("robot", "serial_number"): "UNITREE_ROBOT_SERIAL_NUMBER",
    ("robot", "aes_128_key"): "UNITREE_AES_128_KEY",
    ("robot", "username"): "UNITREE_USERNAME",
    ("robot", "password"): "UNITREE_PASSWORD",
    ("robot", "region"): "UNITREE_REGION",
    ("robot", "device_type"): "UNITREE_DEVICE_TYPE",
    ("robot", "datachannel_timeout"): "UNITREE_WEBRTC_TIMEOUT",
    ("robot", "connect_retries"): "UNITREE_WEBRTC_CONNECT_RETRIES",
    ("robot", "retry_delay_ms"): "UNITREE_WEBRTC_RETRY_DELAY_MS",
    ("robot", "signaling_timeout"): "UNITREE_SIGNALING_TIMEOUT",
    ("robot", "set_remote_timeout"): "UNITREE_SET_REMOTE_TIMEOUT",
    ("robot", "filter_remote_candidates"): "UNITREE_FILTER_REMOTE_CANDIDATES",
    ("robot", "local_ip"): "UNITREE_WEBRTC_LOCAL_IP",
    ("robot", "auto_local_ip"): "UNITREE_WEBRTC_AUTO_LOCAL_IP",
    ("kws", "model_dir"): "KWS_MODEL_DIR",
    ("wake", "keyword"): "WAKE_KEYWORD",
    ("wake", "backend"): "WAKE_BACKEND",
    ("wake", "text"): "WAKE_TEXT",
    ("wake", "aliases"): "WAKE_ALIASES",
    ("wake", "audio"): "WAKE_AUDIO",
    ("wake", "feedback_enabled"): "WAKE_FEEDBACK_ENABLED",
    ("services", "asr_url"): "ASR_URL",
    ("services", "asr_timeout"): "ASR_TIMEOUT",
    ("services", "asr_retries"): "ASR_RETRIES",
    ("services", "asr_retry_delay_ms"): "ASR_RETRY_DELAY_MS",
    ("services", "asr_max_new_tokens"): "QWEN_ASR_MAX_NEW_TOKENS",
    ("services", "nlu_url"): "NLU_URL",
    ("services", "nlu_timeout"): "NLU_TIMEOUT",
    ("services", "nlu_retries"): "NLU_RETRIES",
    ("services", "nlu_retry_delay_ms"): "NLU_RETRY_DELAY_MS",
    ("services", "nlu_structured_early_stop"): "NLU_STRUCTURED_EARLY_STOP",
    ("command", "output_dir"): "COMMAND_OUTPUT_DIR",
    ("command", "service_url"): "COMMAND_SERVICE_URL",
    ("command", "service_timeout"): "COMMAND_SERVICE_TIMEOUT",
    ("command", "fast_response"): "COMMAND_FAST_RESPONSE",
    ("command", "move_step_timeout_ms"): "MOVE_STEP_TIMEOUT_MS",
    ("command", "move_default_timeout_ms"): "MOVE_DEFAULT_TIMEOUT_MS",
    ("command", "auto_stand_before_move"): "AUTO_STAND_BEFORE_MOVE",
    ("command", "move_prepare_delay_ms"): "MOVE_PREPARE_DELAY_MS",
    ("command", "move_linear_speed"): "MOVE_LINEAR_SPEED",
    ("command", "move_yaw_speed"): "MOVE_YAW_SPEED",
    ("command", "move_prime_timeout_ms"): "MOVE_PRIME_TIMEOUT_MS",
    ("command", "move_post_move_delay_ms"): "MOVE_POST_MOVE_DELAY_MS",
    ("command", "move_native_enabled"): "MOVE_NATIVE_ENABLED",
    ("command", "move_native_default_steps"): "MOVE_NATIVE_DEFAULT_STEPS",
    ("command", "move_native_min_steps"): "MOVE_NATIVE_MIN_STEPS",
    ("command", "move_native_timeout_ms"): "MOVE_NATIVE_TIMEOUT_MS",
    ("command", "move_native_linear_speed"): "MOVE_NATIVE_LINEAR_SPEED",
    ("command", "move_native_yaw_speed"): "MOVE_NATIVE_YAW_SPEED",
    ("command", "move_fast_response"): "MOVE_FAST_RESPONSE",
    ("command", "move_fast_native_first"): "MOVE_FAST_NATIVE_FIRST",
    ("command", "move_fast_followup_move"): "MOVE_FAST_FOLLOWUP_MOVE",
    ("command", "move_fast_followup_delay_ms"): "MOVE_FAST_FOLLOWUP_DELAY_MS",
    ("command", "move_fast_auto_stand"): "MOVE_FAST_AUTO_STAND",
    ("command", "move_stop_after_timeout"): "MOVE_STOP_AFTER_TIMEOUT",
    ("command", "success_audio"): "COMMAND_SUCCESS_AUDIO",
    ("command", "failed_audio"): "COMMAND_FAILED_AUDIO",
    ("command", "unavailable_audio"): "COMMAND_UNAVAILABLE_AUDIO",
    ("command", "action_audio"): "COMMAND_ACTION_AUDIO",
    ("command", "rules_enabled"): "COMMAND_RULES_ENABLED",
    ("command", "rules_fast_path"): "COMMAND_RULES_FAST_PATH",
    ("command", "feedback_async"): "COMMAND_FEEDBACK_ASYNC",
    ("command", "feedback_suppress_ms"): "COMMAND_FEEDBACK_SUPPRESS_MS",
    ("microphone", "device"): "MIC_DEVICE",
    ("microphone", "channel"): "MIC_CHANNEL",
    ("microphone", "level_log_interval"): "MIC_LEVEL_LOG_INTERVAL",
    ("microphone", "denoise"): "AUDIO_DENOISE",
    ("microphone", "gain"): "MIC_GAIN",
    ("microphone", "webrtc_gain"): "WEBRTC_AUDIO_GAIN",
    ("microphone", "webrtc_denoise"): "WEBRTC_AUDIO_DENOISE",
    ("microphone", "webrtc_target_peak"): "WEBRTC_TARGET_PEAK",
    ("microphone", "webrtc_noise_gate_rms"): "WEBRTC_NOISE_GATE_RMS",
    ("microphone", "webrtc_noise_gate_attenuation"): "WEBRTC_NOISE_GATE_ATTENUATION",
    ("microphone", "noise_calibration_seconds"): "NOISE_CALIBRATION_SECONDS",
    ("microphone", "noise_gate_multiplier"): "NOISE_GATE_MULTIPLIER",
    ("microphone", "noise_gate_min_rms"): "NOISE_GATE_MIN_RMS",
    ("microphone", "noise_gate_attenuation"): "NOISE_GATE_ATTENUATION",
    ("respeaker", "doa_enabled"): "RESPEAKER_DOA_ENABLED",
    ("respeaker", "doa_required"): "RESPEAKER_DOA_REQUIRED",
    ("respeaker", "vid"): "RESPEAKER_VID",
    ("respeaker", "pid"): "RESPEAKER_PID",
    ("respeaker", "doa_interval"): "RESPEAKER_DOA_INTERVAL",
    ("respeaker", "doa_log_interval"): "RESPEAKER_DOA_LOG_INTERVAL",
    ("respeaker", "angle_offset"): "RESPEAKER_ANGLE_OFFSET",
    ("hardware_serial", "port"): "HARDWARE_SERIAL_PORT",
    ("hardware_serial", "baudrate"): "HARDWARE_SERIAL_BAUDRATE",
    ("hardware_serial", "reconnect_interval"): "HARDWARE_SERIAL_RECONNECT_INTERVAL",
    ("hardware_serial", "audio_channel"): "HARDWARE_AUDIO_CHANNEL",
    ("hardware_serial", "audio_queue_size"): "HARDWARE_AUDIO_QUEUE_SIZE",
    ("hardware_serial", "auto_start_audio"): "HARDWARE_AUTO_START_AUDIO",
    ("hardware_serial", "software_wake_fallback"): "HARDWARE_SOFTWARE_WAKE_FALLBACK",
    ("hardware_serial", "follow_wake_beam"): "HARDWARE_FOLLOW_WAKE_BEAM",
    ("hardware_serial", "follow_wake_angle"): "HARDWARE_FOLLOW_WAKE_ANGLE",
    ("hardware_serial", "set_wake_keyword"): "HARDWARE_SET_WAKE_KEYWORD",
    ("hardware_serial", "wake_keyword"): "HARDWARE_WAKE_KEYWORD",
    ("hardware_serial", "wake_threshold"): "HARDWARE_WAKE_THRESHOLD",
    ("hardware_serial", "beam_directions"): "HARDWARE_BEAM_DIRECTIONS",
    ("hardware_serial", "audio_channel_directions"): "HARDWARE_AUDIO_CHANNEL_DIRECTIONS",
    ("hardware_serial", "audio_channel_angles"): "HARDWARE_AUDIO_CHANNEL_ANGLES",
    ("hardware_serial", "beam_audio_channels"): "HARDWARE_BEAM_AUDIO_CHANNELS",
    ("vad", "mode"): "VAD_MODE",
    ("vad", "aggressiveness"): "VAD_AGGRESSIVENESS",
    ("vad", "silence_rms"): "VAD_SILENCE_RMS",
    ("vad", "silence_multiplier"): "VAD_SILENCE_MULTIPLIER",
    ("vad", "command_silence_rms"): "COMMAND_VAD_SILENCE_RMS",
    ("vad", "command_silence_multiplier"): "COMMAND_VAD_SILENCE_MULTIPLIER",
    ("vad", "debug"): "VAD_DEBUG",
    ("vad", "debug_interval"): "VAD_DEBUG_INTERVAL",
    ("vad", "silence_timeout_ms"): "VAD_SILENCE_TIMEOUT_MS",
    ("vad", "command_silence_timeout_ms"): "COMMAND_VAD_SILENCE_TIMEOUT_MS",
    ("vad", "min_speech_ms"): "VAD_MIN_SPEECH_MS",
    ("vad", "command_listen_timeout_ms"): "COMMAND_LISTEN_TIMEOUT_MS",
    ("vad", "utterance_pad_ms"): "UTTERANCE_PAD_MS",
    ("vad", "utterance_trim_enabled"): "UTTERANCE_TRIM_ENABLED",
    ("vad", "utterance_trim_pad_ms"): "UTTERANCE_TRIM_PAD_MS",
}
CONFIG_PATH_ENV_NAMES = {
    "KWS_MODEL_DIR",
    "WAKE_AUDIO",
    "COMMAND_OUTPUT_DIR",
    "COMMAND_SUCCESS_AUDIO",
    "COMMAND_FAILED_AUDIO",
    "COMMAND_UNAVAILABLE_AUDIO",
}


def _project_path(value):
    if value is None:
        return value
    text = os.path.expanduser(str(value))
    if os.path.isabs(text):
        return text
    return os.path.join(PROJECT_ROOT, text)


def _load_config(path):
    if not path:
        return {}
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        log.info("配置文件不存在，使用内置默认值: %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".json"):
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("读取 YAML 配置需要安装 PyYAML") from exc
            data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件必须是对象/dict: {path}")
    log.info("已加载配置文件: %s", path)
    return data


def _cfg(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _env_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _apply_config_env(config):
    for keys, env_name in CONFIG_ENV_MAP.items():
        value = _cfg(config, *keys)
        if value is not None:
            if env_name in CONFIG_PATH_ENV_NAMES:
                value = _project_path(value)
            elif env_name == "COMMAND_ACTION_AUDIO" and isinstance(value, dict):
                value = {
                    key: _project_path(path) if path else path
                    for key, path in value.items()
                }
            os.environ[env_name] = _env_value(value)


def _proc_alive(proc):
    if isinstance(proc, subprocess.Popen):
        return proc.poll() is None
    return proc.is_alive()


def _proc_exitcode(proc):
    if isinstance(proc, subprocess.Popen):
        return proc.poll()
    return proc.exitcode


def _proc_wait(proc, timeout):
    if isinstance(proc, subprocess.Popen):
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    else:
        proc.join(timeout=timeout)


def _cleanup_children(reason="exit"):
    global _CLEANING_UP
    if _CLEANING_UP:
        return
    _CLEANING_UP = True

    live = [proc for proc in CHILD_PROCS if _proc_alive(proc)]
    if live:
        log.info("清理子进程 (%s): %s", reason, [proc.pid for proc in live])

    for proc in live:
        proc.terminate()
    for proc in live:
        _proc_wait(proc, timeout=3)
    for proc in live:
        if _proc_alive(proc):
            log.warning("子进程 PID=%s 未正常退出，强制结束", proc.pid)
            proc.kill()
    for proc in live:
        _proc_wait(proc, timeout=1)


def _install_cleanup_handlers():
    atexit.register(_cleanup_children, "atexit")

    def _handle_signal(signum, frame):
        log.info("收到信号 %s，准备退出", signum)
        _cleanup_children(f"signal {signum}")
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except (OSError, ValueError):
            pass


def _start_parent_watchdog(name):
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def _watch_parent():
        while parent.is_alive():
            time.sleep(1)
        logging.getLogger("run").error("%s 检测到父进程已退出，子进程自杀", name)
        os._exit(3)

    threading.Thread(target=_watch_parent, name=f"{name}-parent-watchdog", daemon=True).start()


def _start_asr_server(model_dir, host, port, use_gpu):
    """在子进程中启动 ASR HTTP 服务"""
    _start_parent_watchdog("ASR")
    from asr.engine import load_session
    from asr.server import run_serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [ASR] %(message)s")
    logging.getLogger("run").info("ASR model: Qwen3-ASR ONNX")
    engine = load_session(model_dir, use_gpu=use_gpu)
    run_serve(engine, host, port)


def _start_nlu_server(model_dir, tokenizer_dir, host, port, use_gpu=False):
    """在子进程中启动 NLU HTTP 服务"""
    _start_parent_watchdog("NLU")
    from nlu.engine import load_sessions, load_tokenizer
    from nlu.server import run_serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [NLU] %(message)s")
    tokenizer = load_tokenizer(tokenizer_dir)
    enc_sess, dec_sess = load_sessions(model_dir, use_gpu=use_gpu)
    run_serve(enc_sess, dec_sess, tokenizer, host, port)


def _find_rust_binary(configured_path=None):
    candidates = []
    if configured_path:
        candidates.append(_project_path(configured_path))

    dist_dir = os.path.join(PROJECT_ROOT, "dist")
    if os.path.isdir(dist_dir):
        for name in sorted(os.listdir(dist_dir)):
            if name.startswith("voice-infer-ubuntu2204-"):
                for binary_name in RUST_BINARY_NAMES:
                    candidates.append(os.path.join(dist_dir, name, binary_name))

    for profile in ("release", "debug"):
        for binary_name in RUST_BINARY_NAMES:
            candidates.append(os.path.join(PROJECT_ROOT, "voice-infer", "target", profile, binary_name))

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    raise RuntimeError(
        "Rust backend selected but voice-infer binary was not found. "
        "Build it with `cd voice-infer && cargo build --release`, "
        "or set inference.rust_binary in config.yaml."
    )


def _start_rust_infer(args, inference_config):
    binary = _find_rust_binary(inference_config.get("rust_binary"))
    env = os.environ.copy()

    ort_dylib = inference_config.get("rust_ort_dylib")
    if ort_dylib:
        env["ORT_DYLIB_PATH"] = _project_path(ort_dylib)

    ort_opt = inference_config.get("rust_ort_opt")
    if ort_opt:
        env["VOICE_INFER_ORT_OPT"] = str(ort_opt)

    max_new_tokens = inference_config.get("rust_asr_max_new_tokens")
    if max_new_tokens is not None:
        env["QWEN_ASR_MAX_NEW_TOKENS"] = str(max_new_tokens)

    cmd = [
        binary,
        "--asr-model-dir", args.asr_model,
        "--nlu-model-dir", args.nlu_model,
        "--nlu-tokenizer-dir", args.nlu_tokenizer,
        "--host", args.host,
        "--asr-port", str(args.asr_port),
        "--nlu-port", str(args.nlu_port),
    ]
    if args.gpu:
        cmd.append("--gpu")

    log.info("Starting Rust voice-infer: %s", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)


def _wait_for_service(name, health_url, proc, timeout=DEFAULT_SERVICE_TIMEOUT, interval=0.5):
    """等待子进程服务通过健康检查"""
    deadline = time.time() + timeout
    next_log = time.time() + 5
    while time.time() < deadline:
        if not proc.is_alive():
            raise RuntimeError(f"{name} 进程已退出，exitcode={proc.exitcode}")
        try:
            with urllib_request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    log.info("%s 健康检查通过: %s", name, health_url)
                    return
        except urllib_error.URLError:
            pass
        except Exception:
            log.exception("%s 健康检查异常", name)
        if time.time() >= next_log:
            remain = max(0, int(deadline - time.time()))
            log.info("等待 %s 服务启动中，剩余约 %ss: %s", name, remain, health_url)
            next_log = time.time() + 5
        time.sleep(interval)
    raise TimeoutError(f"{name} 健康检查超时: {health_url}")


def _ensure_port_available(name, host, port):
    """启动前确认端口未被其他进程占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"{name} 端口 {host}:{port} 已被占用，请先停止旧进程后再启动"
            ) from exc


def _tcp_port_open(ip: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_service(name, health_url, proc, timeout=DEFAULT_SERVICE_TIMEOUT, interval=0.5):
    deadline = time.time() + timeout
    next_log = time.time() + 5
    while time.time() < deadline:
        if proc is not None and not _proc_alive(proc):
            raise RuntimeError(f"{name} process exited, exitcode={_proc_exitcode(proc)}")
        try:
            with urllib_request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    log.info("%s health check passed: %s", name, health_url)
                    return
        except urllib_error.URLError:
            pass
        except Exception:
            log.exception("%s health check failed", name)
        if time.time() >= next_log:
            remain = max(0, int(deadline - time.time()))
            log.info("waiting for %s service, ~%ss left: %s", name, remain, health_url)
            next_log = time.time() + 5
        time.sleep(interval)
    raise TimeoutError(f"{name} health check timeout: {health_url}")


def _preflight_webrtc(args) -> bool:
    method = str(os.environ.get("UNITREE_WEBRTC_METHOD", "LocalSTA") or "LocalSTA").lower()
    if method in {"localap", "ap"}:
        ip = "192.168.12.1"
    elif method in {"remote", "cloud"}:
        sn = os.environ.get("UNITREE_ROBOT_SERIAL_NUMBER", "").strip()
        username = os.environ.get("UNITREE_USERNAME", "").strip()
        password = os.environ.get("UNITREE_PASSWORD", "").strip()
        ok = bool(sn and username and password)
        if ok:
            log.info("WebRTC Remote 预检通过: serial=%s username=%s", sn, username)
        else:
            log.error("WebRTC Remote 需要 serial_number/username/password")
        return ok
    else:
        ip = os.environ.get("UNITREE_ROBOT_IP", "").strip() or str(_cfg(args.config_data, "robot", "ip", default="")).strip()

    if not ip:
        log.error("WebRTC LocalSTA 需要 robot.ip，或者提供 robot.serial_number 进行局域网发现")
        return False

    ports = (9991, 8081)
    open_ports = [port for port in ports if _tcp_port_open(ip, port)]
    if open_ports:
        log.info("WebRTC Local 预检通过: %s open_ports=%s", ip, open_ports)
        return True

    log.error(
        "WebRTC Local 预检失败: %s 的 9991/8081 都无法连接。请检查 robot.ip、电脑和机器狗是否在同一网络、"
        "机器狗是否开机且未被官方 App 占用；如果连接机器狗热点，尝试 connection_method=LocalAP；"
        "如果走云端，改用 connection_method=Remote 并配置 serial_number/username/password。",
        ip,
    )
    return False


def main():
    _install_cleanup_handlers()

    parser = argparse.ArgumentParser(description="VoiceControl 统一启动器")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径，支持 YAML/JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve-asr", action="store_true", help="仅启动 ASR 服务")
    mode.add_argument("--serve-nlu", action="store_true", help="仅启动 NLU 服务")
    mode.add_argument("--pipeline-only", action="store_true", help="仅启动 Pipeline")
    audio_mode = parser.add_mutually_exclusive_group()
    audio_mode.add_argument("--onboard", action="store_true", help="本机麦克风模式")
    audio_mode.add_argument("--webrtc", action="store_true", help="WebRTC/Go2 音频模式")
    audio_mode.add_argument("--hardware-serial", action="store_true", help="硬件串口唤醒/音频模式")

    parser.add_argument("--asr-model", default=None, help="ASR 模型目录")
    parser.add_argument("--nlu-model", default=None, help="NLU 模型目录")
    parser.add_argument("--nlu-tokenizer", default=None, help="NLU 分词器目录")
    parser.add_argument("--asr-port", type=int, default=None)
    parser.add_argument("--nlu-port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 推理")
    parser.add_argument("--preflight-only", action="store_true", help="仅执行启动前连通性检查，不启动 ASR/NLU/Pipeline")
    parser.add_argument("--skip-preflight", action="store_true", help="跳过 WebRTC 连通性预检")
    denoise = parser.add_mutually_exclusive_group()
    denoise.add_argument("--denoise", action="store_true", help="开启本机麦克风轻量降噪")
    denoise.add_argument("--no-denoise", action="store_true", help="关闭本机麦克风轻量降噪")
    parser.add_argument(
        "--vad-mode",
        choices=("silence", "webrtc"),
        help="VAD 裁切模式：silence 按静音能量裁切，webrtc 使用 WebRTC VAD",
    )
    parser.add_argument("--vad-silence-rms", type=float, help="silence 模式的最低语音 RMS 阈值")
    parser.add_argument("--vad-silence-multiplier", type=float, help="silence 模式的噪声底倍数")
    parser.add_argument("--vad-silence-timeout-ms", type=int, help="句尾静音多久后裁切")
    parser.add_argument("--vad-min-speech-ms", type=int, help="最短有效语音时长")
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=None,
        help="等待 ASR/NLU 服务启动的最长秒数",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    args.config_data = config
    _apply_config_env(config)

    server_config = _cfg(config, "server", default={}) or {}
    inference_config = _cfg(config, "inference", default={}) or {}
    models_config = _cfg(config, "models", default={}) or {}
    inference_backend = str(inference_config.get("backend", "python")).lower()
    if inference_backend not in {"python", "rust", "external"}:
        raise RuntimeError("inference.backend must be one of: python, rust, external")
    audio_source = str(_cfg(config, "audio", "source", default="onboard")).lower()
    if args.onboard:
        audio_source = "onboard"
    elif args.webrtc:
        audio_source = "webrtc"
    elif args.hardware_serial:
        audio_source = "hardware_serial"
    if audio_source not in {"onboard", "webrtc", "hardware_serial"}:
        raise RuntimeError("audio.source 只能是 onboard、webrtc 或 hardware_serial")
    args.onboard = audio_source == "onboard"
    args.webrtc = audio_source == "webrtc"
    args.hardware_serial = audio_source == "hardware_serial"
    if args.hardware_serial:
        fallback = os.environ.get("HARDWARE_SOFTWARE_WAKE_FALLBACK", "1") not in {"0", "false", "False", "no"}
        os.environ["WAKE_BACKEND"] = "asr" if fallback else "hardware"
    args.asr_model = _project_path(args.asr_model or models_config.get("asr") or DEFAULT_ASR_MODEL)
    args.nlu_model = _project_path(args.nlu_model or models_config.get("nlu") or DEFAULT_NLU_MODEL)
    args.nlu_tokenizer = _project_path(
        args.nlu_tokenizer or models_config.get("nlu_tokenizer") or DEFAULT_NLU_TOKENIZER
    )
    args.asr_port = args.asr_port if args.asr_port is not None else int(server_config.get("asr_port", 8000))
    args.nlu_port = args.nlu_port if args.nlu_port is not None else int(server_config.get("nlu_port", 8001))
    args.host = args.host or server_config.get("host") or "0.0.0.0"
    args.gpu = args.gpu or bool(server_config.get("gpu", False))
    args.service_timeout = (
        args.service_timeout
        if args.service_timeout is not None
        else float(server_config.get("service_timeout", DEFAULT_SERVICE_TIMEOUT))
    )

    if args.denoise:
        os.environ["AUDIO_DENOISE"] = "1"
    elif args.no_denoise:
        os.environ["AUDIO_DENOISE"] = "0"
    if args.vad_mode:
        os.environ["VAD_MODE"] = args.vad_mode
    if args.vad_silence_rms is not None:
        os.environ["VAD_SILENCE_RMS"] = str(args.vad_silence_rms)
    if args.vad_silence_multiplier is not None:
        os.environ["VAD_SILENCE_MULTIPLIER"] = str(args.vad_silence_multiplier)
    if args.vad_silence_timeout_ms is not None:
        os.environ["VAD_SILENCE_TIMEOUT_MS"] = str(args.vad_silence_timeout_ms)
    if args.vad_min_speech_ms is not None:
        os.environ["VAD_MIN_SPEECH_MS"] = str(args.vad_min_speech_ms)

    if args.preflight_only:
        ok = True
        if args.webrtc:
            ok = _preflight_webrtc(args)
        if ok:
            log.info("预检通过")
            return
        raise SystemExit(2)

    # ── 单服务模式 ──────────────────────────────────────────────────────────
    if args.serve_asr:
        _start_asr_server(args.asr_model, args.host, args.asr_port, args.gpu)
        return

    if args.serve_nlu:
        _start_nlu_server(args.nlu_model, args.nlu_tokenizer, args.host, args.nlu_port, args.gpu)
        return

    # ── Pipeline-only 模式 ─────────────────────────────────────────────────
    if args.pipeline_only:
        if args.hardware_serial:
            from pipeline.hardware_serial import run_hardware_serial
            asyncio.run(run_hardware_serial())
        elif args.onboard:
            from pipeline.onboard import run_onboard
            asyncio.run(run_onboard())
        else:
            from pipeline.main import run_webrtc
            asyncio.run(run_webrtc())
        return

    # ── 全量启动：ASR + NLU + Pipeline ────────────────────────────────────
    log.info("启动全部服务...")

    if args.webrtc and not args.skip_preflight and not _preflight_webrtc(args):
        raise SystemExit(2)

    if inference_backend in {"rust", "external"}:
        asr_proc = None
        nlu_proc = None

        if inference_backend == "external":
            asr_url = os.environ.get("ASR_URL") or _cfg(config, "services", "asr_url")
            nlu_url = os.environ.get("NLU_URL") or _cfg(config, "services", "nlu_url")
            if not asr_url or not nlu_url:
                raise RuntimeError("external inference backend requires services.asr_url and services.nlu_url")
            os.environ["ASR_URL"] = asr_url
            os.environ["NLU_URL"] = nlu_url
            asr_health_url = asr_url.rsplit("/", 1)[0] + "/health"
            nlu_health_url = nlu_url.rsplit("/", 1)[0] + "/health"
            log.info("Using external inference services: ASR=%s NLU=%s", asr_url, nlu_url)
        else:
            _ensure_port_available("ASR", args.host, args.asr_port)
            _ensure_port_available("NLU", args.host, args.nlu_port)
            os.environ["ASR_URL"] = f"http://127.0.0.1:{args.asr_port}/asr"
            os.environ["NLU_URL"] = f"http://127.0.0.1:{args.nlu_port}/nlu"
            asr_health_url = f"http://127.0.0.1:{args.asr_port}/health"
            nlu_health_url = f"http://127.0.0.1:{args.nlu_port}/health"
            rust_proc = _start_rust_infer(args, inference_config)
            CHILD_PROCS[:] = [rust_proc]
            asr_proc = rust_proc
            nlu_proc = rust_proc
            log.info("Rust voice-infer PID=%d (ASR port %d, NLU port %d)", rust_proc.pid, args.asr_port, args.nlu_port)

        try:
            _wait_for_service("ASR", asr_health_url, asr_proc, timeout=args.service_timeout)
            _wait_for_service("NLU", nlu_health_url, nlu_proc, timeout=args.service_timeout)

            if args.hardware_serial:
                from pipeline.hardware_serial import run_hardware_serial
                asyncio.run(run_hardware_serial())
            elif args.onboard:
                from pipeline.onboard import run_onboard
                asyncio.run(run_onboard())
            else:
                from pipeline.main import run_webrtc
                asyncio.run(run_webrtc())
        finally:
            _cleanup_children("main finally")
        return

    _ensure_port_available("ASR", args.host, args.asr_port)
    _ensure_port_available("NLU", args.host, args.nlu_port)

    # Full startup owns these child services, so point the pipeline at the
    # freshly started local ports even if the shell has stale URLs exported.
    os.environ["ASR_URL"] = f"http://127.0.0.1:{args.asr_port}/asr"
    os.environ["NLU_URL"] = f"http://127.0.0.1:{args.nlu_port}/nlu"

    asr_proc = multiprocessing.Process(
        target=_start_asr_server,
        args=(args.asr_model, args.host, args.asr_port, args.gpu),
        daemon=True,
    )
    nlu_proc = multiprocessing.Process(
        target=_start_nlu_server,
        args=(args.nlu_model, args.nlu_tokenizer, args.host, args.nlu_port, args.gpu),
        daemon=True,
    )

    asr_proc.start()
    nlu_proc.start()
    CHILD_PROCS[:] = [asr_proc, nlu_proc]
    log.info("ASR 服务 PID=%d (port %d)", asr_proc.pid, args.asr_port)
    log.info("NLU 服务 PID=%d (port %d)", nlu_proc.pid, args.nlu_port)

    try:
        _wait_for_service(
            "ASR",
            f"http://127.0.0.1:{args.asr_port}/health",
            asr_proc,
            timeout=args.service_timeout,
        )
        _wait_for_service(
            "NLU",
            f"http://127.0.0.1:{args.nlu_port}/health",
            nlu_proc,
            timeout=args.service_timeout,
        )

        if args.hardware_serial:
            from pipeline.hardware_serial import run_hardware_serial
            log.info("进入硬件串口唤醒/音频模式")
            asyncio.run(run_hardware_serial())
            log.error("硬件串口模式异常结束：run_hardware_serial() 已返回")
        elif args.onboard:
            from pipeline.onboard import run_onboard
            log.info("进入本机麦克风模式")
            asyncio.run(run_onboard())
            log.error("本机麦克风模式异常结束：run_onboard() 已返回")
        else:
            from pipeline.main import run_webrtc
            log.info("进入 WebRTC 模式")
            asyncio.run(run_webrtc())
            log.error("WebRTC 模式异常结束：run_webrtc() 已返回")
    except KeyboardInterrupt:
        log.info("收到退出信号")
    except Exception:
        log.exception("主流程运行失败")
    finally:
        _cleanup_children("main finally")
        log.info("所有服务已停止")


if __name__ == "__main__":
    main()
