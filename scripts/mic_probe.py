#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson/Go2 语音与麦克风探测工具。

模式:
  --detect-assistant   探测机器狗自带语音助手（唤醒词「你好笨笨」）相关进程/服务/配置（默认）
  --list               列出 sounddevice 可见的输入设备
  --scan-all           逐个设备采样电平
  --device N           对指定设备采样

示例:
    python scripts/mic_probe.py
    python scripts/mic_probe.py --detect-assistant --grep-root /unitree /opt/unitree
    python scripts/mic_probe.py --watch-audio 15    # 持续 15 秒监视谁占用 /dev/snd
    python scripts/mic_probe.py --list
    python scripts/mic_probe.py --device 4 --duration 10
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCKSIZE = 480

# 宇树 Go2 自带助手常见唤醒词
NATIVE_WAKE_WORDS = ("你好笨笨", "笨笨", "benben", "BenBen", "BENBEN")

# 进程名/cmdline 里可能出现的关键词
PROC_KEYWORDS = (
    "unitree", "go2", "voice", "speech", "audio", "asr", "nlu", "nlp",
    "wake", "wakeup", "kws", "vad", "assistant", "benben", "笨笨",
    "sherpa", "onnx", "whisper", "paddle", "webrtc", "sport_mode",
    "robot_state", "audio_hub", "sound", "mic", "record",
)

# 默认在机身常见路径里搜唤醒词配置（优先宇树业务目录）
DEFAULT_GREP_ROOTS = (
    "/home/unitree/work",
    "/home/unitree/go2_custom_bridge",
    "/unitree",
    "/opt/unitree",
    "/etc/systemd/system",
    "/lib/systemd/system",
)

# 路径里出现则强烈怀疑为自带语音/Go2 业务（加分）
PROC_HIGH_PRIORITY = (
    "go-play-local-service", "go2_ws_agent", "go2_custom_bridge",
    "go2dds", "webrtc", "voice", "speech", "wake", "wakeup",
    "asr", "kws", "benben", "笨笨", "audio_hub",
)

# 仅因用户名/开发工具路径命中 unitree 的噪声进程（降权或剔除）
PROC_NOISE_MARKERS = (
    ".vscode-server", "code-server", "shellintegration-bash",
    "sshd:", "gsd-sound", "mic_probe.py", "mic.py",
)

# ─── 可选依赖（仅 --list / --scan-all / --device 需要）────────────────────────


def _venv_python_hint() -> str:
    root = Path(__file__).resolve().parents[1]
    vpy = root / ".venv" / "bin" / "python"
    if vpy.is_file():
        return f"sudo {vpy} mic.py"
    return "sudo .venv/bin/python mic.py  （或先 source .venv/bin/activate）"


def _import_sd_optional():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        return None


def _import_sd():
    sd = _import_sd_optional()
    if sd is None:
        raise SystemExit(
            "缺少 sounddevice。\n"
            "探测自带助手不需要该库，请直接: sudo python3 mic.py\n"
            f"若要扫描麦克风，请用虚拟环境: {_venv_python_hint()}"
        )
    return sd


def _import_np():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise SystemExit(
            f"缺少 numpy: {exc}\n"
            f"请用: {_venv_python_hint()}"
        ) from exc


# ─── 原生语音助手探测 ───────────────────────────────────────────────────────────


def _run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(cmd)}"


def _has_cmd(name: str) -> bool:
    code, _, _ = _run(["which", name], timeout=5)
    return code == 0


def _read_proc_cmdline(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return ""
    raw = path.read_bytes()
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _read_proc_comm(pid: int) -> str:
    path = Path(f"/proc/{pid}/comm")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _proc_matches_wake_or_keywords(cmdline: str, comm: str) -> Tuple[bool, List[str]]:
    text = f"{comm} {cmdline}".lower()
    hits = []
    for w in NATIVE_WAKE_WORDS:
        if w.lower() in text or w in cmdline:
            hits.append(f"wake:{w}")
    for kw in PROC_KEYWORDS:
        if kw in text:
            hits.append(f"kw:{kw}")
    return bool(hits), hits


def _proc_score(cmdline: str, hits: List[str]) -> int:
    low = cmdline.lower()
    if any(n in low for n in PROC_NOISE_MARKERS):
        if not any(p in low for p in PROC_HIGH_PRIORITY):
            return -1
    score = len(hits)
    if any(h.startswith("wake:") for h in hits):
        score += 30
    for p in PROC_HIGH_PRIORITY:
        if p in low:
            score += 20
    return score


def scan_processes() -> List[dict]:
    """扫描 /proc，找可能与自带语音助手相关的进程。"""
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            comm = _read_proc_comm(pid)
            cmdline = _read_proc_cmdline(pid)
        except OSError:
            continue
        if not cmdline and comm in ("ps", "grep", "mic_probe.py", "mic.py"):
            continue
        matched, hits = _proc_matches_wake_or_keywords(cmdline, comm)
        if not matched:
            continue
        score = _proc_score(cmdline, hits)
        if score < 0:
            continue
        rows.append({
            "pid": pid,
            "comm": comm,
            "cmdline": cmdline or comm,
            "hits": hits,
            "score": score,
        })
    rows.sort(key=lambda r: (-r["score"], r["pid"]))
    return rows


def scan_systemd_units() -> List[dict]:
    if not _has_cmd("systemctl"):
        return []
    code, out, _ = _run(
        ["systemctl", "list-units", "--type=service", "--all", "--no-pager", "--no-legend"],
        timeout=20,
    )
    if code != 0:
        return []
    hits = []
    for line in out.splitlines():
        name = line.split()[0] if line.split() else ""
        low = name.lower()
        if any(k in low for k in ("unitree", "voice", "speech", "audio", "go2", "benben", "assistant")):
            hits.append({"unit": name, "line": line.strip()})
    return hits


def grep_wake_word(roots: Iterable[str], max_depth: int = 6) -> List[dict]:
    """在常见目录里搜索「你好笨笨」等字符串（配置文件/脚本）。"""
    found = []
    text_ext = {
        ".txt", ".json", ".xml", ".yaml", ".yml", ".ini", ".conf", ".cfg",
        ".sh", ".bash", ".service", ".desktop", ".py", ".cpp", ".h", ".md",
    }
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "models", "tokenizer", ".cache", "VoiceControl",
    }
    patterns = [re.compile(re.escape(w), re.IGNORECASE) for w in NATIVE_WAKE_WORDS]

    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = dirpath[len(str(base)):].count(os.sep)
                if depth > max_depth:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
                for fname in filenames:
                    path = Path(dirpath) / fname
                    if path.suffix.lower() not in text_ext and fname not in (
                        "keywords.txt", "tokens.txt", "Makefile",
                    ):
                        # 仍尝试小文件无扩展名
                        if path.suffix:
                            continue
                    try:
                        if path.stat().st_size > 2_000_000:
                            continue
                        data = path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    for pat in patterns:
                        if pat.search(data):
                            snippet = ""
                            for line in data.splitlines():
                                if pat.search(line):
                                    snippet = line.strip()[:120]
                                    break
                            found.append({
                                "path": str(path),
                                "match": pat.pattern,
                                "snippet": snippet,
                            })
                            break
        except OSError as exc:
            print(f"[warn] 无法遍历 {base}: {exc}", file=sys.stderr)
    return found


def _snd_device_paths() -> List[str]:
    dev = Path("/dev/snd")
    if not dev.is_dir():
        return []
    return [str(p) for p in sorted(dev.iterdir()) if p.is_symlink() or p.is_file()]


def lsof_audio_devices() -> List[dict]:
    """谁打开了 ALSA 设备（需 root 才能看到其他用户进程的全部 fd）。"""
    rows = []
    snd_paths = _snd_device_paths()
    if not snd_paths:
        return rows

    if _has_cmd("lsof"):
        code, out, err = _run(["lsof"] + snd_paths[:32], timeout=15)
        if code == 0 and out:
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) < 2:
                    continue
                rows.append({
                    "pid": parts[1],
                    "user": parts[0],
                    "cmd": parts[-1] if len(parts) > 8 else "",
                    "line": line,
                })
        elif err and "not found" not in err.lower():
            print(f"[hint] lsof: {err}（可尝试 sudo python scripts/mic_probe.py）")
    elif _has_cmd("fuser"):
        _, out, _ = _run(["fuser", "-v"] + snd_paths[:16], timeout=15)
        if out:
            for line in out.splitlines():
                rows.append({"pid": "?", "user": "", "cmd": "", "line": line})
    return rows


def arecord_devices() -> str:
    if not _has_cmd("arecord"):
        return ""
    _, out, _ = _run(["arecord", "-l"], timeout=10)
    return out


def net_listeners() -> List[str]:
    lines = []
    for cmd in (
        ["ss", "-tlnp"],
        ["netstat", "-tlnp"],
    ):
        if not _has_cmd(cmd[0]):
            continue
        code, out, _ = _run(cmd, timeout=10)
        if code != 0:
            continue
        for line in out.splitlines():
            low = line.lower()
            if any(k in low for k in ("8000", "8001", "voice", "speech", "unitree", "audio")):
                lines.append(line.strip())
        break
    return lines


def _print_go2_voice_hint(top_procs: List[dict]) -> None:
    """根据已知 Go2 机载服务路径给出解读。"""
    hints = []
    for p in top_procs:
        cmd = p["cmdline"]
        if "go2_ws_agent" in cmd or "go-play-local-service" in cmd:
            hints.append(
                "  go2_ws_agent (go2-ws-agent.service): GoPlay/APP 与机载通信，"
                "自带语音指令很可能经此转发，不一定是本机麦克风 ASR。"
            )
        if "go2_custom_bridge" in cmd or "bridge.py" in cmd:
            hints.append(
                "  go2_custom_bridge (go2-custom-bridge-onboard.service): "
                "机载桥接服务，可能参与动作/状态，未必直接做唤醒识别。"
            )
    if hints:
        print()
        print("── 结合你当前机型的解读 ──")
        for h in dict.fromkeys(hints):
            print(h)


def map_pid_to_sounddevice(pid: int, processes: List[dict]) -> Optional[str]:
    """根据进程信息猜测可能对应的 PortAudio/sounddevice 设备名。"""
    proc = next((p for p in processes if p["pid"] == pid), None)
    if not proc:
        return None
    sd = _import_sd_optional()
    if sd is None:
        return None
    cmd = proc["cmdline"].lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if not name:
            continue
        # 设备名片段出现在 cmdline 里
        for token in re.split(r"[\s_/\-]+", name):
            if len(token) >= 4 and token in cmd:
                return f"[{idx}] {dev['name']}"
    return None


def detect_native_assistant(
    grep_roots: Iterable[str],
    watch_seconds: int = 0,
    grep_depth: int = 6,
) -> None:
    print("=" * 80)
    print("机器狗自带语音助手探测（唤醒词: 你好笨笨）")
    print("=" * 80)
    print()
    print("说明:")
    print("  1. 先确保自带助手已开启（APP 里打开语音/笨笨狗模式）")
    print("  2. 说「你好笨笨」时请用: sudo python3 mic.py --watch-audio 30")
    print("     （单次快照看不出变化；未 sudo 时可能看不到声卡占用）")
    print("  3. 若 lsof 结果为空，请用 sudo 重新运行本脚本")
    print()

    # ── 进程 ──
    print("── 相关进程 (按相关度排序) ──")
    procs = scan_processes()
    if not procs:
        print("  未在 cmdline/comm 中匹配到明显语音相关进程。")
        print("  建议: 开启自带助手后加 --watch-audio，或 sudo 运行以查看 ALSA 占用。")
    else:
        for i, p in enumerate(procs[:25], 1):
            wake = "★" if any(h.startswith("wake:") for h in p["hits"]) else " "
            print(f"  {wake} PID={p['pid']:>6}  [{', '.join(p['hits'])}]")
            print(f"       comm: {p['comm']}")
            print(f"       cmd:  {p['cmdline'][:200]}")
        if procs:
            top = procs[0]
            print()
            print(f"  → 最可疑进程: PID={top['pid']} ({top['comm']})")
            guess = map_pid_to_sounddevice(top["pid"], procs)
            if guess:
                print(f"  → 可能对应 sounddevice 设备: {guess}")
            _print_go2_voice_hint(procs[:5])

    # ── systemd ──
    print()
    print("── systemd 服务 ──")
    units = scan_systemd_units()
    if units:
        for u in units:
            print(f"  {u['line']}")
    else:
        print("  未找到名称含 unitree/voice/audio 等的 service（或未安装 systemctl）")

    # ── 唤醒词配置文件 ──
    print()
    print("── 配置文件中的唤醒词 ──")
    configs = grep_wake_word(grep_roots, max_depth=grep_depth)
    if configs:
        for c in configs[:30]:
            print(f"  [{c['match']}] {c['path']}")
            if c["snippet"]:
                print(f"       {c['snippet']}")
        if len(configs) > 30:
            print(f"  ... 还有 {len(configs) - 30} 处匹配")
    else:
        print("  在默认路径未搜到「你好笨笨」等字样。")
        print(f"  可扩大范围: --grep-root {' '.join(DEFAULT_GREP_ROOTS)}")

    # ── ALSA ──
    print()
    print("── ALSA 录音设备 (arecord -l) ──")
    alsa = arecord_devices()
    print(alsa if alsa else "  (arecord 不可用)")

    print()
    print("── 当前占用 /dev/snd 的进程 (lsof/fuser) ──")
    audio_holders = lsof_audio_devices()
    if audio_holders:
        seen_pids: Set[str] = set()
        for row in audio_holders:
            print(f"  {row['line']}")
            seen_pids.add(row["pid"])
        print()
        print("  → 占用声卡的 PID:", ", ".join(sorted(seen_pids, key=lambda x: int(x) if x.isdigit() else 0)))
        for pid_s in sorted(seen_pids, key=lambda x: int(x) if x.isdigit() else 0)[:5]:
            if pid_s.isdigit():
                cmd = _read_proc_cmdline(int(pid_s))
                if cmd:
                    print(f"     PID {pid_s}: {cmd[:160]}")
    else:
        print("  当前没有进程占用 /dev/snd（或权限不足看不到）。")

    # ── 网络 ──
    print()
    print("── 可疑网络监听 (语音 HTTP 等) ──")
    listeners = net_listeners()
    if listeners:
        for line in listeners:
            print(f"  {line}")
    else:
        print("  未发现 8000/8001 等常见语音服务端口监听")

    # ── sounddevice 设备（与 VoiceControl 对比用）──
    print()
    print("── sounddevice 输入设备（VoiceControl --onboard 会用其中之一）──")
    if _import_sd_optional() is None:
        print(f"  (跳过: 当前 Python 未安装 sounddevice，可用 {_venv_python_hint()} --list)")
    else:
        list_input_devices()

    if watch_seconds > 0:
        print()
        print(f"── 监视 ALSA 占用 {watch_seconds} 秒（可说「你好笨笨」观察变化）──")
        baseline = {r["pid"] for r in lsof_audio_devices()}
        for t in range(watch_seconds):
            time.sleep(1)
            current = lsof_audio_devices()
            pids = {r["pid"] for r in current}
            new = pids - baseline
            gone = baseline - pids
            if new or gone:
                print(f"  [t={t+1}s] 新增 PID: {new or '-'} | 释放 PID: {gone or '-'}")
                for row in current:
                    if row["pid"] in new:
                        print(f"       + {row['line']}")
                baseline = pids
        print("  监视结束。")

    print()
    print("── 建议 ──")
    print("  • 若自带助手进程已占用麦克风，VoiceControl --onboard 可能无声或报 GetFrames 错误")
    print("  • 对比: 关闭 APP 语音功能后再跑本脚本，看 lsof/进程列表差异")
    print("  • 确认设备: python scripts/mic_probe.py --scan-all --duration 3")
    print("  • 仅 VoiceControl 时: 结束占用 8000/8001 的旧 run.py 进程")


# ─── 麦克风硬件探测（原有功能）────────────────────────────────────────────────


def list_input_devices():
    sd = _import_sd()
    devices = sd.query_devices()
    print("可用输入设备:")
    found = False
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        found = True
        default = " (default)" if idx == sd.default.device[0] else ""
        print(
            f"  [{idx}] {dev['name']}{default} | max_input_channels={max_in} "
            f"| default_samplerate={dev['default_samplerate']}"
        )
    if not found:
        print("  未找到可用输入设备")


def get_input_devices():
    sd = _import_sd()
    devices = sd.query_devices()
    return [
        (idx, dev) for idx, dev in enumerate(devices)
        if int(dev.get("max_input_channels", 0)) > 0
    ]


def scan_all_devices(sample_rate, blocksize, duration, per_device_channels=None):
    devices = get_input_devices()
    if not devices:
        print("未找到可用输入设备")
        return

    print(f"开始逐个扫描 {len(devices)} 个输入设备，每个设备采样 {duration} 秒。")
    for idx, dev in devices:
        max_in = int(dev.get("max_input_channels", 0))
        channels = max_in if per_device_channels is None else min(per_device_channels, max_in)
        print("\n" + "=" * 80)
        print(
            f"扫描设备 [{idx}] {dev['name']} | "
            f"max_input_channels={max_in} | channels={channels}"
        )
        try:
            open_probe_stream(
                device=idx,
                channels=channels,
                sample_rate=sample_rate,
                blocksize=blocksize,
                duration=duration,
                save_wav=None,
            )
        except Exception as exc:
            print(f"[error] 设备 [{idx}] 打开失败: {exc}")


def open_probe_stream(device, channels, sample_rate, blocksize, duration, save_wav=None):
    sd = _import_sd()
    np = _import_np()
    audio_queue = queue.Queue()
    frames = []
    stop_at = time.time() + duration if duration > 0 else None

    device_info = sd.query_devices(device, kind="input")
    if channels is None:
        channels = int(device_info["max_input_channels"])
    channels = max(1, min(channels, int(device_info["max_input_channels"])))

    print(
        f"打开输入设备: {device_info['name']} (device={device}, "
        f"channels={channels}, default_samplerate={device_info['default_samplerate']})"
    )
    print("开始采样，按 Ctrl+C 结束。")

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[warn] stream status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            device=device,
            channels=channels,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype="int16",
            callback=callback,
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    print("[warn] 1 秒内未收到音频数据")
                    if stop_at and time.time() >= stop_at:
                        break
                    continue

                frames.append(chunk)
                peak = np.max(np.abs(chunk), axis=0)
                mean_abs = np.mean(np.abs(chunk), axis=0)
                stats = " | ".join(
                    f"ch{idx}: mean_abs={mean_abs[idx]:6.1f} peak={int(peak[idx]):5d}"
                    for idx in range(chunk.shape[1])
                )
                print(stats)

                if stop_at and time.time() >= stop_at:
                    break
    except KeyboardInterrupt:
        print("\n已停止采样")

    if save_wav and frames:
        pcm = np.concatenate(frames, axis=0)
        with wave.open(save_wav, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.astype(np.int16).tobytes())
        print(f"已保存录音到: {save_wav}")


def main():
    parser = argparse.ArgumentParser(
        description="Go2 自带语音助手 + 麦克风探测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--detect-assistant",
        action="store_true",
        help="探测自带语音助手进程/配置/ALSA 占用（默认）",
    )
    mode.add_argument("--list", action="store_true", help="仅列出 sounddevice 输入设备")
    mode.add_argument("--scan-all", action="store_true", help="逐个扫描所有输入设备电平")
    mode.add_argument("--device", help="对指定输入设备采样")

    parser.add_argument(
        "--watch-audio",
        type=int,
        metavar="SEC",
        default=0,
        help="detect 模式下监视 ALSA 占用变化的秒数",
    )
    parser.add_argument(
        "--grep-root",
        action="append",
        default=None,
        help="额外搜索唤醒词配置的目录（可多次指定）",
    )
    parser.add_argument("--grep-depth", type=int, default=6, help="目录搜索最大深度")
    parser.add_argument("--channels", type=int, help="采样通道数")
    parser.add_argument("--duration", type=int, default=10, help="采样时长（秒），0=持续")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--blocksize", type=int, default=DEFAULT_BLOCKSIZE)
    parser.add_argument("--save-wav", help="保存采样 WAV")
    args = parser.parse_args()

    # 默认：探测自带助手
    if not any((args.detect_assistant, args.list, args.scan_all, args.device)):
        args.detect_assistant = True

    if args.detect_assistant:
        roots = list(DEFAULT_GREP_ROOTS)
        if args.grep_root:
            roots.extend(args.grep_root)
        detect_native_assistant(
            grep_roots=roots,
            watch_seconds=args.watch_audio,
            grep_depth=args.grep_depth,
        )
        return

    if args.list:
        list_input_devices()
        return

    if args.scan_all:
        scan_all_devices(
            sample_rate=args.sample_rate,
            blocksize=args.blocksize,
            duration=args.duration,
            per_device_channels=args.channels,
        )
        return

    if args.device is None:
        parser.error("需要 --device，或使用默认的 --detect-assistant / --list / --scan-all")

    device = args.device
    try:
        device = int(device)
    except ValueError:
        pass

    open_probe_stream(
        device=device,
        channels=args.channels,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
        duration=args.duration,
        save_wav=args.save_wav,
    )


if __name__ == "__main__":
    main()
