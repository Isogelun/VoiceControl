"""
ReSpeaker USB control helpers.

Audio from the ReSpeaker is still captured through the normal system audio
device. This module only talks to the vendor USB control endpoint to read DOA.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
import time
from typing import Optional

try:
    import usb.core
    import usb.util
except ImportError:
    usb = None

try:
    import libusb_package
except ImportError:
    libusb_package = None

log = logging.getLogger(__name__)

CONTROL_SUCCESS = 0
SERVICER_COMMAND_RETRY = 64
DEFAULT_VID = 0x2886
DOA_RESID = 20
DOA_CMDID = 18


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", "no", ""}


def parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip(), 0)


class ReSpeakerUSB:
    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev
        self.vid = getattr(dev, "idVendor", None)
        self.pid = getattr(dev, "idProduct", None)

    def read_doa(self) -> tuple[int, bool]:
        response = self._ctrl_read(DOA_RESID, DOA_CMDID, 2, "uint16")
        angle, speech_detected = struct.unpack("<HH", _response_bytes(response)[1:])
        return int(angle % 360), bool(speech_detected)

    def _ctrl_read(self, resid: int, cmdid: int, count: int, data_type: str):
        length = _payload_length(count, data_type) + 1
        wvalue = 0x80 | cmdid
        attempts = 0
        while True:
            attempts += 1
            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                wvalue,
                resid,
                length,
                self.TIMEOUT,
            )
            status = int(response[0])
            if status == CONTROL_SUCCESS:
                return response
            if status == SERVICER_COMMAND_RETRY and attempts < 100:
                time.sleep(0.01)
                continue
            raise RuntimeError(f"ReSpeaker control read failed: status={status}, attempts={attempts}")

    def close(self):
        usb.util.dispose_resources(self.dev)


def find_respeaker(vid: int = DEFAULT_VID, pid: Optional[int] = None) -> Optional[ReSpeakerUSB]:
    if usb is None:
        raise RuntimeError("ReSpeaker DOA requires pyusb. Install it with: pip install pyusb")
    if sys.platform.startswith("win") and libusb_package is None:
        raise RuntimeError("Windows ReSpeaker DOA requires libusb-package. Install it with: pip install libusb-package")

    usb_find = libusb_package.find if sys.platform.startswith("win") else usb.core.find
    if pid is not None:
        dev = usb_find(idVendor=vid, idProduct=pid)
        return ReSpeakerUSB(dev) if dev else None

    devices = list(usb_find(find_all=True, idVendor=vid) or [])
    if not devices:
        return None
    devices.sort(key=lambda device: getattr(device, "idProduct", 0))
    return ReSpeakerUSB(devices[0])


class ReSpeakerDoAMonitor:
    def __init__(
        self,
        vid: int = DEFAULT_VID,
        pid: Optional[int] = None,
        interval: float = 0.1,
        angle_offset: float = 0.0,
    ):
        self.vid = vid
        self.pid = pid
        self.interval = max(0.02, float(interval))
        self.angle_offset = float(angle_offset)
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dev: Optional[ReSpeakerUSB] = None
        self._last_log = 0.0

    @classmethod
    def from_env(cls) -> "ReSpeakerDoAMonitor":
        return cls(
            vid=parse_int(os.environ.get("RESPEAKER_VID"), DEFAULT_VID),
            pid=parse_int(os.environ.get("RESPEAKER_PID"), None),
            interval=float(os.environ.get("RESPEAKER_DOA_INTERVAL", "0.1")),
            angle_offset=float(os.environ.get("RESPEAKER_ANGLE_OFFSET", "0")),
        )

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="respeaker-doa", daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                log.debug("Failed to close ReSpeaker USB device", exc_info=True)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def _run(self):
        required = env_bool("RESPEAKER_DOA_REQUIRED", "0")
        try:
            self._dev = find_respeaker(self.vid, self.pid)
            if not self._dev:
                message = f"ReSpeaker USB device not found: vid=0x{self.vid:04x}, pid={self.pid}"
                if required:
                    raise RuntimeError(message)
                log.warning("%s; DOA metadata disabled", message)
                return
            log.info(
                "ReSpeaker DOA connected: VID=0x%04x PID=0x%04x interval=%.2fs angle_offset=%.1f",
                self._dev.vid,
                self._dev.pid,
                self.interval,
                self.angle_offset,
            )
            while self._running.is_set():
                angle, speech_detected = self._dev.read_doa()
                adjusted_angle = (angle + self.angle_offset) % 360
                now = time.time()
                snapshot = {
                    "doa_source": "respeaker",
                    "angle": _pretty_number(adjusted_angle),
                    "raw_angle": angle,
                    "speech_detected": speech_detected,
                    "angle_direction": angle_direction(adjusted_angle),
                    "updated_at": now,
                }
                with self._lock:
                    self._snapshot = snapshot
                if now - self._last_log >= float(os.environ.get("RESPEAKER_DOA_LOG_INTERVAL", "3")):
                    self._last_log = now
                    log.info(
                        "ReSpeaker DOA: angle=%s raw=%s speech=%s direction=%s",
                        snapshot["angle"],
                        snapshot["raw_angle"],
                        speech_detected,
                        snapshot["angle_direction"],
                    )
                time.sleep(self.interval)
        except Exception:
            log.exception("ReSpeaker DOA monitor stopped")
            if required:
                raise


def _payload_length(count: int, data_type: str) -> int:
    if data_type in {"uint8", "char"}:
        return count
    if data_type == "uint16":
        return count * 2
    return count * 4


def _response_bytes(response) -> bytes:
    if hasattr(response, "tobytes"):
        return response.tobytes()
    return bytes(response)


def angle_direction(angle: float) -> str:
    labels = [
        "front",
        "front_right",
        "right",
        "back_right",
        "back",
        "back_left",
        "left",
        "front_left",
    ]
    return labels[int(((angle % 360) + 22.5) // 45) % len(labels)]


def _pretty_number(value: float):
    return int(value) if float(value).is_integer() else round(float(value), 2)
