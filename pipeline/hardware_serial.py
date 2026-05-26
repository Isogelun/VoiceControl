"""
Hardware serial wake/audio input.

The hardware sends framed serial messages:
sync(1) + user(1) + type(1) + length(2) + id(2) + payload + checksum(1).

Message type 0x04 is a wake event. Message type 0x06 contains audio frames.
Each audio frame is 16 bytes: six int16 microphone samples followed by 4 bytes
of frame-control data. We pick one channel and feed 16 kHz mono PCM into the
existing pipeline.
"""

import asyncio
import json
import logging
import os
import struct
import threading
import time
from typing import Optional

import numpy as np
try:
    import serial
except ImportError:
    serial = None

from .audio_preprocessor import AudioPreprocessor
from .cleaner import start_cleaner
from .main import VAD_SAMPLE_RATE, VoicePipeline

log = logging.getLogger(__name__)

SYNC_HEADER = 0xA5
USER_ID = 0x01
HANDSHAKE_MSG_TYPE = 0x01
WAKEUP_MSG_TYPE = 0x04
MANUAL_WAKEUP_TYPE = 0x05
AUDIO_DATA_TYPE = 0x06
HANDSHAKE_ACK_TYPE = 0xFF
HEADER_SIZE = 7

SERIAL_PORT = os.environ.get("HARDWARE_SERIAL_PORT", "auto")
SERIAL_BAUDRATE = int(os.environ.get("HARDWARE_SERIAL_BAUDRATE", "115200"))
SERIAL_RECONNECT_INTERVAL = float(os.environ.get("HARDWARE_SERIAL_RECONNECT_INTERVAL", "3"))
SERIAL_AUDIO_CHANNEL = int(os.environ.get("HARDWARE_AUDIO_CHANNEL", "0"))
SERIAL_QUEUE_SIZE = int(os.environ.get("HARDWARE_AUDIO_QUEUE_SIZE", "200"))
SERIAL_AUTO_START_AUDIO = os.environ.get("HARDWARE_AUTO_START_AUDIO", "1") not in {
    "0",
    "false",
    "False",
    "no",
}
SERIAL_SOFTWARE_WAKE_FALLBACK = os.environ.get("HARDWARE_SOFTWARE_WAKE_FALLBACK", "1") not in {
    "0",
    "false",
    "False",
    "no",
}
SERIAL_FOLLOW_WAKE_BEAM = os.environ.get("HARDWARE_FOLLOW_WAKE_BEAM", "1") not in {
    "0",
    "false",
    "False",
    "no",
}
SERIAL_FOLLOW_WAKE_ANGLE = os.environ.get("HARDWARE_FOLLOW_WAKE_ANGLE", "1") not in {
    "0",
    "false",
    "False",
    "no",
}
SERIAL_SET_WAKE_KEYWORD = os.environ.get("HARDWARE_SET_WAKE_KEYWORD", "0") not in {
    "0",
    "false",
    "False",
    "no",
}
SERIAL_WAKE_KEYWORD = os.environ.get("HARDWARE_WAKE_KEYWORD") or os.environ.get("WAKE_KEYWORD", "你好花花")
SERIAL_WAKE_THRESHOLD = os.environ.get("HARDWARE_WAKE_THRESHOLD", "700")
SERIAL_BEAM_DIRECTIONS = os.environ.get("HARDWARE_BEAM_DIRECTIONS", "front,front_left,left,back,right,front_right")
SERIAL_AUDIO_CHANNEL_DIRECTIONS = os.environ.get(
    "HARDWARE_AUDIO_CHANNEL_DIRECTIONS",
    SERIAL_BEAM_DIRECTIONS,
)
SERIAL_AUDIO_CHANNEL_ANGLES = os.environ.get("HARDWARE_AUDIO_CHANNEL_ANGLES", "60,120,180,240,300,0")
SERIAL_BEAM_AUDIO_CHANNELS = os.environ.get("HARDWARE_BEAM_AUDIO_CHANNELS", "0,1,2,3,4,5")
BEAM_DIRECTION_NAMES = [item.strip() for item in SERIAL_BEAM_DIRECTIONS.split(",")]
AUDIO_CHANNEL_DIRECTION_NAMES = [item.strip() for item in SERIAL_AUDIO_CHANNEL_DIRECTIONS.split(",")]
AUDIO_CHANNEL_ANGLES = []
for item in SERIAL_AUDIO_CHANNEL_ANGLES.split(","):
    try:
        AUDIO_CHANNEL_ANGLES.append(float(item.strip()) % 360)
    except ValueError:
        pass
BEAM_AUDIO_CHANNELS = []
for item in SERIAL_BEAM_AUDIO_CHANNELS.split(","):
    try:
        BEAM_AUDIO_CHANNELS.append(int(item.strip()))
    except ValueError:
        pass


def _available_ports():
    if serial is None:
        return []
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return list(list_ports.comports())


def _format_available_ports() -> str:
    ports = _available_ports()
    if not ports:
        return "无"
    return ", ".join(
        f"{item.device} ({item.description})" if item.description else item.device
        for item in ports
    )


def _resolve_serial_port(configured_port: str) -> str:
    port = (configured_port or "auto").strip()
    if port.lower() != "auto":
        return port

    ports = _available_ports()
    if not ports:
        raise RuntimeError("未发现可用串口，请检查硬件连接/驱动")
    if len(ports) == 1:
        selected = ports[0].device
        log.info("自动选择硬件串口: %s (%s)", selected, ports[0].description)
        return selected

    port_list = _format_available_ports()
    raise RuntimeError(f"发现多个串口，请在 config.yaml 的 hardware_serial.port 中指定一个: {port_list}")


class SerialProtocol:
    def __init__(self):
        self.message_id = 0

    def checksum(self, data: bytes) -> int:
        checksum = sum(data) & 0xFF
        return ((~checksum) + 1) & 0xFF

    def encode(self, msg_type: int, payload: bytes, msg_id: Optional[int] = None) -> bytes:
        payload_len = len(payload)
        current_id = self.message_id if msg_id is None else msg_id
        header = bytearray()
        header.append(SYNC_HEADER)
        header.append(USER_ID)
        header.append(msg_type)
        header.extend(struct.pack("<H", payload_len))
        header.extend(struct.pack("<H", current_id))
        header.extend(payload)
        header.append(self.checksum(header))
        if msg_id is None:
            self.message_id = (self.message_id + 1) % 65536
        return bytes(header)

    def parse_header(self, data: bytes) -> Optional[dict]:
        if len(data) < HEADER_SIZE:
            return None
        return {
            "sync_header": data[0],
            "user_id": data[1],
            "msg_type": data[2],
            "msg_length": struct.unpack("<H", data[3:5])[0],
            "msg_id": struct.unpack("<H", data[5:7])[0],
        }

    def validate(self, data: bytes):
        header = self.parse_header(data)
        if not header:
            return False, None, None
        if header["sync_header"] != SYNC_HEADER or header["user_id"] != USER_ID:
            return False, None, None

        expected_len = HEADER_SIZE + header["msg_length"] + 1
        if len(data) < expected_len:
            return False, None, None

        payload = data[HEADER_SIZE:HEADER_SIZE + header["msg_length"]]
        received = data[HEADER_SIZE + header["msg_length"]]
        expected = self.checksum(data[:HEADER_SIZE + header["msg_length"]])
        return received == expected, header, payload


class HardwareSerialSource:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        audio_queue: asyncio.Queue,
        pipeline: VoicePipeline,
    ):
        self.loop = loop
        self.audio_queue = audio_queue
        self.pipeline = pipeline
        self.protocol = SerialProtocol()
        self.serial_conn = None
        self.running = threading.Event()
        self.running.set()
        self.thread = None
        self.current_audio_channel = SERIAL_AUDIO_CHANNEL
        self._mixed_remainder = np.array([], dtype=np.int16)

    def start(self):
        if serial is None:
            raise RuntimeError("硬件串口模式需要安装 pyserial：pip install pyserial")
        self.thread = threading.Thread(target=self._read_loop, name="hardware-serial", daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        try:
            self._send_original_audio(False)
        except Exception:
            pass
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        if self.thread:
            self.thread.join(timeout=2)

    def _read_loop(self):
        while self.running.is_set():
            try:
                self._open_and_read_until_disconnect()
            except Exception as exc:
                if self.serial_conn and self.serial_conn.is_open:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
                log.warning(
                    "硬件串口暂不可用，%ss 后重试: %s | 可用串口: %s",
                    SERIAL_RECONNECT_INTERVAL,
                    exc,
                    _format_available_ports(),
                )
                time.sleep(SERIAL_RECONNECT_INTERVAL)

    def _open_and_read_until_disconnect(self):
        buffer = b""
        port = _resolve_serial_port(SERIAL_PORT)
        self.serial_conn = serial.Serial(
            port=port,
            baudrate=SERIAL_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2,
        )
        log.info(
            "硬件串口已打开: port=%s baudrate=%s default_audio_channel=%s(%s, angle=%s)",
            port,
            SERIAL_BAUDRATE,
            SERIAL_AUDIO_CHANNEL,
            _direction_name(AUDIO_CHANNEL_DIRECTION_NAMES, SERIAL_AUDIO_CHANNEL),
            _channel_angle(SERIAL_AUDIO_CHANNEL),
        )
        if SERIAL_SET_WAKE_KEYWORD:
            self._send_wake_keyword(SERIAL_WAKE_KEYWORD, SERIAL_WAKE_THRESHOLD)
        if SERIAL_AUTO_START_AUDIO:
            self._send_original_audio(True)

        while self.running.is_set():
            waiting = self.serial_conn.in_waiting
            chunk = self.serial_conn.read(waiting or 1)
            if not chunk:
                continue
            buffer += chunk

            while len(buffer) >= HEADER_SIZE:
                sync_pos = buffer.find(bytes([SYNC_HEADER]))
                if sync_pos == -1:
                    buffer = b""
                    break
                if sync_pos > 0:
                    buffer = buffer[sync_pos:]

                header = self.protocol.parse_header(buffer)
                if not header:
                    buffer = buffer[1:]
                    continue

                total_len = HEADER_SIZE + header["msg_length"] + 1
                if len(buffer) < total_len:
                    break

                message = buffer[:total_len]
                buffer = buffer[total_len:]
                valid, header, payload = self.protocol.validate(message)
                if not valid:
                    log.warning("跳过无效硬件串口消息")
                    continue
                self._handle_message(header, payload)

    def _handle_message(self, header: dict, payload: bytes):
        msg_type = header["msg_type"]
        if msg_type == HANDSHAKE_MSG_TYPE:
            self._send_handshake_ack(header["msg_id"])
        elif msg_type == WAKEUP_MSG_TYPE:
            metadata = self._decode_wake_payload(payload)
            self._update_audio_channel_from_wake(metadata)
            metadata["audio_channel"] = self.current_audio_channel
            metadata["audio_channel_direction"] = _direction_name(
                AUDIO_CHANNEL_DIRECTION_NAMES,
                self.current_audio_channel,
            )
            metadata["audio_channel_angle"] = _channel_angle(self.current_audio_channel)
            metadata["audio_channel_followed_angle"] = SERIAL_FOLLOW_WAKE_ANGLE
            metadata["audio_channel_followed_beam"] = SERIAL_FOLLOW_WAKE_BEAM
            log.info(
                "硬件唤醒方向: beam=%s(%s) angle=%s keyword=%s | ASR取音频通道=%s(%s, angle=%s)",
                metadata.get("beam"),
                metadata.get("beam_direction"),
                metadata.get("angle"),
                metadata.get("keyword"),
                self.current_audio_channel,
                metadata.get("audio_channel_direction"),
                metadata.get("audio_channel_angle"),
            )
            self.loop.call_soon_threadsafe(
                asyncio.create_task,
                self.pipeline.trigger_wake(metadata),
            )
        elif msg_type == AUDIO_DATA_TYPE:
            pcm = self._extract_runtime_pcm(payload)
            if pcm.size:
                self.loop.call_soon_threadsafe(self._enqueue_pcm, pcm)
        else:
            text = payload.decode("utf-8", errors="ignore").strip()
            if text:
                log.debug("硬件串口消息 type=0x%02X payload=%s", msg_type, text)

    def _enqueue_pcm(self, pcm: np.ndarray):
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.audio_queue.put_nowait(pcm)

    def _extract_runtime_pcm(self, payload: bytes) -> np.ndarray:
        if SERIAL_SOFTWARE_WAKE_FALLBACK and self.pipeline._state == "waiting":
            return self._extract_mixed_pcm(payload)
        return _extract_audio_channel(payload, self.current_audio_channel)

    def _extract_mixed_pcm(self, payload: bytes) -> np.ndarray:
        frame_size = 16
        frame_count = len(payload) // frame_size
        if frame_count <= 0:
            return np.array([], dtype=np.int16)

        mixed = np.empty(frame_count, dtype=np.int16)
        for idx in range(frame_count):
            frame = payload[idx * frame_size:(idx + 1) * frame_size]
            channels = np.frombuffer(frame[:12], dtype="<i2")
            mixed[idx] = int(np.clip(np.mean(channels), -32768, 32767))
        return mixed

    def _update_audio_channel_from_wake(self, metadata: dict):
        if not SERIAL_FOLLOW_WAKE_ANGLE and not SERIAL_FOLLOW_WAKE_BEAM:
            self.current_audio_channel = SERIAL_AUDIO_CHANNEL
            return

        angle = _to_float(metadata.get("angle"))
        if SERIAL_FOLLOW_WAKE_ANGLE and angle is not None and AUDIO_CHANNEL_ANGLES:
            self.current_audio_channel = _nearest_channel_for_angle(angle)
            return

        beam = _to_int(metadata.get("beam"))
        if SERIAL_FOLLOW_WAKE_BEAM and beam is not None:
            channel = _channel_for_beam(beam)
            if channel is not None:
                self.current_audio_channel = channel
                return

        self.current_audio_channel = SERIAL_AUDIO_CHANNEL

    def _send(self, msg_type: int, payload: bytes, msg_id: Optional[int] = None):
        if not self.serial_conn or not self.serial_conn.is_open:
            return
        self.serial_conn.write(self.protocol.encode(msg_type, payload, msg_id=msg_id))

    def _send_handshake_ack(self, msg_id: int):
        payload = bytes([0xA5, 0x00, 0x00, 0x00])
        self._send(HANDSHAKE_ACK_TYPE, payload, msg_id=msg_id)
        log.info("已回复硬件握手: id=%s", msg_id)

    def _send_original_audio(self, enabled: bool):
        payload = json.dumps(
            {"type": "get_original_audio", "content": {"audio": 1 if enabled else 0}},
            ensure_ascii=False,
        ).encode("utf-8")
        self._send(MANUAL_WAKEUP_TYPE, payload)
        log.info("硬件原始音频输出: %s", "开启" if enabled else "关闭")

    def _send_wake_keyword(self, keyword: str, threshold: str):
        payload = json.dumps(
            {
                "type": "wakeup_keywords",
                "content": {
                    "keyword": keyword,
                    "threshold": str(threshold),
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._send(MANUAL_WAKEUP_TYPE, payload)
        log.info("已发送硬件唤醒词配置: keyword=%s threshold=%s", keyword, threshold)

    def _decode_wake_payload(self, payload: bytes) -> dict:
        text = payload.decode("utf-8", errors="ignore")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

        content = data.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        metadata = {"raw": data}
        if isinstance(content, dict):
            metadata.update(_extract_wake_fields(content))
            info = content.get("info")
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except json.JSONDecodeError:
                    pass
            if isinstance(info, dict) and isinstance(info.get("ivw"), dict):
                metadata.update(_extract_wake_fields(info["ivw"]))
        metadata.update(_extract_wake_fields(data))
        _attach_direction_metadata(metadata)
        return metadata


def _extract_wake_fields(data: dict) -> dict:
    result = {}
    for src_key, dst_key in (("angle", "angle"), ("physical", "beam"), ("keyword", "keyword")):
        if src_key in data and data[src_key] is not None:
            result[dst_key] = data[src_key]
    return result


def _attach_direction_metadata(metadata: dict):
    beam = _to_int(metadata.get("beam"))
    angle = _to_float(metadata.get("angle"))
    if beam is not None:
        metadata["beam"] = beam
        metadata["beam_direction"] = _direction_name(BEAM_DIRECTION_NAMES, beam)
    elif angle is not None:
        inferred_beam = _beam_from_angle(angle)
        metadata["beam"] = inferred_beam
        metadata["beam_direction"] = _direction_name(BEAM_DIRECTION_NAMES, inferred_beam)
    if angle is not None:
        metadata["angle"] = angle
        metadata["angle_direction"] = _angle_direction(angle)


def _direction_name(names, index: int) -> str:
    if 0 <= index < len(names) and names[index]:
        return names[index]
    return f"index_{index}"


def _channel_angle(channel: int):
    if 0 <= channel < len(AUDIO_CHANNEL_ANGLES):
        value = AUDIO_CHANNEL_ANGLES[channel]
        return int(value) if value.is_integer() else value
    return None


def _nearest_channel_for_angle(angle: float) -> int:
    best_channel = SERIAL_AUDIO_CHANNEL
    best_distance = 361.0
    for channel, channel_angle in enumerate(AUDIO_CHANNEL_ANGLES):
        distance = _angle_distance(angle, channel_angle)
        if distance < best_distance:
            best_distance = distance
            best_channel = channel
    return best_channel


def _channel_for_beam(beam: int):
    if 0 <= beam < len(BEAM_AUDIO_CHANNELS):
        channel = BEAM_AUDIO_CHANNELS[beam]
        if 0 <= channel <= 5:
            return channel
    if 0 <= beam <= 5:
        return beam
    return None


def _angle_distance(a: float, b: float) -> float:
    diff = abs((a - b) % 360)
    return min(diff, 360 - diff)


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number % 360


def _beam_from_angle(angle: float) -> int:
    return int(((angle % 360) + 30) // 60) % 6


def _angle_direction(angle: float) -> str:
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


def _extract_audio_channel(payload: bytes, channel: int) -> np.ndarray:
    if channel < 0 or channel > 5:
        raise ValueError("HARDWARE_AUDIO_CHANNEL 必须在 0-5 之间")

    frame_size = 16
    frame_count = len(payload) // frame_size
    if frame_count <= 0:
        return np.array([], dtype=np.int16)

    samples = np.empty(frame_count, dtype=np.int16)
    offset = channel * 2
    for idx in range(frame_count):
        start = idx * frame_size + offset
        samples[idx] = struct.unpack_from("<h", payload, start)[0]
    return samples


async def run_hardware_serial():
    pipeline = VoicePipeline()
    preprocessor = AudioPreprocessor.from_env(sample_rate=VAD_SAMPLE_RATE)
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue(maxsize=SERIAL_QUEUE_SIZE)
    source = HardwareSerialSource(loop, audio_queue, pipeline)
    last_log = 0.0

    try:
        source.start()
        asyncio.create_task(start_cleaner())
        log.info("硬件串口模式已启动，等待硬件唤醒...")
        log.info(
            "硬件音频: default_channel=%s(%s, angle=%s) software_wake_fallback=%s follow_angle=%s follow_beam=%s denoise=%s auto_start_audio=%s",
            SERIAL_AUDIO_CHANNEL,
            _direction_name(AUDIO_CHANNEL_DIRECTION_NAMES, SERIAL_AUDIO_CHANNEL),
            _channel_angle(SERIAL_AUDIO_CHANNEL),
            SERIAL_SOFTWARE_WAKE_FALLBACK,
            SERIAL_FOLLOW_WAKE_ANGLE,
            SERIAL_FOLLOW_WAKE_BEAM,
            preprocessor.enabled,
            SERIAL_AUTO_START_AUDIO,
        )
        while source.running.is_set():
            try:
                pcm = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            processed = preprocessor.process(pcm)
            await pipeline.push_pcm(processed)

            now = loop.time()
            if now - last_log >= 3:
                last_log = now
                log.info(
                    "硬件音频电平: raw_rms=%.1f processed_rms=%.1f threshold=%.1f queue=%s",
                    preprocessor.last_raw_rms,
                    preprocessor.last_processed_rms,
                    preprocessor.last_threshold,
                    audio_queue.qsize(),
                )
    except KeyboardInterrupt:
        log.info("收到退出信号，停止硬件串口监听")
    finally:
        source.stop()
        pipeline.close()
        log.info("硬件串口监听结束")
