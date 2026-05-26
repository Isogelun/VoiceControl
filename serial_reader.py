#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口消息读取和验证脚本
Serial Message Reading and Validation Script

用于读取串口发送的消息，验证头部和校验码，并打印消息数据。
Used to read messages sent via serial port, validate headers and checksums, and print message data.

消息格式 / Message Format:
同步头(1) + 用户ID(1) + 消息类型(1) + 消息长度(2) + 消息ID(2) + 消息数据(N) + 校验码(1)
Sync Header(1) + User ID(1) + Message Type(1) + Message Length(2) + Message ID(2) + Message Data(N) + Checksum(1)
"""

import serial
import struct
import time
import sys
import argparse
import json
import threading
import select
from typing import Optional, Tuple


class SerialMessageReader:
    """
    串口消息读取器
    Serial Message Reader
    """

    # 消息格式常量 / Message format constants
    SYNC_HEADER = 0xA5
    USER_ID = 0x01
    WAKEUP_MSG_TYPE = 0x04
    MANUAL_WAKEUP_TYPE = 0x05
    AUDIO_DATA_TYPE = 0x06
    HANDSHAKE_MSG_TYPE = 0x01
    HANDSHAKE_ACK_TYPE = 0xFF
    HEADER_SIZE = 7
    MAX_MESSAGE_SIZE = 1024

    def __init__(self, port: str, baudrate: int = 115200, pcm_file: str = 'audio.pcm'):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.message_count = 0
        self.message_id = 0
        self.pcm_file = pcm_file
        self.running = True
        self.lock = threading.Lock()
        self.is_recording = False  # 是否正在录制音频 / Whether recording audio

    # =============================
    # 串口操作 / Serial Port Operations
    # =============================
    def open_serial(self) -> bool:
        """
        打开串口
        Open serial port
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            if self.serial_conn.is_open:
                print(f"串口已打开: {self.port}, 波特率: {self.baudrate}")
                return True
            print(f"无法打开串口: {self.port}")
            return False
        except Exception as e:
            print(f"串口打开失败: {e}")
            return False

    def close_serial(self):
        """
        关闭串口
        Close serial port
        """
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("串口已关闭")

    # =============================
    # 校验与编码 / Checksum and Encoding
    # =============================
    def calculate_checksum(self, data: bytes) -> int:
        """
        计算校验码
        Calculate checksum
        """
        checksum = sum(data) & 0xFF
        return ((~checksum) + 1) & 0xFF

    def encode_message(self, msg_type: int, payload: bytes, custom_msg_id: Optional[int] = None) -> bytes:
        """
        编码消息
        Encode message
        """
        payload_len = len(payload)
        total_size = self.HEADER_SIZE + payload_len + 1
        message = bytearray(total_size)
        offset = 0

        message[offset] = self.SYNC_HEADER
        offset += 1
        message[offset] = self.USER_ID
        offset += 1
        message[offset] = msg_type
        offset += 1
        message[offset:offset + 2] = struct.pack('<H', payload_len)
        offset += 2
        msg_id = custom_msg_id if custom_msg_id is not None else self.message_id
        message[offset:offset + 2] = struct.pack('<H', msg_id)
        offset += 2
        message[offset:offset + payload_len] = payload
        offset += payload_len
        message[offset] = self.calculate_checksum(message[:offset])

        if custom_msg_id is None:
            self.message_id = (self.message_id + 1) % 65536
        return bytes(message)

    # =============================
    # 握手处理 / Handshake Handling
    # =============================
    def send_handshake_ack(self, handshake_msg_id: int) -> bool:
        """
        发送握手确认
        Send handshake acknowledgment
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送握手确认")
            return False
        try:
            handshake_ack_data = bytes([0xA5, 0x00, 0x00, 0x00])
            message = self.encode_message(
                self.HANDSHAKE_ACK_TYPE, handshake_ack_data, handshake_msg_id)
            self.serial_conn.write(message)
            print(f"握手确认消息已发送 (ID: {handshake_msg_id})")
            return True
        except Exception as e:
            print(f"发送握手确认失败: {e}")
            return False

    # =============================
    # 消息解析与验证 / Message Parsing and Validation
    # =============================
    def parse_message_header(self, data: bytes) -> Optional[dict]:
        """
        解析消息头
        Parse message header
        """
        if len(data) < self.HEADER_SIZE:
            return None
        try:
            return {
                'sync_header': data[0],
                'user_id': data[1],
                'msg_type': data[2],
                'msg_length': struct.unpack('<H', data[3:5])[0],
                'msg_id': struct.unpack('<H', data[5:7])[0]
            }
        except struct.error:
            return None

    def validate_message(self, data: bytes) -> Tuple[bool, Optional[dict], Optional[bytes]]:
        """
        验证消息格式与校验码
        Validate message format and checksum
        """
        if len(data) < self.HEADER_SIZE + 1:
            return False, None, None
        header_info = self.parse_message_header(data)
        if not header_info:
            return False, None, None

        if header_info['sync_header'] != self.SYNC_HEADER:
            print(f"无效同步头: 0x{header_info['sync_header']:02X}")
            return False, None, None
        if header_info['user_id'] != self.USER_ID:
            print(f"无效用户ID: 0x{header_info['user_id']:02X}")
            return False, None, None

        expected_total_length = self.HEADER_SIZE + \
            header_info['msg_length'] + 1
        if len(data) < expected_total_length:
            return False, None, None

        message_data = data[self.HEADER_SIZE:self.HEADER_SIZE +
                            header_info['msg_length']]
        received_checksum = data[self.HEADER_SIZE + header_info['msg_length']]
        expected_checksum = self.calculate_checksum(
            data[:self.HEADER_SIZE + header_info['msg_length']])

        if received_checksum != expected_checksum:
            print(
                f"校验码错误: 期望0x{expected_checksum:02X} 实际0x{received_checksum:02X}")
            return False, None, None

        return True, header_info, message_data

    # =============================
    # 输出与读取循环 / Output and Read Loop
    # =============================
    def get_message_type_name(self, msg_type: int) -> str:
        """
        获取消息类型名
        Get message type name
        """
        mapping = {
            self.HANDSHAKE_MSG_TYPE: "握手消息",
            self.WAKEUP_MSG_TYPE: "设备消息",
            self.MANUAL_WAKEUP_TYPE: "手动唤醒",
            self.AUDIO_DATA_TYPE: "音频数据",
            self.HANDSHAKE_ACK_TYPE: "握手确认"
        }
        return mapping.get(msg_type, f"未知类型(0x{msg_type:02X})")

    def handle_wakeup_message(self, message_data: bytes):
        """
        处理设备唤醒消息
        Handle device wakeup message
        """
        try:
            text_data = message_data.decode('utf-8', errors='ignore')
            msg_json = json.loads(text_data)

            print("\n" + "=" * 50)
            print("检测到唤醒事件!")
            print("=" * 50)
            
            # 尝试提取角度、波束、关键词信息
            # Try to extract angle, beam, and keyword information
            angle = None
            beam = None
            keyword = None
            
            # 尝试从不同的 JSON 结构中提取信息
            # Try to extract info from different JSON structures
            if "content" in msg_json:
                content = msg_json["content"]
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        pass
                
                if isinstance(content, dict):
                    # 直接在 content 层 / Directly in content layer
                    if "angle" in content:
                        angle = content["angle"]
                    if "physical" in content:
                        beam = content["physical"]
                    if "keyword" in content:
                        keyword = content["keyword"]
                    
                    # 在 content.info 层 / In content.info layer
                    if "info" in content:
                        info = content["info"]
                        if isinstance(info, str):
                            try:
                                info = json.loads(info)
                            except:
                                pass
                        if isinstance(info, dict) and "ivw" in info:
                            ivw = info["ivw"]
                            angle = ivw.get("angle", angle)
                            beam = ivw.get("physical", beam)
                            keyword = ivw.get("keyword", keyword)
            
            # 直接在根层 / Directly at root level
            if angle is None and "angle" in msg_json:
                angle = msg_json["angle"]
            if beam is None and "physical" in msg_json:
                beam = msg_json["physical"]
            if keyword is None and "keyword" in msg_json:
                keyword = msg_json["keyword"]
            
            # 打印提取的信息 / Print extracted information
            if angle is not None:
                print(f"声源角度: {angle}°")
            if beam is not None:
                print(f"波束索引: {beam}")
            if keyword is not None:
                print(f"唤醒词: {keyword}")
            
            # 如果没有提取到任何信息，打印原始 JSON
            # If no info extracted, print raw JSON
            if angle is None and beam is None and keyword is None:
                print("  原始数据:")
                print(json.dumps(msg_json, indent=4, ensure_ascii=False))
            
            print("=" * 50 + "\n")
            
        except json.JSONDecodeError:
            print(f"唤醒消息解析失败，原始数据: {message_data}")
        except Exception as e:
            print(f"唤醒消息处理异常: {e}")

    def init_pcm_file(self):
        """
        初始化PCM文件（程序启动时清空）
        Initialize PCM file (clear on program start)
        """
        try:
            with open(self.pcm_file, 'wb') as f:
                pass  # 创建/清空文件 / Create/clear file
            print(f"PCM文件已初始化: {self.pcm_file}")
        except Exception as e:
            print(f"初始化PCM文件失败: {e}")

    def handle_audio_data(self, message_data: bytes):
        """
        处理音频数据，追加到PCM文件
        Process audio data and append to PCM file
        
        硬件发送的数据格式：每16字节为一帧
        Hardware data format: each frame is 16 bytes
        - 前12字节：6个麦克风的音频数据（每个麦克风2字节，16-bit）
        - First 12 bytes: audio data from 6 microphones (2 bytes per mic, 16-bit)
        - 后4字节：帧控制信息（需要去除）
        - Last 4 bytes: frame control info (to be removed)
        
        保存时只提取纯音频数据（前12字节），播放参数：16000Hz, 6声道, 16-bit
        Only extract pure audio data (first 12 bytes) when saving, playback params: 16000Hz, 6 channels, 16-bit
        """
        try:
            with open(self.pcm_file, 'ab') as f:
                # 按16字节帧处理，去除每帧末尾4字节控制信息
                # Process by 16-byte frames, remove 4-byte control info at the end of each frame
                frame_size = 16
                audio_size = 12  # 6声道 * 2字节 / 6 channels * 2 bytes
                
                for i in range(0, len(message_data) - frame_size + 1, frame_size):
                    frame = message_data[i:i + frame_size]
                    audio_data = frame[:audio_size]  # 只取前12字节纯音频 / Only take first 12 bytes of pure audio
                    f.write(audio_data)
        except Exception as e:
            print(f"保存音频数据失败: {e}")

    def send_get_original_audio(self):
        """
        发送开启原始音频消息
        Send enable original audio message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        try:
            reply = {
                "type": "get_original_audio",
                "content": {
                    "audio": 1
                }
            }
            reply_bytes = json.dumps(reply, ensure_ascii=False).encode('utf-8')
            message = self.encode_message(self.MANUAL_WAKEUP_TYPE, reply_bytes)
            with self.lock:
                self.serial_conn.write(message)
            self.is_recording = True
            print(f"录制音频中... (按 0 停止)")
            return True
        except Exception as e:
            print(f"发送开启原始音频消息失败: {e}")
            return False

    def send_stop_original_audio(self):
        """
        发送关闭原始音频消息
        Send disable original audio message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        try:
            reply = {
                "type": "get_original_audio",
                "content": {
                    "audio": 0
                }
            }
            reply_bytes = json.dumps(reply, ensure_ascii=False).encode('utf-8')
            message = self.encode_message(self.MANUAL_WAKEUP_TYPE, reply_bytes)
            with self.lock:
                self.serial_conn.write(message)
            self.is_recording = False
            print(f"\n录制已停止，音频已保存到: {self.pcm_file}")
            return True
        except Exception as e:
            print(f"发送关闭原始音频消息失败: {e}")
            return False

    def send_manual_wakeup(self):
        """
        发送手动唤醒消息
        Send manual wakeup message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        try:
            reply = {
                "type": "manual_wakeup",
                "content": {
                    "beam": 0
                }
            }
            reply_bytes = json.dumps(reply, ensure_ascii=False).encode('utf-8')
            message = self.encode_message(self.MANUAL_WAKEUP_TYPE, reply_bytes)
            with self.lock:
                self.serial_conn.write(message)
            print(f"已发送手动唤醒消息 (0x05) len is {len(reply_bytes)}")
            return True
        except Exception as e:
            print(f"发送手动唤醒消息失败: {e}")
            return False

    def send_wakeup_keywords(self):
        """
        发送唤醒关键词消息
        Send wakeup keyword message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        try:
            # 注意：支持任意中文唤醒词，仅支持1个自定义词（设置新词覆盖旧词）
            # Note: Supports any Chinese wakeup word, only 1 custom word supported (new word overwrites old)
            # 阈值范围：500-1500（字符串格式），越高越严格
            # Threshold range: 500-1500 (string format), higher = stricter
            # 警告：threshold 必须是字符串，否则硬件会卡死！
            # WARNING: threshold MUST be a string, otherwise hardware will freeze!
            # 设置后需重新上电才生效，掉电保留
            # Power cycle required after setting, retained after power off
            wakeup_keywords_msg = {
                "type": "wakeup_keywords",
                "content": {
                    "keyword": "你好小亚",
                    "threshold": "700"  # 必须是字符串！ / Must be string!
                }
            }
            wakeup_keywords_bytes = json.dumps(
                wakeup_keywords_msg, ensure_ascii=False).encode('utf-8')
            wakeup_keywords_message = self.encode_message(
                self.MANUAL_WAKEUP_TYPE, wakeup_keywords_bytes)
            with self.lock:
                self.serial_conn.write(wakeup_keywords_message)
            print(f"已发送唤醒关键词消息: keyword={wakeup_keywords_msg['content']['keyword']}, threshold={wakeup_keywords_msg['content']['threshold']}")
            return True
        except Exception as e:
            print(f"发送唤醒关键词消息失败: {e}")
            return False

    def send_get_version(self):
        """
        发送获取版本消息
        Send get version message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        try:
            version_msg = {
                "type": "version",
            }
            version_bytes = json.dumps(version_msg, ensure_ascii=False).encode('utf-8')
            version_message = self.encode_message(self.MANUAL_WAKEUP_TYPE, version_bytes)
            with self.lock:
                self.serial_conn.write(version_message)
            print(f"已发送获取版本消息")
            return True
        except Exception as e:
            print(f"发送获取版本消息失败: {e}")
            return False

    def send_set_beam(self, beam: int = 0):
        """
        发送设置波束消息
        Send set beam message
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开，无法发送消息")
            return False
        if beam < 0 or beam > 5:
            print(f"波束值无效: {beam}，有效范围是 0-5")
            return False
        try:
            # 波束范围：0-5，对应六个方向
            # Beam range: 0-5, corresponding to six directions
            beam_msg = {
                "type": "manual_wakeup",
                "content": {
                    "beam": beam
                }
            }
            beam_bytes = json.dumps(beam_msg, ensure_ascii=False).encode('utf-8')
            beam_message = self.encode_message(self.MANUAL_WAKEUP_TYPE, beam_bytes)
            with self.lock:
                self.serial_conn.write(beam_message)
            print(f"已发送设置波束消息: beam={beam}")
            return True
        except Exception as e:
            print(f"发送设置波束消息失败: {e}")
            return False

    def handle_keyboard_input(self):
        """
        处理键盘输入（在独立线程中运行）
        Handle keyboard input (runs in separate thread)
        """
        print("\n键盘控制说明:")
        print("  0 - 【关闭】原始音频输出")
        print("  1 - 【开启】原始音频输出")
        print("  2 - 手动唤醒（触发唤醒事件）")
        print("  3 - 设置唤醒关键词")
        print("  4 - 获取固件版本")
        print("  5 - 设置波束方向")
        print("  q - 退出程序")
        print("=" * 50)

        waiting_for_beam = False  # 是否正在等待波束输入 / Whether waiting for beam input
        input_buffer = ""  # Windows 下的输入缓冲 / Input buffer for Windows

        # 检测操作系统 / Detect operating system
        is_windows = sys.platform == 'win32'
        
        if is_windows:
            import msvcrt

        try:
            while self.running:
                try:
                    key = None
                    
                    if is_windows:
                        # Windows: 使用 msvcrt 检测键盘输入
                        # Windows: Use msvcrt to detect keyboard input
                        if msvcrt.kbhit():
                            char = msvcrt.getwch()
                            if char == '\r':  # 回车键 / Enter key
                                key = input_buffer.strip()
                                input_buffer = ""
                            elif char == '\x03':  # Ctrl+C
                                raise KeyboardInterrupt
                            else:
                                input_buffer += char
                                print(char, end='', flush=True)
                        else:
                            time.sleep(0.05)
                            continue
                    else:
                        # Linux: 使用 select
                        # Linux: Use select
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            key = sys.stdin.readline().strip()
                        else:
                            continue
                    
                    if not key:
                        continue
                    
                    print()  # Windows 下需要换行 / Need newline on Windows

                    # 如果正在等待波束输入 / If waiting for beam input
                    if waiting_for_beam:
                        if key.isdigit() and 0 <= int(key) <= 5:
                            self.send_set_beam(int(key))
                        else:
                            print(f"无效的波束值: {key}，请输入 0-5")
                        waiting_for_beam = False
                        continue

                    if key == 'q' or key == 'Q':
                        print("\n正在退出...")
                        self.running = False
                        break
                    elif key == '0':
                        self.send_stop_original_audio()
                    elif key == '1':
                        self.send_get_original_audio()
                    elif key == '2':
                        self.send_manual_wakeup()
                    elif key == '3':
                        self.send_wakeup_keywords()
                    elif key == '4':
                        self.send_get_version()
                    elif key == '5':
                        print("请输入波束值 (0-5): ", end='', flush=True)
                        waiting_for_beam = True
                    else:
                        print(f"未知按键: {key}，请按 0/1/2/3/4/5/q")
                except (OSError, ValueError) as e:
                    time.sleep(0.1)
                    continue
        except Exception as e:
            print(f"键盘输入处理异常: {e}")
            self.running = False

    def print_message_info(self, header_info: dict, message_data: bytes):
        """
        打印消息内容
        Print message content
        """
        self.message_count += 1
        print(f"\n--- 消息 #{self.message_count} ---")
        print(f"类型: {self.get_message_type_name(header_info['msg_type'])}")
        print(f"ID: {header_info['msg_id']}")
        print(f"长度: {header_info['msg_length']} 字节")

        if message_data:

            # 尝试打印文本数据 / Try to print text data
            try:
                text_data = message_data.decode('utf-8', errors='ignore')
                if text_data.isprintable() or any(c.isprintable() for c in text_data):
                    print(f"数据(文本): {text_data}")
            except Exception:
                pass

    def read_messages(self):
        """
        持续读取消息
        Continuously read messages
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("串口未打开")
            return

        # 程序启动时清空PCM文件 / Clear PCM file on program start
        self.init_pcm_file()

        # 启动键盘输入线程 / Start keyboard input thread
        keyboard_thread = threading.Thread(
            target=self.handle_keyboard_input, daemon=True)
        keyboard_thread.start()

        print(f"开始监听 {self.port} ... (按 'q' 退出)")
        buffer = b''

        try:
            while self.running:
                if self.serial_conn.in_waiting > 0:
                    buffer += self.serial_conn.read(
                        self.serial_conn.in_waiting)

                    while len(buffer) >= self.HEADER_SIZE:
                        sync_pos = buffer.find(bytes([self.SYNC_HEADER]))
                        if sync_pos == -1:
                            buffer = b''
                            break
                        if sync_pos > 0:
                            buffer = buffer[sync_pos:]

                        header_info = self.parse_message_header(buffer)
                        if not header_info:
                            buffer = buffer[1:]
                            continue

                        total_len = self.HEADER_SIZE + \
                            header_info['msg_length'] + 1
                        if len(buffer) < total_len:
                            break

                        message = buffer[:total_len]
                        buffer = buffer[total_len:]

                        valid, header_info, msg_data = self.validate_message(
                            message)
                        if valid:
                            # 音频数据不打印详情，避免乱码
                            # Skip printing audio data details to avoid garbled output
                            if header_info['msg_type'] == self.AUDIO_DATA_TYPE:
                                self.handle_audio_data(msg_data)
                            else:
                                self.print_message_info(header_info, msg_data)

                                if header_info['msg_type'] == self.HANDSHAKE_MSG_TYPE:
                                    print(
                                        f"检测到握手消息 -> 自动回复握手确认 (ID: {header_info['msg_id']})")
                                    self.send_handshake_ack(header_info['msg_id'])
                                elif header_info['msg_type'] == self.WAKEUP_MSG_TYPE:
                                    self.handle_wakeup_message(msg_data)
                        else:
                            print("无效消息，已跳过。")
                time.sleep(0.01)
        except KeyboardInterrupt:
            print(f"\n收到中断信号，正在退出...")
            self.running = False
        except Exception as e:
            print(f"读取错误: {e}")
            self.running = False
        finally:
            print(f"\n退出，共读取 {self.message_count} 条消息。")


def main():
    parser = argparse.ArgumentParser(description='串口消息读取与验证工具')
    parser.add_argument('-p', '--port', default='COM25',
                        help='串口设备路径 (默认: /dev/ttyACM0)')
    parser.add_argument('-b', '--baudrate', type=int,
                        default=115200, help='波特率 (默认: 115200)')
    parser.add_argument('-o', '--output', default='audio.pcm',
                        help='PCM音频输出文件路径 (默认: audio.pcm)')
    args = parser.parse_args()

    reader = SerialMessageReader(args.port, args.baudrate, args.output)
    try:
        if not reader.open_serial():
            sys.exit(1)
        reader.read_messages()
    finally:
        reader.close_serial()


if __name__ == '__main__':
    main()
