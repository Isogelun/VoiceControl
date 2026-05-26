"""
pipeline/speaker.py

机器狗音频播放模块。

两种模式：
  1. 播放预置音频 — 按文件名从 AudioHub 列表中查找并播放
  2. 播放 TTS 音频 — 上传 WAV/MP3 后立即播放（后期接 TTS 时使用）

用法：
    speaker = Speaker(conn)
    await speaker.play_preset("greeting")
    await speaker.play_file("/tmp/tts_out.wav")
"""

import logging

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, conn):
        # 延迟导入：仅 WebRTC 模式需要 unitree_webrtc_connect SDK，
        # onboard（本机麦克风）模式即使该 SDK 未安装也能正常启动 pipeline。
        from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
        self._hub = WebRTCAudioHub(conn)

    async def play_preset(self, name: str):
        """按文件名播放 AudioHub 中已有的音频"""
        audio_list = await self._hub.get_audio_list()
        files = audio_list.get("data", {}).get("audio_list", [])
        match = next((f for f in files if f.get("file_name") == name), None)
        if not match:
            log.warning("预置音频 [%s] 不存在，可用: %s", name, [f.get("file_name") for f in files])
            return
        await self._hub.play_by_uuid(match["unique_id"])
        log.info("播放: %s", name)

    async def play_file(self, path: str):
        """上传音频文件并播放（支持 WAV / MP3）"""
        response = await self._hub.upload_audio_file(path)
        uid = response.get("data", {}).get("unique_id")
        if not uid:
            log.error("上传失败: %s", response)
            return
        await self._hub.play_by_uuid(uid)
        log.info("播放上传文件: %s", path)
