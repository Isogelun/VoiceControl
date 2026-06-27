"""
Robot speaker playback helpers for WebRTC mode.

The Go2 AudioHub API returns slightly different field names across firmware
versions, so this wrapper normalizes the list/upload responses before playing.
"""

import json
import logging
import os

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, conn):
        # Import lazily so onboard/local microphone mode can run without the
        # Unitree WebRTC dependencies installed.
        from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

        self._hub = WebRTCAudioHub(conn)

    async def play_preset(self, name: str):
        """Play an existing AudioHub item by name."""
        files = await self._get_audio_items()
        match = next((f for f in files if self._item_name(f) == name), None)
        if not match:
            log.warning("Preset audio [%s] not found; available=%s", name, [self._item_name(f) for f in files])
            return

        uid = self._item_uuid(match)
        if not uid:
            log.warning("Preset audio [%s] has no uuid: %s", name, match)
            return
        await self._hub.play_by_uuid(uid)
        log.info("Played preset audio: %s", name)

    async def play_file(self, path: str):
        """Upload a WAV/MP3 file to the robot and play it through Go2."""
        filename = os.path.splitext(os.path.basename(path))[0]
        matches = await self._find_audio_matches(filename)
        uid = self._item_uuid(matches[0]) if matches else None
        if uid:
            await self._hub.play_by_uuid(uid)
            await self._cleanup_duplicate_audio(filename, keep_uid=uid, matches=matches)
            log.info("Played existing robot audio: %s", filename)
            return

        response = await self._hub.upload_audio_file(path)
        matches = await self._find_audio_matches(filename)
        uid = self._extract_upload_uuid(response) or (self._item_uuid(matches[0]) if matches else None)
        if not uid:
            log.error("Audio upload did not return/find a uuid: %s", response)
            return

        await self._hub.play_by_uuid(uid)
        await self._cleanup_duplicate_audio(filename, keep_uid=uid, matches=matches)
        log.info("Uploaded and played audio file: %s", path)

    async def _find_audio_matches(self, name: str):
        return [item for item in await self._get_audio_items() if self._name_matches(item, name)]

    async def _cleanup_duplicate_audio(self, name: str, keep_uid: str, matches=None):
        """Keep one robot-side copy of a managed local audio file."""
        matches = matches if matches is not None else await self._find_audio_matches(name)
        seen = {keep_uid}
        deleted = 0
        for item in matches:
            uid = self._item_uuid(item)
            if not uid or uid in seen:
                continue
            seen.add(uid)
            try:
                await self._hub.delete_record(uid)
                deleted += 1
            except Exception as exc:
                log.warning("Failed to delete duplicate robot audio %s (%s): %s", name, uid, exc)
        if deleted:
            log.info("Deleted %s duplicate robot audio record(s) for %s", deleted, name)

    async def _get_audio_items(self):
        response = await self._hub.get_audio_list()
        data = response.get("data", {}) if isinstance(response, dict) else {}
        if isinstance(data, dict) and isinstance(data.get("audio_list"), list):
            return data["audio_list"]

        nested = data.get("data") if isinstance(data, dict) else None
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                log.warning("AudioHub audio list JSON parse failed: %s", nested[:200])
                return []

        if isinstance(nested, dict) and isinstance(nested.get("audio_list"), list):
            return nested["audio_list"]
        return []

    def _name_matches(self, item: dict, name: str) -> bool:
        item_name = self._item_name(item)
        return item_name == name or os.path.splitext(item_name)[0] == name

    def _item_name(self, item: dict) -> str:
        return str(
            item.get("file_name")
            or item.get("FILE_NAME")
            or item.get("custom_name")
            or item.get("CUSTOM_NAME")
            or ""
        )

    def _item_uuid(self, item: dict):
        return item.get("unique_id") or item.get("UNIQUE_ID") or item.get("uuid") or item.get("UUID")

    def _extract_upload_uuid(self, response: dict):
        if not isinstance(response, dict):
            return None
        data = response.get("data")
        if isinstance(data, dict):
            return data.get("unique_id") or data.get("UNIQUE_ID")
        return None
