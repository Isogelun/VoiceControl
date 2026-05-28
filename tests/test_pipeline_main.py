import unittest
from unittest import mock

import pipeline.main as pipeline_main
from pipeline.main import VoicePipeline


class InlineWakeCommandTests(unittest.TestCase):
    def test_strip_wake_phrase_preserves_original_command_text(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.wake_texts = ["你好花花", "花花"]

        self.assertEqual(pipe._strip_wake_phrase("你好花花，向前走三步"), "向前走三步")
        self.assertEqual(pipe._strip_wake_phrase("你好，花花，请向左转"), "请向左转")

    def test_strip_wake_phrase_returns_empty_without_inline_command(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.wake_texts = ["你好花花"]

        self.assertEqual(pipe._strip_wake_phrase("你好花花"), "")


class CommandParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_enter_listening_plays_wake_audio_through_dispatcher(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.play_audio = mock.AsyncMock()
        pipe._reset_speech_capture = mock.Mock()

        with mock.patch.object(pipeline_main, "WAKE_FEEDBACK_ENABLED", True):
            with mock.patch.object(pipeline_main, "WAKE_AUDIO", "audio/wake.mp3"):
                with mock.patch.object(pipeline_main.asyncio, "create_task") as create_task:
                    pipe._enter_listening({"source": "test"})

        self.assertEqual(pipe._state, "listening")
        self.assertEqual(pipe._wake_metadata, {"source": "test"})
        create_task.assert_called_once()
        pipe.dispatcher.play_audio.assert_called_once_with("audio/wake.mp3", success=True)

    async def test_process_utterance_uses_nlu_before_rules(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr"}
        pipe.wake_texts = ["你好曼波", "曼波", "慢播", "快播", "那波"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})

        nlu_result = {"intent": "stand_up", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="站起来。")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)) as call_nlu:
                with mock.patch.object(pipeline_main, "parse_command_rule") as parse_command_rule:
                    await pipe._process_utterance(b"pcm")

        call_nlu.assert_awaited_once_with("站起来。")
        parse_command_rule.assert_not_called()
        pipe.dispatcher.dispatch.assert_awaited_once()
        dispatched = pipe.dispatcher.dispatch.await_args.args[0]
        self.assertEqual(dispatched["source"], "model")

    async def test_process_utterance_strips_repeated_wake_prefix_before_nlu(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr", "keyword": "你好，慢播。"}
        pipe.wake_texts = ["你好曼波", "曼波", "慢播", "快播", "那波"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})

        nlu_result = {"intent": "move_forward", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="odicology你好，慢播，向前走一步。")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)) as call_nlu:
                await pipe._process_utterance(b"pcm")

        call_nlu.assert_awaited_once_with("向前走一步")
        pipe.dispatcher.dispatch.assert_awaited_once()
        self.assertEqual(pipe.dispatcher.dispatch.await_args.args[2], "向前走一步")

    async def test_process_utterance_enables_feedback_suppression(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr"}
        pipe.wake_texts = ["你好曼波", "曼波", "慢播", "快播", "那波"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})
        pipe._pcm_buf = pipeline_main.np.array([], dtype=pipeline_main.np.int16)
        pipe._speech_buf = b""
        pipe._silence_count = 0
        pipe._speech_frame_count = 0
        pipe._listen_frame_count = 0
        pipe._feedback_suppress_until = 0.0

        nlu_result = {"intent": "sit_down", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="下来。")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)):
                await pipe._process_utterance(b"pcm")

        self.assertGreater(pipe._feedback_suppress_until, 0.0)
        self.assertTrue(pipe._is_feedback_suppressed())


if __name__ == "__main__":
    unittest.main()
