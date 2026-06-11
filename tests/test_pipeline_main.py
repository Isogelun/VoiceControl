import unittest
from unittest import mock

import pipeline.main as pipeline_main
from pipeline.main import VoicePipeline


class InlineWakeCommandTests(unittest.TestCase):
    def test_strip_wake_phrase_preserves_inline_command(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.wake_texts = ["hello buddy", "buddy"]

        self.assertEqual(pipe._strip_wake_phrase("hello buddy, move forward"), "move forward")
        self.assertEqual(pipe._strip_wake_phrase("hello, buddy, turn left"), "turn left")

    def test_strip_wake_phrase_returns_empty_without_inline_command(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.wake_texts = ["hello buddy"]

        self.assertEqual(pipe._strip_wake_phrase("hello buddy"), "")

    def test_extract_inline_command_returns_empty_for_split_wake_aliases(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.wake_texts = ["hello buddy", "hello", "buddy"]

        self.assertEqual(pipe._extract_inline_command("hello, buddy"), "")
        self.assertEqual(pipe._extract_inline_command("hello buddy"), "")


class CommandParsingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._fast_path_patch = mock.patch.object(pipeline_main, "COMMAND_RULES_FAST_PATH", False)
        self._fast_path_patch.start()

    async def asyncTearDown(self):
        self._fast_path_patch.stop()

    async def test_enter_listening_plays_wake_audio_through_dispatcher(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.play_audio = mock.AsyncMock()
        pipe._reset_speech_capture = mock.Mock()

        with mock.patch.object(pipeline_main, "WAKE_FEEDBACK_ENABLED", True):
            with mock.patch.object(pipeline_main, "WAKE_AUDIO", "audio/wake.mp3"):
                with mock.patch.object(pipeline_main.asyncio, "create_task") as create_task:
                    create_task.side_effect = lambda coro: coro.close()
                    pipe._enter_listening({"source": "test"})

        self.assertEqual(pipe._state, "listening")
        self.assertEqual(pipe._wake_metadata, {"source": "test"})
        create_task.assert_called_once()
        pipe.dispatcher.play_audio.assert_called_once_with("audio/wake.mp3", success=True)

    async def test_process_utterance_uses_nlu_before_rules(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr"}
        pipe.wake_texts = ["hello buddy", "buddy"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})

        nlu_result = {"intent": "stand_up", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="stand up")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)) as call_nlu:
                with mock.patch.object(pipeline_main, "parse_command_rule") as parse_command_rule:
                    await pipe._process_utterance(b"pcm")

        call_nlu.assert_awaited_once_with("stand up")
        parse_command_rule.assert_not_called()
        pipe.dispatcher.dispatch.assert_awaited_once()
        dispatched = pipe.dispatcher.dispatch.await_args.args[0]
        self.assertEqual(dispatched["source"], "model")

    async def test_process_wake_utterance_enters_listening_for_wake_only_phrase(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._state = "waiting"
        pipe.wake_texts = ["hello buddy", "hello", "buddy"]
        pipe._wake_metadata = {}
        pipe._enter_listening = mock.Mock()
        pipe._process_command_text = mock.AsyncMock()

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="hello, buddy")):
            await pipe._process_wake_utterance(b"pcm")

        pipe._enter_listening.assert_called_once()
        pipe._process_command_text.assert_not_awaited()

    async def test_process_wake_utterance_dispatches_inline_command_when_present(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._state = "waiting"
        pipe.wake_texts = ["hello buddy", "buddy"]
        pipe._wake_metadata = {}
        pipe._reset_speech_capture = mock.Mock()
        pipe._process_command_text = mock.AsyncMock()

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="hello buddy, stand up")):
            await pipe._process_wake_utterance(b"pcm")

        pipe._process_command_text.assert_awaited_once_with("stand up", "hello buddy, stand up", "stand up")

    async def test_parse_command_fast_path_skips_nlu_for_known_commands(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        rule_result = {"intent": "move_forward", "slots": {"direction": "forward"}, "source": "rule"}

        with mock.patch.object(pipeline_main, "COMMAND_RULES_FAST_PATH", True):
            with mock.patch.object(pipeline_main, "parse_command_rule", return_value=rule_result) as parse_command_rule:
                with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock()) as call_nlu:
                    result = await pipe._parse_command_with_nlu("forward")

        parse_command_rule.assert_called_once_with("forward")
        call_nlu.assert_not_called()
        self.assertEqual(result, rule_result)

    async def test_process_utterance_strips_repeated_wake_prefix_before_nlu(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr", "keyword": "hello, buddy"}
        pipe.wake_texts = ["hello buddy", "hello", "buddy"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})

        nlu_result = {"intent": "move_forward", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="noise hello, buddy, move forward")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)) as call_nlu:
                await pipe._process_utterance(b"pcm")

        call_nlu.assert_awaited_once_with("move forward")
        pipe.dispatcher.dispatch.assert_awaited_once()
        self.assertEqual(pipe.dispatcher.dispatch.await_args.args[2], "move forward")

    async def test_process_utterance_enables_feedback_suppression(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._wake_metadata = {"source": "asr"}
        pipe.wake_texts = ["hello buddy", "buddy"]
        pipe.dispatcher = mock.Mock()
        pipe.dispatcher.dispatch = mock.AsyncMock(return_value={"status": "accepted"})
        pipe._pcm_buf = pipeline_main.np.array([], dtype=pipeline_main.np.int16)
        pipe._speech_buf = b""
        pipe._silence_count = 0
        pipe._speech_frame_count = 0
        pipe._listen_frame_count = 0
        pipe._feedback_suppress_until = 0.0

        nlu_result = {"intent": "sit_down", "slots": {}, "raw": "model", "source": "model"}

        with mock.patch.object(pipeline_main, "call_asr", mock.AsyncMock(return_value="sit down")):
            with mock.patch.object(pipeline_main, "call_nlu", mock.AsyncMock(return_value=nlu_result)):
                await pipe._process_utterance(b"pcm")

        self.assertGreater(pipe._feedback_suppress_until, 0.0)
        self.assertTrue(pipe._is_feedback_suppressed())

    async def test_trim_utterance_removes_leading_and_trailing_silence(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._noise_rms = 10.0
        silence = pipeline_main.np.zeros(pipeline_main.VAD_FRAME_SAMPLES * 6, dtype=pipeline_main.np.int16)
        speech = pipeline_main.np.full(pipeline_main.VAD_FRAME_SAMPLES * 2, 1000, dtype=pipeline_main.np.int16)
        pcm = pipeline_main.np.concatenate([silence, speech, silence])

        trimmed = pipeline_main.np.frombuffer(pipe._trim_utterance(pcm.tobytes()), dtype=pipeline_main.np.int16)

        self.assertLess(trimmed.size, pcm.size)
        self.assertGreaterEqual(trimmed.size, speech.size)

    async def test_current_speech_threshold_uses_command_threshold_while_listening(self):
        pipe = VoicePipeline.__new__(VoicePipeline)
        pipe._noise_rms = 100.0

        pipe._state = "waiting"
        self.assertEqual(
            pipe._current_speech_threshold(),
            max(pipeline_main.VAD_SILENCE_RMS, 100.0 * pipeline_main.VAD_SILENCE_MULTIPLIER),
        )

        pipe._state = "listening"
        self.assertEqual(
            pipe._current_speech_threshold(),
            max(
                pipeline_main.COMMAND_VAD_SILENCE_RMS,
                100.0 * pipeline_main.COMMAND_VAD_SILENCE_MULTIPLIER,
            ),
        )


if __name__ == "__main__":
    unittest.main()
