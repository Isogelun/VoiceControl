import unittest
from unittest import mock

import pipeline.command_dispatcher as command_dispatcher
from pipeline.command_dispatcher import CommandDispatcher


class CommandDispatcherValidationTests(unittest.TestCase):
    def test_accepts_rule_command(self):
        dispatcher = CommandDispatcher()

        self.assertTrue(
            dispatcher._is_actionable(
                {"intent": "move_forward", "slots": {"direction": "forward"}}
            )
        )

    def test_rejects_unknown_or_unsupported_intent(self):
        dispatcher = CommandDispatcher()

        self.assertFalse(dispatcher._is_actionable({"intent": "unknown", "slots": {}}))
        self.assertFalse(dispatcher._is_actionable({"intent": "dance", "slots": {}}))

    def test_rejects_malformed_model_command(self):
        dispatcher = CommandDispatcher()

        self.assertFalse(
            dispatcher._is_actionable(
                {
                    "intent": "move_forward",
                    "slots": {"direction": "forward"},
                    "command": {"type": "cmd", "payload": {}},
                }
            )
        )

    def test_success_audio_uses_action_specific_mapping(self):
        dispatcher = CommandDispatcher()

        with mock.patch.object(
            command_dispatcher,
            "COMMAND_ACTION_AUDIO",
            '{"move_forward": "audio/forward.mp3", "default": "audio/default.mp3"}',
        ):
            command_dispatcher.ACTION_AUDIO_MAP.clear()
            selected = dispatcher._select_success_audio({"intent": "move_forward", "slots": {}})
            default = dispatcher._select_success_audio({"intent": "shake_body", "slots": {}})

        self.assertEqual(selected, str(command_dispatcher._PROJECT_ROOT / "audio/forward.mp3"))
        self.assertEqual(default, str(command_dispatcher._PROJECT_ROOT / "audio/default.mp3"))
        command_dispatcher.ACTION_AUDIO_MAP.clear()

    def test_service_payload_uses_motion_api_shape_for_stand_down(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "sit_down",
                "slots": {"command_type": "StandDown"},
                "command": {
                    "type": "cmd",
                    "payload": {"command_type": "StandDown", "payload_json": {}},
                },
            }
        }

        self.assertEqual(
            dispatcher._make_service_payload(envelope),
            {"command_type": "stand_down", "params": {}},
        )

    def test_service_payload_uses_motion_api_shape_for_stand_up(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "stand_up",
                "slots": {"command_type": "StandUp"},
                "command": {
                    "type": "cmd",
                    "payload": {"command_type": "StandUp", "payload_json": {}},
                },
            }
        }

        self.assertEqual(
            dispatcher._make_service_payload(envelope),
            {"command_type": "stand_up", "params": {}},
        )

    def test_service_payload_normalizes_direction_move_to_move_with_native_followup(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "move_right",
                "slots": {"command_type": "MoveRight", "steps": 2},
                "command": {
                    "type": "cmd",
                    "payload": {
                        "command_type": "MoveRight",
                        "payload_json": {"vx": 0, "vy": -0.3, "vyaw": 0},
                    },
                },
            }
        }

        self.assertEqual(
            dispatcher._make_service_payload(envelope),
            {
                "command_type": "move",
                "params": {
                    "vx": 0,
                    "vy": -0.3,
                    "wz": 0,
                    "timeout_ms": 2000,
                },
                command_dispatcher.NATIVE_MOVE_PAYLOAD_KEY: {
                    "command_type": "move_right",
                    "params": {
                        "step": 2,
                        "timeout_ms": 1000,
                        "vy": -1.0,
                    },
                },
            },
        )

    def test_rule_move_payload_gets_default_velocity(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "move_forward",
                "slots": {"direction": "forward", "steps": 1},
                "raw": "forward",
                "source": "rule",
            }
        }

        self.assertEqual(
            dispatcher._make_service_payload(envelope),
            {
                "command_type": "move",
                "params": {
                    "vx": 0.2,
                    "vy": 0.0,
                    "wz": 0.0,
                    "timeout_ms": 1200,
                },
                command_dispatcher.NATIVE_MOVE_PAYLOAD_KEY: {
                    "command_type": "move_forward",
                    "params": {
                        "step": 1,
                        "timeout_ms": 1000,
                        "vx": 1.0,
                    },
                },
            },
        )

    def test_directional_move_uses_configured_timing_and_steps(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "move_forward",
                "slots": {"direction": "forward", "steps": 1},
                "raw": "forward",
                "source": "rule",
            }
        }

        with mock.patch.object(command_dispatcher, "MOVE_PRIME_TIMEOUT_MS", 1500):
            with mock.patch.object(command_dispatcher, "MOVE_NATIVE_MIN_STEPS", 3):
                payload = dispatcher._make_service_payload(envelope)

        self.assertEqual(
            payload,
            {
                "command_type": "move",
                "params": {
                    "vx": 0.2,
                    "vy": 0.0,
                    "wz": 0.0,
                    "timeout_ms": 1500,
                },
                command_dispatcher.NATIVE_MOVE_PAYLOAD_KEY: {
                    "command_type": "move_forward",
                    "params": {
                        "step": 3,
                        "timeout_ms": 1000,
                        "vx": 1.0,
                    },
                },
            },
        )

    def test_generic_move_payload_uses_wz_and_timeout(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "command": {
                "intent": "move_forward",
                "slots": {"command_type": "Move", "steps": 2},
                "command": {
                    "type": "cmd",
                    "payload": {
                        "command_type": "Move",
                        "payload_json": {"vx": 0.5, "vy": 0, "vyaw": 0},
                    },
                },
            },
        }

        self.assertEqual(
            dispatcher._make_service_payload(envelope),
            {
                "command_type": "move",
                "params": {
                    "vx": 0.5,
                    "vy": 0,
                    "wz": 0,
                    "timeout_ms": 2000,
                },
            },
        )
        self.assertFalse(
            dispatcher._is_actionable(
                {
                    "intent": "move_forward",
                    "slots": {"direction": "forward"},
                    "command": {"type": "chat", "payload": {"message": "hello"}},
                }
            )
        )


class CommandDispatcherServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_move_sequence_posts_stand_up_move_then_stop(self):
        dispatcher = CommandDispatcher()
        payload = {
            "command_type": "move",
            "params": {"vx": 0.5, "vy": 0, "wz": 0, "timeout_ms": 1200},
        }

        async def fake_post(posted_payload):
            return {"http_status": 200, "json": {"ok": True}, "request_json": dict(posted_payload)}

        with mock.patch.object(dispatcher, "_post_payload", mock.AsyncMock(side_effect=fake_post)) as post_payload:
            with mock.patch("pipeline.command_dispatcher.asyncio.sleep", mock.AsyncMock()):
                result = await dispatcher._post_move_sequence(payload)

        self.assertTrue(dispatcher._service_ok(result))
        self.assertEqual(post_payload.await_count, 3)
        posted = [call.args[0] for call in post_payload.await_args_list]
        self.assertEqual(posted[0], {"command_type": "stand_up", "params": {}})
        self.assertEqual(posted[1], payload)
        self.assertEqual(posted[2], {"command_type": "stop", "params": {}})
        self.assertEqual(len(result["sequence"]), 3)

    async def test_directional_move_sequence_posts_native_followup_without_stop(self):
        dispatcher = CommandDispatcher()
        native_payload = {
            "command_type": "move_forward",
            "params": {"step": 3, "vx": 1.0, "timeout_ms": 1000},
        }
        payload = {
            "command_type": "move",
            "params": {"vx": 0.25, "vy": 0, "wz": 0, "timeout_ms": 1500},
            command_dispatcher.NATIVE_MOVE_PAYLOAD_KEY: native_payload,
        }

        async def fake_post(posted_payload):
            return {"http_status": 200, "json": {"ok": True}, "request_json": dict(posted_payload)}

        with mock.patch.object(command_dispatcher, "MOVE_PREPARE_DELAY_MS", 2000):
            with mock.patch.object(command_dispatcher, "MOVE_POST_MOVE_DELAY_MS", 2000):
                with mock.patch.object(dispatcher, "_post_payload", mock.AsyncMock(side_effect=fake_post)) as post_payload:
                    with mock.patch("pipeline.command_dispatcher.asyncio.sleep", mock.AsyncMock()) as sleep:
                        result = await dispatcher._post_move_sequence(payload)

        self.assertTrue(dispatcher._service_ok(result))
        posted = [call.args[0] for call in post_payload.await_args_list]
        self.assertEqual(
            posted,
            [
                {"command_type": "stand_up", "params": {}},
                {"command_type": "move", "params": {"vx": 0.25, "vy": 0, "wz": 0, "timeout_ms": 1500}},
                native_payload,
            ],
        )
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [2.0, 2.0, 1.0])
        self.assertEqual(len(result["sequence"]), 3)

    async def test_fast_move_sequence_queues_native_first_without_waiting(self):
        dispatcher = CommandDispatcher()
        native_payload = {
            "command_type": "move_forward",
            "params": {"step": 3, "vx": 1.0, "timeout_ms": 1000},
        }
        payload = {
            "command_type": "move",
            "params": {"vx": 0.25, "vy": 0, "wz": 0, "timeout_ms": 1500},
            command_dispatcher.NATIVE_MOVE_PAYLOAD_KEY: native_payload,
        }

        async def fake_post(posted_payload):
            return {"http_status": 200, "json": {"ok": True}, "request_json": dict(posted_payload)}

        command_dispatcher.BACKGROUND_POST_TASKS.clear()
        with mock.patch.object(command_dispatcher, "MOVE_FAST_RESPONSE", True):
            with mock.patch.object(command_dispatcher, "MOVE_FAST_NATIVE_FIRST", True):
                with mock.patch.object(command_dispatcher, "MOVE_FAST_FOLLOWUP_MOVE", True):
                    with mock.patch.object(command_dispatcher, "MOVE_FAST_FOLLOWUP_DELAY_MS", 0):
                        with mock.patch.object(dispatcher, "_post_payload", mock.AsyncMock(side_effect=fake_post)) as post_payload:
                            result = await dispatcher._post_move_sequence(payload)
                            for _ in range(5):
                                if post_payload.await_count >= 2:
                                    break
                                await command_dispatcher.asyncio.sleep(0)

        self.assertEqual(result["http_status"], 202)
        self.assertTrue(result["queued"])
        self.assertEqual(result["request_json"], native_payload)
        posted = [call.args[0] for call in post_payload.await_args_list]
        self.assertEqual(posted, [native_payload, {"command_type": "move", "params": {"vx": 0.25, "vy": 0, "wz": 0, "timeout_ms": 1500}}])

    async def test_fast_non_move_dispatch_queues_without_waiting(self):
        dispatcher = CommandDispatcher()

        async def fake_post(posted_payload):
            return {"http_status": 200, "json": {"ok": True}, "request_json": dict(posted_payload)}

        command_dispatcher.BACKGROUND_POST_TASKS.clear()
        with mock.patch.object(command_dispatcher, "COMMAND_SERVICE_URL", "http://motion.local"):
            with mock.patch.object(command_dispatcher, "COMMAND_FAST_RESPONSE", True):
                with mock.patch.object(dispatcher, "_post_payload", mock.AsyncMock(side_effect=fake_post)) as post_payload:
                    with mock.patch.object(dispatcher, "_maybe_play_feedback", mock.AsyncMock()):
                        result = await dispatcher.dispatch(
                            {"intent": "stand_up", "slots": {}, "source": "rule"},
                            "stand up",
                            "stand up",
                        )
                    for _ in range(5):
                        if post_payload.await_count >= 1:
                            break
                        await command_dispatcher.asyncio.sleep(0)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["service_result"]["queued"])
        self.assertEqual(
            result["service_result"]["request_json"],
            {"command_type": "stand_up", "params": {}},
        )
        post_payload.assert_awaited_once_with({"command_type": "stand_up", "params": {}})


if __name__ == "__main__":
    unittest.main()
