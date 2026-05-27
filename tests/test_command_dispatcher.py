import unittest
from unittest import mock

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
            {"command_type": "stand_down"},
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
            {"command_type": "stand_up"},
        )

    def test_service_payload_normalizes_move_payload_for_motion_api(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "normalized_text": "向右走两步",
            "command": {
                "intent": "move_right",
                "slots": {"command_type": "MoveRight"},
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
            {"command_type": "move_right", "vx": 0, "vy": -0.3, "vyaw": 0},
        )

    def test_generic_move_payload_uses_wz_and_timeout(self):
        dispatcher = CommandDispatcher()
        envelope = {
            "normalized_text": "向前走两步",
            "command": {
                "intent": "move_forward",
                "slots": {"command_type": "Move"},
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
                "vx": 0.5,
                "vy": 0,
                "wz": 0,
                "timeout_ms": 2400,
                "payload_json": {"vx": 0.5, "vy": 0, "wz": 0, "timeout_ms": 2400},
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
    async def test_move_sequence_posts_stand_up_then_five_move_repeats(self):
        dispatcher = CommandDispatcher()
        payload = {
            "command_type": "move",
            "vx": 0.5,
            "vy": 0,
            "wz": 0,
            "timeout_ms": 1200,
            "payload_json": {"vx": 0.5, "vy": 0, "wz": 0, "timeout_ms": 1200},
        }

        async def fake_post(posted_payload):
            return {"http_status": 200, "json": {"ok": True}, "request_json": dict(posted_payload)}

        with mock.patch.object(dispatcher, "_post_payload", mock.AsyncMock(side_effect=fake_post)) as post_payload:
            with mock.patch("pipeline.command_dispatcher.asyncio.sleep", mock.AsyncMock()):
                result = await dispatcher._post_move_sequence(payload)

        self.assertTrue(dispatcher._service_ok(result))
        self.assertEqual(post_payload.await_count, 6)
        posted = [call.args[0] for call in post_payload.await_args_list]
        self.assertEqual(posted[0], {"command_type": "stand_up"})
        self.assertEqual(posted[1:], [payload] * 5)
        self.assertEqual(len(result["sequence"]), 6)


if __name__ == "__main__":
    unittest.main()
