import json
import unittest

from nlu.engine import parse_nlu_output


class ParseNluOutputTests(unittest.TestCase):
    def test_cmd_json_preserves_command_and_infers_forward_move(self):
        raw = json.dumps(
            {
                "type": "cmd",
                "payload": {
                    "command_type": "Move",
                    "payload_json": {"vx": 0.5, "vy": 0, "vyaw": 0},
                },
            }
        )

        result = parse_nlu_output(raw)

        self.assertEqual(result["intent"], "move_forward")
        self.assertEqual(result["slots"]["direction"], "forward")
        self.assertEqual(result["slots"]["command_type"], "Move")
        self.assertEqual(result["command"]["payload"]["command_type"], "Move")

    def test_cmd_json_infers_lateral_and_turn_moves(self):
        right = parse_nlu_output(
            json.dumps(
                {
                    "type": "cmd",
                    "payload": {
                        "command_type": "Move",
                        "payload_json": {"vx": 0, "vy": -0.3, "vyaw": 0},
                    },
                }
            )
        )
        turn_left = parse_nlu_output(
            json.dumps(
                {
                    "type": "cmd",
                    "payload": {
                        "command_type": "Move",
                        "payload_json": {"vx": 0, "vy": 0, "vyaw": 0.5},
                    },
                }
            )
        )

        self.assertEqual(right["intent"], "move_right")
        self.assertEqual(right["slots"]["direction"], "right")
        self.assertEqual(turn_left["intent"], "turn_left")
        self.assertEqual(turn_left["slots"]["direction"], "left")

    def test_chat_json_is_not_actionable(self):
        raw = json.dumps({"type": "chat", "payload": {"message": "hello"}})

        result = parse_nlu_output(raw)

        self.assertEqual(result["intent"], "unknown")
        self.assertEqual(result["source"], "chat")
        self.assertEqual(result["message"], "hello")


if __name__ == "__main__":
    unittest.main()
