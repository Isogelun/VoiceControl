import unittest

from pipeline.text_normalizer import parse_command_rule


class CommandRuleTests(unittest.TestCase):
    def test_lateral_move_rules(self):
        left = parse_command_rule("向左走两步")
        right = parse_command_rule("右移")

        self.assertEqual(left["intent"], "move_left")
        self.assertEqual(left["slots"]["direction"], "left")
        self.assertEqual(left["slots"]["steps"], 2)
        self.assertEqual(right["intent"], "move_right")
        self.assertEqual(right["slots"]["direction"], "right")

    def test_bare_walk_rules_default_to_forward(self):
        inline = parse_command_rule("漫波，起来走两步")
        bare = parse_command_rule("漫播，走一步")

        self.assertEqual(inline["intent"], "move_forward")
        self.assertEqual(inline["slots"]["direction"], "forward")
        self.assertEqual(inline["slots"]["steps"], 2)
        self.assertEqual(bare["intent"], "move_forward")
        self.assertEqual(bare["slots"]["steps"], 1)

    def test_extra_action_rules(self):
        self.assertEqual(parse_command_rule("打招呼")["intent"], "greet")
        self.assertEqual(parse_command_rule("摇身体")["intent"], "shake_body")
        self.assertEqual(parse_command_rule("伸懒腰")["intent"], "stretch")


if __name__ == "__main__":
    unittest.main()
