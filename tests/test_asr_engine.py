import json
import os
import tempfile
import unittest

from asr import engine


class AsrEngineSelectionTests(unittest.TestCase):
    def test_model_suffix_prefers_int4(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "encoder.onnx"), "wb").close()
            open(os.path.join(tmp, "encoder.int4.onnx"), "wb").close()

            self.assertEqual(engine._model_suffix(tmp), ".int4.onnx")

    def test_validate_model_dir_reports_missing_qwen_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"model_type": "qwen3_asr"}, f)

            with self.assertRaises(FileNotFoundError) as ctx:
                engine._validate_model_dir(tmp, ".int4.onnx")

        self.assertIn("tokenizer.json", str(ctx.exception))
        self.assertIn("encoder.int4.onnx", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
