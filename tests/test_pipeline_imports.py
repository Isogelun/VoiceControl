import importlib
import os
import unittest


class PipelineImportTests(unittest.TestCase):
    def test_pipeline_import_does_not_parse_onboard_mic_channel(self):
        old_value = os.environ.get("MIC_CHANNEL")
        os.environ["MIC_CHANNEL"] = "0W"
        try:
            pipeline = importlib.import_module("pipeline")
            self.assertTrue(hasattr(pipeline, "VoicePipeline"))
        finally:
            if old_value is None:
                os.environ.pop("MIC_CHANNEL", None)
            else:
                os.environ["MIC_CHANNEL"] = old_value


if __name__ == "__main__":
    unittest.main()
