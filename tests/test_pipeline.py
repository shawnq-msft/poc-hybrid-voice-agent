import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.pipeline import build_pipeline_blueprint


class PipelineBlueprintTests(unittest.TestCase):
    def test_foundry_asr_uses_batch_transport_without_silent_fallback(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        blueprint = build_pipeline_blueprint(settings)
        asr_stage = next(stage for stage in blueprint.stages if stage.name == "asr")
        vad_stage = next(stage for stage in blueprint.stages if stage.name == "vad")

        self.assertTrue(blueprint.full_duplex)
        self.assertTrue(blueprint.barge_in_enabled)
        self.assertEqual(blueprint.asr_inference_mode, "streaming")
        self.assertEqual(blueprint.asr_transport_mode, "batch")
        self.assertEqual(blueprint.transport, "browser-batch")
        self.assertEqual(vad_stage.role, "start-end")
        self.assertEqual(asr_stage.provider, "foundry-local")
        self.assertIsNone(asr_stage.fallback_provider)

    def test_azure_embedded_asr_uses_streaming_transport_and_vad_start_only(self):
        settings = Settings.from_env({"VOICE_AGENT_ASR_PROVIDER": "azure-embedded"}, base_dir=Path("C:/workspace"))

        blueprint = build_pipeline_blueprint(settings)
        vad_stage = next(stage for stage in blueprint.stages if stage.name == "vad")

        self.assertEqual(blueprint.asr_inference_mode, "streaming")
        self.assertEqual(blueprint.asr_transport_mode, "streaming")
        self.assertEqual(blueprint.transport, "browser-streaming")
        self.assertEqual(blueprint.vad_role, "start")
        self.assertEqual(vad_stage.role, "start")


if __name__ == "__main__":
    unittest.main()
