import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.pipecat_runtime import build_runtime_plan
from voice_agent.pipecat_runtime.events import progress_event


class PipecatRuntimePlanTests(unittest.TestCase):
    def test_foundry_asr_upload_stream_still_uses_vad_for_turn_end(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        plan = build_runtime_plan(settings)

        self.assertEqual(plan.input_lane, "streaming-asr")
        self.assertTrue(plan.vad_starts_stream)
        self.assertTrue(plan.vad_ends_turn)

    def test_streaming_asr_lane_uses_vad_for_start_only(self):
        settings = Settings.from_env({"VOICE_AGENT_ASR_PROVIDER": "azure-embedded"}, base_dir=Path("C:/workspace"))

        plan = build_runtime_plan(settings)

        self.assertEqual(plan.input_lane, "streaming-asr")
        self.assertTrue(plan.vad_starts_stream)
        self.assertFalse(plan.vad_ends_turn)

    def test_pipeline_event_keeps_frontend_progress_shape(self):
        event = progress_event("llm", "token", text="hello", latency_ms=12.3)

        self.assertEqual(
            event.as_progress(),
            {"stage": "llm", "status": "token", "text": "hello", "latencyMs": 12.3},
        )


if __name__ == "__main__":
    unittest.main()
