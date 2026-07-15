import unittest
import tempfile
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Gemma4E2BSettings
from voice_agent.providers.asr import plan_asr_provider
from voice_agent.providers.gemma4_e2b import _parse_audio_turn_response, check_gemma_4_e2b_ready
from voice_agent.providers.vad import EnergyVad, plan_vad_provider
from voice_agent.providers.tts_windows import plan_windows_tts


class ProviderPlanTests(unittest.TestCase):
    def test_foundry_asr_plan_uses_requested_provider_directly(self):
        plan = plan_asr_provider("foundry-local", "auto", cloud_fallback_enabled=False)

        self.assertEqual(plan.primary, "foundry-local")
        self.assertIsNone(plan.fallback)

    def test_vad_plan_keeps_energy_fallback(self):
        plan = plan_vad_provider("silero")

        self.assertEqual(plan.fallback, "energy")

    def test_energy_vad_detects_loud_pcm(self):
        frame = (12000).to_bytes(2, "little", signed=True) * 160

        decision = EnergyVad(threshold=0.01).analyze_pcm16(frame)

        self.assertTrue(decision.is_speech)

    def test_windows_tts_plan_uses_sapi_fallback(self):
        plan = plan_windows_tts("windows-winrt", None, 24000)

        self.assertEqual(plan.fallback_provider, "windows-sapi")


class Gemma4E2BProviderTests(unittest.TestCase):
    def test_gemma_ready_reports_missing_local_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ready = check_gemma_4_e2b_ready(Gemma4E2BSettings(model_dir=Path(temp_dir) / "gemma"))

        self.assertFalse(ready["ready"])
        self.assertEqual(ready["runtime"], "hf-transformers")
        self.assertIn("gemma4", str(ready.get("modelType")))
        self.assertIn("torchVersion", ready)
        self.assertIn("gemma4_e2b_smoke.py --download", ready["error"])

    def test_gemma_audio_turn_parser_accepts_short_fused_format(self):
        transcript, assistant = _parse_audio_turn_response("T: What color is the sky today?\nA: The sky is blue.")

        self.assertEqual(transcript, "What color is the sky today?")
        self.assertEqual(assistant, "The sky is blue.")

    def test_gemma_audio_turn_parser_ignores_gemma_turn_markers(self):
        transcript, assistant = _parse_audio_turn_response("<start_of_turn>model\nT: Sky color?\nA: Blue.<end_of_turn>")

        self.assertEqual(transcript, "Sky color?")
        self.assertEqual(assistant, "Blue.")

    def test_gemma_audio_turn_parser_accepts_conversation_labels(self):
        transcript, assistant = _parse_audio_turn_response("User: Sky color?\nAssistant: Blue.")

        self.assertEqual(transcript, "Sky color?")
        self.assertEqual(assistant, "Blue.")


if __name__ == "__main__":
    unittest.main()
