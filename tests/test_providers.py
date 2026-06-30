import unittest

import _path  # noqa: F401

from voice_agent.providers.asr import plan_asr_provider
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


if __name__ == "__main__":
    unittest.main()
