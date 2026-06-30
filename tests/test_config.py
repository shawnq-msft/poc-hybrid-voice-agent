import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings, discover_foundry_endpoint


class SettingsTests(unittest.TestCase):
    def test_defaults_are_local_first(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        self.assertEqual(settings.providers.asr, "foundry-local")
        self.assertEqual(settings.providers.llm, "foundry-local")
        self.assertEqual(settings.providers.tts, "windows-sapi")
        self.assertFalse(settings.providers.cloud_fallback_enabled)
        self.assertEqual(settings.foundry.llm_model, "qwen2.5-0.5b-instruct-cuda-gpu:4")
        self.assertEqual(settings.foundry.asr_model, "nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(settings.audio.edge_tts_voice, "zh-CN-XiaoxiaoNeural")
        self.assertEqual(settings.audio.azure_embedded_grpc_url, "127.0.0.1:8792")
        self.assertEqual(settings.audio.azure_embedded_tts_voice, "azure-embedded-zh-CN-XiaoxiaoNeuralHD")
        self.assertEqual(settings.audio.azure_embedded_asr_locale, "en-GB")
        self.assertEqual(settings.audio.azure_embedded_asr_sidecar_url, "ws://127.0.0.1:8791/asr")
        self.assertEqual(
            settings.audio.azure_embedded_asr_zh_cn_model_dir,
            Path("C:/workspace/models/azure-embedded/asr/zh-CN/encrypted/35M"),
        )
        self.assertIsNone(settings.audio.pasco_model_key)

    def test_public_summary_exposes_asr_transport_capabilities(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        capabilities = settings.public_summary()["providers"]["asrCapabilities"]

        self.assertEqual(capabilities["azure-embedded"]["transportMode"], "streaming")
        self.assertEqual(capabilities["azure-embedded"]["vadRole"], "start")
        self.assertEqual(capabilities["foundry-local"]["inferenceMode"], "streaming")
        self.assertEqual(capabilities["foundry-local"]["transportMode"], "batch")
        self.assertEqual(capabilities["faster-whisper"]["vadRole"], "start-end")

    def test_rejects_unknown_asr_provider(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"VOICE_AGENT_ASR_PROVIDER": "random-cloud"})

    def test_relative_paths_are_resolved_under_base_dir(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_COPILOT_AUDIT_LOG": ".voice-agent/test.jsonl"},
            base_dir=Path("C:/workspace"),
        )

        self.assertEqual(settings.copilot_tools.audit_log_path, Path("C:/workspace/.voice-agent/test.jsonl"))

    def test_explicit_foundry_endpoint_wins(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_FOUNDRY_ENDPOINT": "http://127.0.0.1:9999/v1"},
            base_dir=Path("C:/workspace"),
        )

        self.assertEqual(settings.foundry.endpoint, "http://127.0.0.1:9999/v1")


if __name__ == "__main__":
    unittest.main()
