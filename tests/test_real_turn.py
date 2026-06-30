import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.providers.asr import ASRTranscript
from voice_agent.providers.llm_foundry import ChatMessage
from voice_agent.real_turn import AzureEmbeddedTTSClient, run_real_turn, run_text_turn


class FakeASR:
    async def transcribe_audio(self, audio_bytes, filename, media_type, language="auto"):
        self.audio_bytes = audio_bytes
        self.filename = filename
        self.media_type = media_type
        return ASRTranscript("你好，测试真实链路", "zh", "fake-asr")


class FakeLLM:
    async def complete(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return "真实链路响应成功"


class FakeTTS:
    def synthesize(self, text: str) -> bytes:
        self.text = text
        return b"RIFF....WAVE"


class FakeVAD:
    async def analyze_audio(self, audio_bytes, filename, media_type):
        from voice_agent.providers.vad import VadDecision

        return VadDecision(True, 1.0, "fake-vad")


class RealTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_turn_uses_audio_asr_llm_and_tts(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        result = await run_real_turn(
            settings,
            audio_bytes=b"0" * 4096,
            filename="recording.webm",
            media_type="audio/webm",
            asr_client=FakeASR(),
            llm_client=FakeLLM(),
            tts_client=FakeTTS(),
            vad_client=FakeVAD(),
        )
        payload = result.as_dict()

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["userText"], "你好，测试真实链路")
        self.assertEqual(payload["assistantText"], "真实链路响应成功")
        self.assertEqual(payload["tts"]["audioMediaType"], "audio/wav")
        self.assertFalse(payload["tts"]["browserFallback"])

    async def test_real_turn_progress_events_keep_frontend_shape(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        events = []

        async def collect(event):
            events.append(event)

        await run_real_turn(
            settings,
            audio_bytes=b"0" * 4096,
            filename="recording.webm",
            media_type="audio/webm",
            asr_client=FakeASR(),
            llm_client=FakeLLM(),
            tts_client=FakeTTS(),
            vad_client=FakeVAD(),
            progress_callback=collect,
        )

        self.assertIn({"stage": "vad", "status": "running"}, events)
        self.assertTrue(any(event.get("stage") == "asr" and event.get("status") == "idle" and event.get("text") for event in events))
        self.assertTrue(any(event.get("stage") == "llm" and event.get("status") == "idle" and event.get("text") for event in events))
        self.assertTrue(any(event.get("stage") == "tts" and event.get("status") == "idle" for event in events))

    async def test_real_turn_rejects_tiny_audio(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        with self.assertRaises(RuntimeError):
            await run_real_turn(
                settings,
                audio_bytes=b"tiny",
                filename="recording.webm",
                media_type="audio/webm",
                asr_client=FakeASR(),
                llm_client=FakeLLM(),
                tts_client=FakeTTS(),
            )

    async def test_text_turn_uses_existing_user_text(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        llm = FakeLLM()

        result = await run_text_turn(
            settings,
            "streaming asr final text",
            llm_client=llm,
            tts_client=FakeTTS(),
            vad_ms=500,
            asr_ms=120,
        )
        payload = result.as_dict()

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["userText"], "streaming asr final text")
        self.assertEqual(payload["vad"]["provider"], "browser-vad")
        self.assertEqual(payload["asr"]["provider"], "azure-embedded")
        self.assertEqual(payload["timingsMs"]["vad"], 500)
        self.assertEqual(payload["timingsMs"]["asr"], 120)
        self.assertEqual(llm.messages[-1].content, "streaming asr final text")

    async def test_azure_embedded_tts_client_uses_audio_settings(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_TTS_PROVIDER": "azure-embedded"},
            base_dir=Path("C:/workspace"),
        )

        client = AzureEmbeddedTTSClient(settings)

        self.assertEqual(client.settings.audio.azure_embedded_tts_voice, "azure-embedded-zh-CN-XiaoxiaoNeuralHD")


if __name__ == "__main__":
    unittest.main()