import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.providers.asr import ASRTranscript
from voice_agent.providers.llm_foundry import ChatMessage
from voice_agent.real_turn import AzureEmbeddedTTSClient, iter_warm_real_chain, run_real_turn, run_text_turn, warm_real_chain


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


class SlowStreamingLLM:
    def __init__(self):
        self.closed = False

    async def stream(self, messages: list[ChatMessage]):
        try:
            yield "你好，"
            await asyncio.Event().wait()
        finally:
            self.closed = True


class SlowTTS:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def synthesize_async(self, text: str) -> bytes:
        self.started.set()
        try:
            await asyncio.Event().wait()
            return b"RIFF....WAVE"
        finally:
            self.cancelled = True


class FakeVAD:
    async def analyze_audio(self, audio_bytes, filename, media_type):
        from voice_agent.providers.vad import VadDecision

        return VadDecision(True, 1.0, "fake-vad")


class SlowASR:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe_audio(self, audio_bytes, filename, media_type, language="auto"):
        self.started.set()
        await self.release.wait()
        return ASRTranscript("提前准备测试", "zh", "fake-asr")


class PreparingLLM(FakeLLM):
    def __init__(self):
        self.prepare_started = asyncio.Event()
        self.prepare_can_finish = asyncio.Event()

    async def prepare_turn(self, messages):
        self.prepare_started.set()
        await self.prepare_can_finish.wait()
        from voice_agent.real_turn import PreparedMessagesLLMTurn

        return PreparedMessagesLLMTurn(self, list(messages))


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

    async def test_real_turn_prepares_llm_during_asr(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        asr = SlowASR()
        llm = PreparingLLM()
        task = asyncio.create_task(
            run_real_turn(
                settings,
                audio_bytes=b"0" * 4096,
                filename="recording.webm",
                media_type="audio/webm",
                asr_client=asr,
                llm_client=llm,
                tts_client=FakeTTS(),
                vad_client=FakeVAD(),
            )
        )

        await asyncio.wait_for(asr.started.wait(), timeout=1)
        await asyncio.wait_for(llm.prepare_started.wait(), timeout=1)
        self.assertFalse(task.done())
        llm.prepare_can_finish.set()
        asr.release.set()
        result = await asyncio.wait_for(task, timeout=1)

        self.assertEqual(result.user_text, "提前准备测试")
        self.assertEqual(llm.messages[-1].content, "提前准备测试")

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

    async def test_text_turn_uses_custom_prompt_and_context(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        llm = FakeLLM()

        await run_text_turn(
            settings,
            "用户问题",
            llm_client=llm,
            tts_client=FakeTTS(),
            llm_prompt="用中文简洁回答。",
            llm_context="当前场景：本地语音助手。",
        )

        self.assertEqual([message.role for message in llm.messages], ["system", "system", "user"])
        self.assertEqual(llm.messages[0].content, "用中文简洁回答。")
        self.assertEqual(llm.messages[1].content, "Context:\n当前场景：本地语音助手。")
        self.assertEqual(llm.messages[2].content, "用户问题")

    async def test_text_turn_cancellation_stops_streaming_llm_and_tts(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        llm = SlowStreamingLLM()
        tts = SlowTTS()

        task = asyncio.create_task(run_text_turn(settings, "打断测试", llm_client=llm, tts_client=tts))
        await asyncio.wait_for(tts.started.wait(), timeout=1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(llm.closed)
        self.assertTrue(tts.cancelled)

    async def test_azure_embedded_tts_client_uses_audio_settings(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_TTS_PROVIDER": "azure-embedded"},
            base_dir=Path("C:/workspace"),
        )

        client = AzureEmbeddedTTSClient(settings)

        self.assertEqual(client.settings.audio.azure_embedded_tts_voice, "azure-embedded-zh-CN-XiaoxiaoNeuralV6")

    async def test_warm_real_chain_emits_sequential_model_load_events(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))

        class FakeFoundryLLM:
            def __init__(self, settings):
                self.settings = settings

            async def complete(self, messages):
                return "OK"

        with (
            patch("voice_agent.real_turn.warm_silero_vad"),
            patch("voice_agent.real_turn.check_llama_cpp_ready", new=AsyncMock(return_value={"ready": True, "model": "gemma-4-e2b"})),
            patch("voice_agent.real_turn.warm_foundry_streaming_asr", return_value={"provider": "foundry-local", "model": "asr"}),
            patch("voice_agent.real_turn.LlamaCppLLM", FakeFoundryLLM),
            patch("voice_agent.real_turn._check_azure_embedded_health", return_value={"provider": "azure-embedded", "model": "tts"}),
            patch("voice_agent.real_turn._synthesize_tts", new=AsyncMock(return_value=(b"RIFF....WAVE", "audio/wav"))),
        ):
            events = [event async for event in iter_warm_real_chain(settings)]
            summary = await warm_real_chain(settings)

        self.assertEqual([event["stage"] for event in events], ["vad", "asr", "llm", "tts"])
        self.assertTrue(all(event["event"] == "model_loaded" for event in events))
        self.assertTrue(all("latencyMs" in event for event in events))
        self.assertIn("memoryRssMb", events[-1])
        self.assertEqual(summary["status"], "ready")
        self.assertIn("tts", summary["timingsMs"])


if __name__ == "__main__":
    unittest.main()