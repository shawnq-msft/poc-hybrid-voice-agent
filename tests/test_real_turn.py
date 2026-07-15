import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.providers.asr import ASRTranscript
from voice_agent.providers.lfm2_audio import LFM2AudioVoiceClient
from voice_agent.providers.llm_foundry import ChatMessage
from voice_agent.real_turn import AzureEmbeddedTTSClient, RealTurnResult, check_foundry_ready, iter_warm_real_chain, run_real_turn, run_text_turn, warm_real_chain


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

    async def test_real_turn_routes_lfm2_audio_after_vad_without_default_asr(self):
        settings = Settings.from_env(
            {
                "VOICE_AGENT_ASR_PROVIDER": "lfm2-audio",
                "VOICE_AGENT_LLM_PROVIDER": "lfm2-audio",
                "VOICE_AGENT_TTS_PROVIDER": "lfm2-audio",
            },
            base_dir=Path("C:/workspace"),
        )
        calls = []
        events = []

        async def collect(event):
            events.append(event)

        class FakeLFM2AudioVoiceClient:
            def __init__(self, client_settings):
                self.settings = client_settings

            def warm(self):
                calls.append({"warm": True})
                return {"provider": "lfm2-audio", "model": self.settings.lfm2_audio.model_id}

            async def run_audio_turn(self, audio_bytes, filename, media_type, **kwargs):
                calls.append({"audioBytes": len(audio_bytes), "filename": filename, "mediaType": media_type, **kwargs})
                if kwargs.get("progress_callback") is not None:
                    await kwargs["progress_callback"]({"stage": "llm", "status": "idle", "text": "端到端响应"})
                return RealTurnResult(
                    status="passed",
                    user_text="用户音频",
                    assistant_text="端到端响应",
                    vad_provider=kwargs.get("vad_provider", "fake-vad"),
                    asr_provider="direct-audio",
                    llm_provider="lfm2-audio",
                    tts_provider="lfm2-audio",
                    audio_media_type="audio/wav",
                    audio_base64="",
                    browser_tts_fallback=False,
                    timings_ms={"vad": kwargs.get("vad_ms", 0.0), "llm": 1.0, "tts": 2.0, "backendTotal": 3.0},
                    tts_voice=self.settings.lfm2_audio.model_id,
                )

        with patch("voice_agent.real_turn.LFM2AudioVoiceClient", FakeLFM2AudioVoiceClient):
            result = await run_real_turn(
                settings,
                audio_bytes=b"0" * 4096,
                filename="recording.webm",
                media_type="audio/webm",
                vad_client=FakeVAD(),
                progress_callback=collect,
            )

        self.assertEqual(result.asr_provider, "direct-audio")
        self.assertEqual(result.llm_provider, "lfm2-audio")
        self.assertEqual(result.tts_provider, "lfm2-audio")
        self.assertEqual(calls[0], {"warm": True})
        self.assertEqual(calls[1]["filename"], "recording.webm")
        self.assertEqual(calls[1]["vad_provider"], "fake-vad")
        self.assertFalse(calls[1]["emit_vad_progress"])
        self.assertEqual([event["stage"] for event in events], ["vad", "vad", "llm"])

    async def test_lfm2_audio_turn_emits_direct_audio_without_asr_progress(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        client = LFM2AudioVoiceClient(settings)
        events = []

        async def collect(event):
            events.append(event)

        def fake_run_audio_turn_sync(audio_bytes, filename, media_type, llm_prompt, llm_context, stream_audio_chunk=None):
            if stream_audio_chunk is not None:
                stream_audio_chunk(b"RIFF....WAVE", "端到端响应", 7.0, 7.0)
            return (
                "Audio input",
                "端到端响应",
                b"RIFF....WAVE",
                {"asr": 0.0, "llm": 7.0, "llmTotal": 8.0, "tts": 0.0, "ttsTotal": 0.0, "streamedAudioChunks": 1},
            )

        client._run_audio_turn_sync = fake_run_audio_turn_sync
        result = await client.run_audio_turn(
            b"0" * 4096,
            "recording.webm",
            "audio/webm",
            progress_callback=collect,
            emit_vad_progress=False,
        )

        self.assertEqual(result.asr_provider, "direct-audio")
        self.assertNotIn("asr", [event["stage"] for event in events])
        self.assertEqual(events[0]["stage"], "llm")
        self.assertTrue(any(event["stage"] == "tts" and event["status"] == "audio" for event in events))

    async def test_lfm2_audio_warm_reports_cache_status(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        client = LFM2AudioVoiceClient(settings)
        model_source = Path("C:/models/lfm2")
        cache_key = (settings.lfm2_audio.model_id, str(model_source), settings.lfm2_audio.allow_download)
        original_cache = LFM2AudioVoiceClient._components_cache
        LFM2AudioVoiceClient._components_cache = {}
        try:
            with (
                patch("voice_agent.providers.lfm2_audio._model_source", return_value=model_source),
                patch.object(client, "_load_components"),
            ):
                cold_details = client.warm()
                LFM2AudioVoiceClient._components_cache[cache_key] = object()
                warm_details = client.warm()
        finally:
            LFM2AudioVoiceClient._components_cache = original_cache

        self.assertFalse(cold_details["cached"])
        self.assertTrue(warm_details["cached"])

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

    async def test_text_turn_allows_empty_prompt_with_context_only(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        llm = FakeLLM()

        await run_text_turn(
            settings,
            "现在的问题",
            llm_client=llm,
            tts_client=FakeTTS(),
            llm_prompt="",
            llm_context="User: 你好\nAgent: 你好，有什么可以帮你？",
        )

        self.assertEqual([message.role for message in llm.messages], ["system", "user"])
        self.assertEqual(llm.messages[0].content, "Context:\nUser: 你好\nAgent: 你好，有什么可以帮你？")
        self.assertEqual(llm.messages[1].content, "现在的问题")

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
            patch("voice_agent.real_turn.check_foundry_ready", new=AsyncMock(return_value={"ready": True, "models": ["qwen2.5-0.5b-instruct-cuda-gpu:4"]})),
            patch("voice_agent.real_turn.warm_foundry_streaming_asr", return_value={"provider": "foundry-local", "model": "asr"}),
            patch("voice_agent.real_turn.FoundryLocalLLM", FakeFoundryLLM),
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

    async def test_warm_real_chain_preloads_lfm2_audio_once_for_asr_llm_tts(self):
        settings = Settings.from_env(
            {
                "VOICE_AGENT_ASR_PROVIDER": "lfm2-audio",
                "VOICE_AGENT_LLM_PROVIDER": "lfm2-audio",
                "VOICE_AGENT_TTS_PROVIDER": "lfm2-audio",
            },
            base_dir=Path("C:/workspace"),
        )
        calls = []

        class FakeLFM2AudioVoiceClient:
            def __init__(self, settings):
                self.settings = settings

            def warm(self):
                calls.append(self.settings.lfm2_audio.model_id)
                return {"provider": "lfm2-audio", "model": self.settings.lfm2_audio.model_id, "mode": "speech-to-speech"}

        with (
            patch("voice_agent.real_turn.warm_silero_vad"),
            patch("voice_agent.real_turn.LFM2AudioVoiceClient", FakeLFM2AudioVoiceClient),
        ):
            events = [event async for event in iter_warm_real_chain(settings)]

        loaded_events = [event for event in events if event["event"] == "model_loaded"]
        loading_events = [event for event in events if event["event"] == "model_loading"]
        self.assertEqual(calls, ["LiquidAI/LFM2.5-Audio-1.5B"])
        self.assertEqual([event["stage"] for event in loading_events], ["asr", "llm", "tts"])
        self.assertEqual([event["stage"] for event in loaded_events], ["vad", "asr", "llm", "tts"])
        self.assertEqual([event["details"].get("provider") for event in loaded_events[1:]], ["lfm2-audio", "lfm2-audio", "lfm2-audio"])
        self.assertTrue(all(event["details"].get("mode") == "speech-to-speech" for event in loaded_events[1:]))

    async def test_warm_real_chain_rejects_missing_foundry_llm_model_before_completion(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_LLM_PROVIDER": "foundry-local", "VOICE_AGENT_FOUNDRY_LLM_MODEL": "gemma-4-e2b"},
            base_dir=Path("C:/workspace"),
        )

        class FailingFoundryLLM:
            def __init__(self, settings):
                self.settings = settings

            async def complete(self, messages):
                raise AssertionError("Foundry completion should not be called for an unavailable model")

        with (
            patch("voice_agent.real_turn.warm_silero_vad"),
            patch("voice_agent.real_turn.warm_foundry_streaming_asr", return_value={"provider": "foundry-local", "model": "asr"}),
            patch(
                "voice_agent.real_turn.check_foundry_ready",
                new=AsyncMock(
                    side_effect=[
                        {
                            "ready": False,
                            "models": ["qwen2.5-0.5b-instruct-cuda-gpu:4"],
                            "error": "Configured Foundry LLM model is not loaded: gemma-4-e2b",
                        },
                    ]
                ),
            ),
            patch("voice_agent.real_turn.FoundryLocalLLM", FailingFoundryLLM),
        ):
            with self.assertRaisesRegex(RuntimeError, "gemma-4-e2b"):
                [event async for event in iter_warm_real_chain(settings)]

    async def test_foundry_ready_reports_configured_model_availability(self):
        settings = Settings.from_env(
            {"VOICE_AGENT_FOUNDRY_ENDPOINT": "http://foundry.test/v1", "VOICE_AGENT_FOUNDRY_LLM_MODEL": "gemma-4-e2b"},
            base_dir=Path("C:/workspace"),
        )

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"id": "qwen2.5-0.5b-instruct-cuda-gpu:4"}]}

        class FakeAsyncClient:
            def __init__(self, timeout):
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url):
                return FakeResponse()

        with patch("httpx.AsyncClient", FakeAsyncClient):
            status = await check_foundry_ready(settings)

        self.assertFalse(status["ready"])
        self.assertFalse(status["llmModelAvailable"])
        self.assertEqual(status["availableLlmModels"], ["qwen2.5-0.5b-instruct-cuda-gpu:4"])
        self.assertIn("gemma-4-e2b", status["error"])


if __name__ == "__main__":
    unittest.main()