import unittest
from pathlib import Path
from unittest.mock import patch

import _path  # noqa: F401

from voice_agent.config import LFM2_AUDIO_PROVIDER, Settings
from voice_agent.pipecat_runtime.pure_turn import run_pure_pipecat_text_turn
from voice_agent.providers.llm_foundry import ChatMessage
from voice_agent.real_turn import RealTurnResult


class FakeLLM:
    async def complete(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return "pure pipecat response"


class FakeTTS:
    def synthesize(self, text: str) -> bytes:
        self.text = text
        return b"RIFF....WAVE"


class PurePipecatTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_turn_runs_through_pipecat_pipeline(self):
        settings = Settings.from_env({}, base_dir=Path("C:/workspace"))
        llm = FakeLLM()
        tts = FakeTTS()
        events = []

        async def progress(event):
            events.append(event)

        result = await run_pure_pipecat_text_turn(
            settings,
            "hello from pure pipecat",
            llm_client=llm,
            tts_client=tts,
            progress_callback=progress,
            llm_prompt="You are a helpful voice assistant.",
            llm_context="test context",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.user_text, "hello from pure pipecat")
        self.assertEqual(result.assistant_text, "pure pipecat response")
        self.assertEqual(tts.text, "pure pipecat response")
        self.assertEqual([message.role for message in llm.messages], ["system", "system", "user"])
        self.assertEqual(llm.messages[-1].content, "hello from pure pipecat")
        self.assertTrue(any(event.get("stage") == "llm" for event in events))

    async def test_lfm2_audio_chain_uses_unified_voice_client(self):
        settings = Settings.from_env(
            {
                "VOICE_AGENT_ASR_PROVIDER": LFM2_AUDIO_PROVIDER,
                "VOICE_AGENT_LLM_PROVIDER": LFM2_AUDIO_PROVIDER,
                "VOICE_AGENT_TTS_PROVIDER": LFM2_AUDIO_PROVIDER,
            },
            base_dir=Path("C:/workspace"),
        )
        calls = []

        class FakeLFM2AudioClient:
            def __init__(self, settings):
                self.settings = settings

            async def run_text_turn(self, user_text, **kwargs):
                calls.append((user_text, kwargs))
                return RealTurnResult(
                    status="passed",
                    user_text=user_text,
                    assistant_text="lfm audio response",
                    vad_provider=kwargs["vad_provider"],
                    asr_provider=kwargs["asr_provider"],
                    llm_provider=LFM2_AUDIO_PROVIDER,
                    tts_provider=LFM2_AUDIO_PROVIDER,
                    audio_media_type="audio/wav",
                    audio_base64="UklGRg==",
                    browser_tts_fallback=False,
                    timings_ms={"backendTotal": 1.0},
                    tts_voice=settings.lfm2_audio.model_id,
                )

        with patch("voice_agent.pipecat_runtime.pure_turn.LFM2AudioVoiceClient", new=FakeLFM2AudioClient):
            result = await run_pure_pipecat_text_turn(settings, "hello lfm", llm_prompt="", llm_context="User: hi")

        self.assertEqual(result.assistant_text, "lfm audio response")
        self.assertEqual(result.llm_provider, LFM2_AUDIO_PROVIDER)
        self.assertEqual(result.tts_provider, LFM2_AUDIO_PROVIDER)
        self.assertEqual(calls[0][0], "hello lfm")
        self.assertEqual(calls[0][1]["llm_context"], "User: hi")


if __name__ == "__main__":
    unittest.main()
