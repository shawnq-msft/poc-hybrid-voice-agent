import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.pipecat_runtime.pure_turn import run_pure_pipecat_text_turn
from voice_agent.providers.llm_foundry import ChatMessage


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


if __name__ == "__main__":
    unittest.main()
