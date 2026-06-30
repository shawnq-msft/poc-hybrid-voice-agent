import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.smoke import run_smoke_turn


class SmokeTests(unittest.TestCase):
    def test_mock_end_to_end_smoke_turn_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.from_env(
                {
                    "VOICE_AGENT_COPILOT_AUDIT_LOG": "audit.jsonl",
                    "VOICE_AGENT_COPILOT_TOOLS_DRY_RUN": "true",
                },
                base_dir=Path(temp_dir),
            )

            result = run_smoke_turn(settings, user_text="请 Copilot 总结这个项目")
            payload = result.as_dict()

            self.assertEqual(payload["status"], "passed")
            self.assertTrue(payload["vad"]["isSpeech"])
            self.assertEqual(payload["asr"]["configuredProvider"], "foundry-local")
            self.assertEqual(payload["tool"]["status"], "dry-run")
            self.assertTrue((Path(temp_dir) / "audit.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
