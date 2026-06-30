import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.server import create_app


class ServerTests(unittest.TestCase):
    def test_session_start_returns_smoke_result(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.from_env(
                {
                    "VOICE_AGENT_COPILOT_AUDIT_LOG": "audit.jsonl",
                    "VOICE_AGENT_COPILOT_TOOLS_DRY_RUN": "true",
                },
                base_dir=Path(temp_dir),
            )
            client = TestClient(create_app(settings))

            response = client.post("/api/session/start", json={"text": "测试 UI 会话"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "smoke-session")
            self.assertEqual(payload["result"]["status"], "passed")
            self.assertEqual(payload["result"]["userText"], "测试 UI 会话")

    def test_legacy_offer_route_no_longer_returns_501(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        response = client.post("/api/session/offer", json={"type": "offer", "sdp": "placeholder"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["status"], "passed")

    def test_ready_reports_foundry_key(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        response = client.get("/api/ready")

        self.assertEqual(response.status_code, 200)
        self.assertIn("foundry", response.json())


if __name__ == "__main__":
    unittest.main()
