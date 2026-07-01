import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.server import _settings_with_request_options, create_app


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

    def test_request_options_apply_faster_whisper_model(self):
        settings = Settings.from_env({}, base_dir=Path.cwd())

        updated = _settings_with_request_options(
            settings,
            {"asrProvider": "faster-whisper", "asrModel": "base", "asrLanguage": "en"},
        )

        self.assertEqual(updated.providers.asr, "faster-whisper")
        self.assertEqual(updated.audio.faster_whisper_model, "base")
        self.assertEqual(updated.audio.asr_language, "en")

    def test_request_options_apply_llm_model(self):
        settings = Settings.from_env({}, base_dir=Path.cwd())

        updated = _settings_with_request_options(settings, {"llmModel": "custom-local-llm"})

        self.assertEqual(updated.foundry.llm_model, "custom-local-llm")

    def test_llm_config_endpoint_updates_prompt_and_context(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        response = client.post("/api/llm-config", json={"prompt": "Custom prompt", "context": "Custom context"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prompt": "Custom prompt", "context": "Custom context"})
        self.assertEqual(client.get("/api/llm-config").json(), {"prompt": "Custom prompt", "context": "Custom context"})

    def test_text_turn_websocket_disconnect_cancels_turn(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        cancelled = False
        started = None

        async def slow_text_turn(*args, **kwargs):
            nonlocal cancelled, started
            started = asyncio.Event()
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled = True
                raise

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        with patch("voice_agent.server.run_text_turn", new=slow_text_turn):
            with client.websocket_connect("/api/session/text-turn-ws") as websocket:
                websocket.send_json({"type": "text_turn", "text": "打断测试"})

        self.assertTrue(cancelled)


if __name__ == "__main__":
    unittest.main()
