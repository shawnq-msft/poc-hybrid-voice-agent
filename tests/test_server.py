import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch

import _path  # noqa: F401

from voice_agent.config import Settings
from voice_agent.pipecat_server import create_pipecat_app
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

        self.assertEqual(updated.llama_cpp.model, "custom-local-llm")

    def test_request_options_apply_foundry_llm_model(self):
        settings = Settings.from_env({}, base_dir=Path.cwd())

        updated = _settings_with_request_options(settings, {"llmProvider": "foundry-local", "llmModel": "gemma-4-e2b"})

        self.assertEqual(updated.providers.llm, "foundry-local")
        self.assertEqual(updated.foundry.llm_model, "gemma-4-e2b")

    def test_request_options_apply_llama_cpp_llm(self):
        settings = Settings.from_env({}, base_dir=Path.cwd())

        updated = _settings_with_request_options(settings, {"llmProvider": "llama-cpp", "llmModel": "gemma-4-e2b"})

        self.assertEqual(updated.providers.llm, "llama-cpp")
        self.assertEqual(updated.llama_cpp.model, "gemma-4-e2b")

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

    def test_pipecat_app_uses_pure_pipecat_mode(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        async def fake_audio_turn(*args, **kwargs):
            from voice_agent.real_turn import RealTurnResult

            return RealTurnResult(
                status="passed",
                user_text="audio user",
                assistant_text="OK",
                vad_provider="browser-vad",
                asr_provider="azure-embedded",
                llm_provider="foundry-local",
                tts_provider="azure-embedded",
                audio_media_type=None,
                audio_base64=None,
                browser_tts_fallback=False,
                timings_ms={"llm": 1.0, "tts": 1.0},
            )

        settings = Settings.from_env({}, base_dir=Path.cwd())
        self.assertEqual(create_pipecat_app(settings).title, "Pure Pipecat Voice Agent")
        client = TestClient(
            create_app(settings, audio_turn_runner=fake_audio_turn, app_title="Pure Pipecat Voice Agent", turn_mode="pure-pipecat")
        )

        response = client.post(
            "/api/session/turn",
            files={"audio": ("recording.webm", b"fake audio bytes", "audio/webm")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "pure-pipecat-audio-turn")

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

    def test_text_turn_websocket_can_prepare_before_final_text(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        captured = {}

        class FakePreparedTurn:
            def with_user_text(self, user_text):
                return self

        async def fake_prepare_llm_turn(*args, **kwargs):
            captured["prepared"] = True
            captured["prompt"] = kwargs.get("llm_prompt")
            captured["context"] = kwargs.get("llm_context")
            return FakePreparedTurn()

        async def fake_text_turn(*args, **kwargs):
            captured["user_text"] = args[1]
            captured["prepared_llm_turn"] = kwargs.get("prepared_llm_turn")
            from voice_agent.real_turn import RealTurnResult

            return RealTurnResult(
                status="passed",
                user_text=args[1],
                assistant_text="OK",
                vad_provider="browser-vad",
                asr_provider="azure-embedded",
                llm_provider="foundry-local",
                tts_provider="azure-embedded",
                audio_media_type=None,
                audio_base64=None,
                browser_tts_fallback=False,
                timings_ms={"llm": 1.0, "tts": 1.0},
            )

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        with (
            patch("voice_agent.server.prepare_llm_turn", new=fake_prepare_llm_turn),
            patch("voice_agent.server.run_text_turn", new=fake_text_turn),
        ):
            with client.websocket_connect("/api/session/text-turn-ws") as websocket:
                websocket.send_json({"type": "prepare_text_turn", "llmPrompt": "Prompt", "llmContext": "Context"})
                self.assertEqual(websocket.receive_json()["event"], "prepared")
                websocket.send_json({"type": "text_turn", "text": "ASR final"})
                events = []
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event.get("event") == "done":
                        break

        self.assertTrue(captured["prepared"])
        self.assertEqual(captured["prompt"], "Prompt")
        self.assertEqual(captured["context"], "Context")
        self.assertEqual(captured["user_text"], "ASR final")
        self.assertIsInstance(captured["prepared_llm_turn"], FakePreparedTurn)
        self.assertTrue(any(event.get("event") == "result" for event in events))

    def test_text_turn_websocket_prepare_disconnect_is_clean(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        class FakePreparedTurn:
            def with_user_text(self, user_text):
                return self

        async def fake_prepare_llm_turn(*args, **kwargs):
            return FakePreparedTurn()

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(create_app(settings))

        with patch("voice_agent.server.prepare_llm_turn", new=fake_prepare_llm_turn):
            with client.websocket_connect("/api/session/text-turn-ws") as websocket:
                websocket.send_json({"type": "prepare_text_turn"})
                self.assertEqual(websocket.receive_json()["event"], "prepared")

    def test_pipecat_text_turn_websocket_uses_pure_runner(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI TestClient is not installed")

        captured = {}

        async def fake_text_turn(*args, **kwargs):
            captured["user_text"] = args[1]
            captured["prompt"] = kwargs.get("llm_prompt")
            captured["context"] = kwargs.get("llm_context")
            from voice_agent.real_turn import RealTurnResult

            return RealTurnResult(
                status="passed",
                user_text=args[1],
                assistant_text="OK",
                vad_provider="browser-vad",
                asr_provider="azure-embedded",
                llm_provider="foundry-local",
                tts_provider="azure-embedded",
                audio_media_type=None,
                audio_base64=None,
                browser_tts_fallback=False,
                timings_ms={"llm": 1.0, "tts": 1.0},
            )

        settings = Settings.from_env({}, base_dir=Path.cwd())
        client = TestClient(
            create_app(settings, text_turn_runner=fake_text_turn, app_title="Pure Pipecat Voice Agent", turn_mode="pure-pipecat")
        )

        with client.websocket_connect("/api/session/text-turn-ws") as websocket:
            websocket.send_json({"type": "text_turn", "text": "Pipecat text", "llmPrompt": "Prompt", "llmContext": "Context"})
            events = []
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event.get("event") == "done":
                    break

        self.assertEqual(captured["user_text"], "Pipecat text")
        self.assertEqual(captured["prompt"], "Prompt")
        self.assertEqual(captured["context"], "Context")
        self.assertTrue(any(event.get("mode") == "pure-pipecat-text-turn" for event in events))


if __name__ == "__main__":
    unittest.main()
