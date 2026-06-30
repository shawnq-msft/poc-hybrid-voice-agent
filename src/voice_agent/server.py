from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import replace
from pathlib import Path

try:
    from fastapi import UploadFile, WebSocket
except ImportError:  # pragma: no cover - create_app raises a clearer install error.
    UploadFile = object  # type: ignore[assignment]
    WebSocket = object  # type: ignore[assignment]

from voice_agent.config import Settings
from voice_agent.providers.tts_windows import synthesize_sapi_wav
from voice_agent.health import collect_health
from voice_agent.real_turn import check_foundry_ready, run_real_turn, run_text_turn, warm_real_chain
from voice_agent.smoke import run_smoke_turn


def create_app(settings: Settings | None = None):
    try:
        from fastapi import Body, FastAPI, File, Form, HTTPException, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install server dependencies with `python -m pip install -e .`") from exc

    if settings is None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
    resolved_settings = settings or Settings.from_env(os.environ)
    app = FastAPI(title="Hybrid Voice Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def warm_models_on_startup() -> None:
        async def warm() -> None:
            try:
                await warm_real_chain(resolved_settings)
            except Exception:
                pass

        asyncio.create_task(warm())

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return collect_health(resolved_settings)

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        return resolved_settings.public_summary()

    @app.get("/api/ready")
    async def ready() -> dict[str, object]:
        return {"foundry": await check_foundry_ready(resolved_settings)}

    @app.post("/api/models/load")
    async def load_models(payload: dict[str, str] | None = Body(default=None)) -> dict[str, object]:
        request_settings = _settings_with_request_options(resolved_settings, payload or {})
        try:
            return await warm_real_chain(request_settings)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"type": type(exc).__name__, "message": str(exc)},
            ) from exc

    @app.get("/api/smoke")
    async def smoke() -> dict[str, object]:
        return run_smoke_turn(resolved_settings).as_dict()

    @app.post("/api/smoke")
    async def smoke_with_prompt(payload: dict[str, str]) -> dict[str, object]:
        user_text = payload.get("text") or "Open Copilot and summarize this workspace"
        return run_smoke_turn(resolved_settings, user_text=user_text).as_dict()

    @app.post("/api/session/start")
    async def start_smoke_session(payload: dict[str, str]) -> dict[str, object]:
        user_text = payload.get("text") or "Open Copilot and summarize this workspace"
        result = run_smoke_turn(resolved_settings, user_text=user_text).as_dict()
        return {"mode": "smoke-session", "media": "microphone-permission-only", "result": result}

    @app.post("/api/session/offer")
    async def create_session_offer(payload: dict[str, str]) -> dict[str, object]:
        user_text = payload.get("text") or "Open Copilot and summarize this workspace"
        result = run_smoke_turn(resolved_settings, user_text=user_text).as_dict()
        return {"mode": "smoke-session", "media": "webrtc-not-yet-enabled", "result": result}

    @app.post("/api/session/turn")
    async def real_audio_turn(
        audio: UploadFile = File(...),
        tts_provider: str | None = Form(default=None),
        asr_provider: str | None = Form(default=None),
        asr_locale: str | None = Form(default=None),
        asr_model: str | None = Form(default=None),
        asr_language: str | None = Form(default=None),
    ) -> dict[str, object]:
        audio_bytes = await audio.read()
        request_settings = _settings_with_request_options(
            resolved_settings,
            {
                "ttsProvider": tts_provider,
                "asrProvider": asr_provider,
                "asrLocale": asr_locale,
                "asrModel": asr_model,
                "asrLanguage": asr_language,
            },
        )
        try:
            result = await run_real_turn(
                request_settings,
                audio_bytes,
                filename=audio.filename or "recording.webm",
                media_type=audio.content_type or "application/octet-stream",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "hint": "Start Foundry Local, confirm /v1/models, and set exact ASR/LLM model IDs in .env.",
                },
            ) from exc
        return {"mode": "real-audio-turn", "result": result.as_dict()}

    @app.post("/api/session/turn-events")
    async def real_audio_turn_events(
        audio: UploadFile = File(...),
        tts_provider: str | None = Form(default=None),
        asr_provider: str | None = Form(default=None),
        asr_locale: str | None = Form(default=None),
        asr_model: str | None = Form(default=None),
        asr_language: str | None = Form(default=None),
    ) -> StreamingResponse:
        audio_bytes = await audio.read()
        request_settings = _settings_with_request_options(
            resolved_settings,
            {
                "ttsProvider": tts_provider,
                "asrProvider": asr_provider,
                "asrLocale": asr_locale,
                "asrModel": asr_model,
                "asrLanguage": asr_language,
            },
        )

        async def event_stream():
            queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

            async def progress(event: dict[str, object]) -> None:
                await queue.put({"event": "progress", **event})

            async def run_turn():
                try:
                    result = await run_real_turn(
                        request_settings,
                        audio_bytes,
                        filename=audio.filename or "recording.webm",
                        media_type=audio.content_type or "application/octet-stream",
                        progress_callback=progress,
                    )
                    await queue.put({"event": "result", "mode": "real-audio-turn", "result": result.as_dict()})
                except Exception as exc:
                    await queue.put({"event": "error", "type": type(exc).__name__, "message": str(exc)})
                finally:
                    await queue.put({"event": "done"})

            task = asyncio.create_task(run_turn())
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event.get("event") == "done":
                    break
            await task

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.websocket("/api/session/turn-ws")
    async def real_audio_turn_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        request_settings = resolved_settings
        audio_chunks: list[bytes] = []
        filename = "recording.webm"
        media_type = "audio/webm"
        try:
            while True:
                message = await websocket.receive()
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "config":
                        request_settings = _settings_with_request_options(resolved_settings, payload)
                        filename = payload.get("filename") or filename
                        media_type = payload.get("mediaType") or media_type
                    if payload.get("type") == "end":
                        break
                elif message.get("bytes") is not None:
                    audio_chunks.append(message["bytes"])

            async def progress(event: dict[str, object]) -> None:
                if event.get("stage") == "tts" and event.get("status") == "audio" and event.get("audioBase64"):
                    audio_bytes = base64.b64decode(str(event["audioBase64"]))
                    metadata = {key: value for key, value in event.items() if key != "audioBase64"}
                    metadata["event"] = "progress"
                    metadata["bytes"] = len(audio_bytes)
                    await websocket.send_text(json.dumps(metadata, ensure_ascii=False))
                    await websocket.send_bytes(audio_bytes)
                else:
                    await websocket.send_text(json.dumps({"event": "progress", **event}, ensure_ascii=False))

            result = await run_real_turn(
                request_settings,
                b"".join(audio_chunks),
                filename=filename,
                media_type=media_type,
                progress_callback=progress,
            )
            result_payload = result.as_dict()
            result_payload["tts"]["audioBase64"] = None
            await websocket.send_text(json.dumps({"event": "result", "mode": "real-audio-turn", "result": result_payload}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"event": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))

    @app.websocket("/api/session/text-turn-ws")
    async def real_text_turn_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            message = await websocket.receive()
            if message.get("text") is None:
                raise RuntimeError("Text turn expects an initial JSON config message")
            payload = json.loads(message["text"])
            request_settings = _settings_with_request_options(resolved_settings, payload)
            user_text = str(payload.get("text") or payload.get("userText") or "").strip()
            if not user_text:
                raise RuntimeError("Text turn requires non-empty user text")

            async def progress(event: dict[str, object]) -> None:
                if event.get("stage") == "tts" and event.get("status") == "audio" and event.get("audioBase64"):
                    audio_bytes = base64.b64decode(str(event["audioBase64"]))
                    metadata = {key: value for key, value in event.items() if key != "audioBase64"}
                    metadata["event"] = "progress"
                    metadata["bytes"] = len(audio_bytes)
                    await websocket.send_text(json.dumps(metadata, ensure_ascii=False))
                    await websocket.send_bytes(audio_bytes)
                else:
                    await websocket.send_text(json.dumps({"event": "progress", **event}, ensure_ascii=False))

            result = await run_text_turn(
                request_settings,
                user_text,
                progress_callback=progress,
                vad_provider=str(payload.get("vadProvider") or "browser-vad"),
                asr_provider=str(payload.get("asrProvider") or "azure-embedded"),
                vad_ms=_optional_float(payload.get("vadLatencyMs")),
                asr_ms=_optional_float(payload.get("asrLatencyMs")),
            )
            result_payload = result.as_dict()
            result_payload["tts"]["audioBase64"] = None
            await websocket.send_text(json.dumps({"event": "result", "mode": "real-text-turn", "result": result_payload}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"event": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))

    @app.post("/api/session/backend-check")
    async def backend_real_chain_check(payload: dict[str, str] | None = Body(default=None)) -> dict[str, object]:
        request_settings = _settings_with_request_options(resolved_settings, payload or {})
        try:
            audio_bytes = synthesize_sapi_wav("Hello, how are you today?")
            result = await run_real_turn(
                request_settings,
                audio_bytes,
                filename="backend-check.wav",
                media_type="audio/wav",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "hint": "Check Foundry Local, faster-whisper, and Windows TTS.",
                },
            ) from exc
        return {"mode": "backend-real-chain-check", "result": result.as_dict()}

    web_dir = resolved_settings.server.web_dir
    if web_dir.exists():
        app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")
    else:

        @app.get("/")
        async def missing_web() -> dict[str, str]:
            return {"status": "web client directory not found", "path": str(web_dir)}

    return app


def _settings_with_tts_provider(settings: Settings, tts_provider: str | None) -> Settings:
    if not tts_provider:
        return settings
    normalized = _normalize_tts_provider(tts_provider)
    providers = replace(settings.providers, tts=normalized)
    providers.validate()
    return replace(settings, providers=providers)


def _settings_with_request_options(settings: Settings, options: dict[str, str | None]) -> Settings:
    updated = _settings_with_tts_provider(settings, options.get("ttsProvider"))
    asr_provider = options.get("asrProvider")
    asr_locale = options.get("asrLocale")
    asr_model = options.get("asrModel")
    asr_language = options.get("asrLanguage")
    if asr_provider:
        providers = replace(updated.providers, asr=_normalize_asr_provider(asr_provider))
        providers.validate()
        updated = replace(updated, providers=providers)
    if asr_model and updated.providers.asr == "foundry-local":
        foundry = replace(updated.foundry, asr_model=asr_model)
        updated = replace(updated, foundry=foundry)
    if asr_language:
        audio = replace(updated.audio, asr_language=asr_language)
        updated = replace(updated, audio=audio)
    if asr_locale and asr_locale != "auto":
        audio = replace(updated.audio, azure_embedded_asr_locale=asr_locale)
        updated = replace(updated, audio=audio)
    return updated


def _normalize_asr_provider(asr_provider: str) -> str:
    aliases = {
        "foundry-local": "foundry-local",
        "faster-whisper": "faster-whisper",
        "azure-embedded": "azure-embedded",
    }
    normalized = aliases.get(asr_provider)
    if normalized is None:
        raise ValueError(f"Unsupported ASR provider selection: {asr_provider}")
    return normalized


def _normalize_tts_provider(tts_provider: str) -> str:
    aliases = {
        "edge": "edge-tts",
        "edge-tts": "edge-tts",
        "windows-sapi": "windows-sapi",
        "sapi": "windows-sapi",
        "windows-winrt": "windows-winrt",
        "azure-speech": "azure-speech",
        "azure-embedded": "azure-embedded",
    }
    normalized = aliases.get(tts_provider)
    if normalized is None:
        raise ValueError(f"Unsupported TTS provider selection: {tts_provider}")
    return normalized


def _optional_float(value: object) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    try:
        import uvicorn
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install uvicorn with `python -m pip install -e .`") from exc

    load_dotenv(dotenv_path=Path.cwd() / ".env")
    settings = Settings.from_env(os.environ, base_dir=Path.cwd())
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
