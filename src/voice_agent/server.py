from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
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
from voice_agent.real_turn import DEFAULT_SYSTEM_PROMPT, check_foundry_ready, check_llama_cpp_ready, iter_warm_real_chain, prepare_llm_turn, run_real_turn, run_text_turn, warm_real_chain
from voice_agent.providers.llm_foundry import FoundryLocalLLM
from voice_agent.providers.llm_llama_cpp import LlamaCppLLM
from voice_agent.smoke import run_smoke_turn


def create_app(
    settings: Settings | None = None,
    *,
    audio_turn_runner=None,
    text_turn_runner=None,
    app_title: str = "Hybrid Voice Agent",
    turn_mode: str = "real",
):
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

    async def run_audio_turn(*args, **kwargs):
        runner = audio_turn_runner or run_real_turn
        return await runner(*args, **kwargs)

    async def run_text_turn_impl(*args, **kwargs):
        runner = text_turn_runner or run_text_turn
        return await runner(*args, **kwargs)

    llm_config = {"prompt": DEFAULT_SYSTEM_PROMPT, "context": ""}
    app = FastAPI(title=app_title, version="0.1.0")
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
        summary = resolved_settings.public_summary()
        summary["llmDefaults"] = dict(llm_config)
        return summary

    @app.get("/api/llm-config")
    async def get_llm_config() -> dict[str, str]:
        return dict(llm_config)

    @app.post("/api/llm-config")
    async def update_llm_config(payload: dict[str, str] | None = Body(default=None)) -> dict[str, str]:
        if payload is None:
            payload = {}
        prompt = str(payload.get("prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT
        context = str(payload.get("context") or "")
        llm_config["prompt"] = prompt
        llm_config["context"] = context
        return dict(llm_config)

    @app.get("/api/ready")
    async def ready() -> dict[str, object]:
        return {"foundry": await check_foundry_ready(resolved_settings), "llamaCpp": await check_llama_cpp_ready(resolved_settings)}

    @app.post("/api/models/load")
    async def load_models(payload: dict[str, str] | None = Body(default=None)) -> StreamingResponse:
        request_settings = _settings_with_request_options(resolved_settings, payload or {})

        async def event_stream():
            timings: dict[str, float] = {}
            models: dict[str, object] = {}
            try:
                async for event in iter_warm_real_chain(request_settings):
                    if event.get("event") == "model_loaded":
                        stage = str(event["stage"])
                        timings[stage] = float(event.get("latencyMs") or 0.0)
                        models[stage] = event.get("details", {})
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                yield json.dumps({"event": "result", "status": "ready", "timingsMs": timings, "models": models}, ensure_ascii=False) + "\n"
            except Exception as exc:
                yield json.dumps({"event": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False) + "\n"
            finally:
                yield json.dumps({"event": "done"}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.websocket("/api/azure-embedded/asr-ws")
    async def azure_embedded_asr_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        event_queue: queue.Queue[dict[str, object] | None] = queue.Queue()
        request_settings = resolved_settings
        started = False
        worker_task: asyncio.Task[None] | None = None
        pump_task: asyncio.Task[None] | None = None
        disconnected = False

        async def send_event(event: dict[str, object]) -> None:
            nonlocal disconnected
            if disconnected:
                raise WebSocketDisconnect()
            try:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
            except RuntimeError as exc:
                disconnected = True
                raise WebSocketDisconnect() from exc

        def recognize_worker(locale: str, model: str) -> None:
            try:
                import grpc
                from voice_agent.providers import azure_embedded_pb2 as pb
                from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc
            except Exception as exc:
                event_queue.put({"type": "error", "message": str(exc)})
                event_queue.put(None)
                return

            def requests():
                yield pb.AsrRequest(
                    config=pb.AsrConfig(
                        model=model,
                        locale=locale,
                        sample_rate_hz=16000,
                        channels=1,
                        bits_per_sample=16,
                    )
                )
                while True:
                    item = audio_queue.get()
                    if item is None:
                        yield pb.AsrRequest(end=True)
                        break
                    yield pb.AsrRequest(pcm16=item)

            channel = grpc.insecure_channel(request_settings.audio.azure_embedded_grpc_url)
            try:
                grpc.channel_ready_future(channel).result(timeout=3)
                stub = pb_grpc.AzureEmbeddedSpeechStub(channel)
                for event in stub.Recognize(requests(), timeout=60):
                    event_queue.put({
                        "type": event.type,
                        "model": event.model,
                        "locale": event.locale,
                        "text": event.text,
                        "bytes": event.bytes,
                        "elapsedMs": event.elapsed_ms,
                        "details": event.detail,
                        "message": event.detail,
                    })
                    if event.type in {"final", "error"}:
                        break
            except Exception as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                channel.close()
                event_queue.put(None)

        async def pump_events() -> None:
            try:
                while True:
                    event = await asyncio.to_thread(event_queue.get)
                    if event is None:
                        break
                    await send_event(event)
            except WebSocketDisconnect:
                audio_queue.put(None)

        try:
            while True:
                message = await websocket.receive()
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    message_type = payload.get("type")
                    if message_type == "start":
                        locale = str(payload.get("locale") or request_settings.audio.azure_embedded_asr_locale)
                        model = str(payload.get("model") or f"azure-embedded-{locale}-35M")
                        request_settings = replace(
                            resolved_settings,
                            audio=replace(resolved_settings.audio, azure_embedded_asr_locale=locale),
                        )
                        started = True
                        worker_task = asyncio.create_task(asyncio.to_thread(recognize_worker, locale, model))
                        pump_task = asyncio.create_task(pump_events())
                    elif message_type == "end":
                        audio_queue.put(None)
                        if worker_task is not None:
                            await worker_task
                        break
                elif message.get("bytes") is not None:
                    if started:
                        audio_queue.put(message["bytes"])
        except (WebSocketDisconnect, RuntimeError):
            disconnected = True
            audio_queue.put(None)
        except Exception as exc:
            try:
                await send_event({"type": "error", "message": str(exc)})
            except WebSocketDisconnect:
                return
        finally:
            if pump_task is not None:
                event_queue.put(None)
                await asyncio.gather(pump_task, return_exceptions=True)

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
        llm_model: str | None = Form(default=None),
        llm_prompt: str | None = Form(default=None),
        llm_context: str | None = Form(default=None),
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
                "llmModel": llm_model,
                "llmPrompt": llm_prompt,
                "llmContext": llm_context,
            },
        )
        try:
            result = await run_audio_turn(
                request_settings,
                audio_bytes,
                filename=audio.filename or "recording.webm",
                media_type=audio.content_type or "application/octet-stream",
                llm_prompt=_turn_llm_prompt(llm_prompt, llm_config),
                llm_context=_turn_llm_context(llm_context, llm_config),
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
        return {"mode": f"{turn_mode}-audio-turn", "result": result.as_dict()}

    @app.post("/api/session/turn-events")
    async def real_audio_turn_events(
        audio: UploadFile = File(...),
        tts_provider: str | None = Form(default=None),
        asr_provider: str | None = Form(default=None),
        asr_locale: str | None = Form(default=None),
        asr_model: str | None = Form(default=None),
        asr_language: str | None = Form(default=None),
        llm_model: str | None = Form(default=None),
        llm_prompt: str | None = Form(default=None),
        llm_context: str | None = Form(default=None),
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
                "llmModel": llm_model,
                "llmPrompt": llm_prompt,
                "llmContext": llm_context,
            },
        )

        async def event_stream():
            queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

            async def progress(event: dict[str, object]) -> None:
                await queue.put({"event": "progress", **event})

            async def run_turn():
                try:
                    result = await run_audio_turn(
                        request_settings,
                        audio_bytes,
                        filename=audio.filename or "recording.webm",
                        media_type=audio.content_type or "application/octet-stream",
                        progress_callback=progress,
                        llm_prompt=_turn_llm_prompt(llm_prompt, llm_config),
                        llm_context=_turn_llm_context(llm_context, llm_config),
                    )
                    await queue.put({"event": "result", "mode": f"{turn_mode}-audio-turn", "result": result.as_dict()})
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
        llm_prompt = None
        llm_context = None
        turn_task = None
        disconnect_task = None
        try:
            while True:
                message = await websocket.receive()
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "config":
                        request_settings = _settings_with_request_options(resolved_settings, payload)
                        filename = payload.get("filename") or filename
                        media_type = payload.get("mediaType") or media_type
                        llm_prompt = payload.get("llmPrompt")
                        llm_context = payload.get("llmContext")
                    if payload.get("type") == "end":
                        break
                elif message.get("bytes") is not None:
                    audio_chunks.append(message["bytes"])
            if not audio_chunks:
                raise RuntimeError("Turn stream ended without audio chunks")

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

            async def wait_for_disconnect() -> None:
                try:
                    while True:
                        await websocket.receive()
                except (WebSocketDisconnect, RuntimeError):
                    return

            turn_task = asyncio.create_task(
                run_audio_turn(
                    request_settings,
                    b"".join(audio_chunks),
                    filename=filename,
                    media_type=media_type,
                    progress_callback=progress,
                    llm_prompt=_turn_llm_prompt(llm_prompt, llm_config),
                    llm_context=_turn_llm_context(llm_context, llm_config),
                )
            )
            disconnect_task = asyncio.create_task(wait_for_disconnect())
            done, pending = await asyncio.wait({turn_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
                return
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            result = await turn_task
            result_payload = result.as_dict()
            result_payload["tts"]["audioBase64"] = None
            await websocket.send_text(json.dumps({"event": "result", "mode": f"{turn_mode}-audio-turn", "result": result_payload}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))
        except asyncio.CancelledError:
            if turn_task is not None:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
            if disconnect_task is not None:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
            return
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"event": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
            await websocket.send_text(json.dumps({"event": "done"}))

    @app.websocket("/api/session/text-turn-ws")
    async def real_text_turn_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        turn_task = None
        disconnect_task = None
        disconnected = False
        try:
            message = await websocket.receive()
            if message.get("text") is None:
                raise RuntimeError("Text turn expects an initial JSON config message")
            payload = json.loads(message["text"])
            async def safe_send_text(payload: dict[str, object]) -> None:
                nonlocal disconnected
                if disconnected:
                    raise WebSocketDisconnect()
                try:
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                except RuntimeError as exc:
                    disconnected = True
                    raise WebSocketDisconnect() from exc

            async def safe_send_bytes(payload: bytes) -> None:
                nonlocal disconnected
                if disconnected:
                    raise WebSocketDisconnect()
                try:
                    await websocket.send_bytes(payload)
                except RuntimeError as exc:
                    disconnected = True
                    raise WebSocketDisconnect() from exc

            async def progress(event: dict[str, object]) -> None:
                if event.get("stage") == "tts" and event.get("status") == "audio" and event.get("audioBase64"):
                    audio_bytes = base64.b64decode(str(event["audioBase64"]))
                    metadata = {key: value for key, value in event.items() if key != "audioBase64"}
                    metadata["event"] = "progress"
                    metadata["bytes"] = len(audio_bytes)
                    await safe_send_text(metadata)
                    await safe_send_bytes(audio_bytes)
                else:
                    await safe_send_text({"event": "progress", **event})

            request_settings = _settings_with_request_options(resolved_settings, payload)
            llm_prompt = _turn_llm_prompt(payload.get("llmPrompt"), llm_config)
            llm_context = _turn_llm_context(payload.get("llmContext"), llm_config)
            prepared_llm_turn = None
            if payload.get("type") in {"prepare_text_turn", "prepare"}:
                prepared_llm_turn = await prepare_llm_turn(_llm_client_for_settings(request_settings), llm_prompt=llm_prompt, llm_context=llm_context)
                await safe_send_text({"event": "prepared", "stage": "llm", "status": "prepared"})
                try:
                    while True:
                        message = await websocket.receive()
                        if message.get("text") is None:
                            continue
                        payload = json.loads(message["text"])
                        if payload.get("type") in {"text_turn", "turn"}:
                            break
                except (WebSocketDisconnect, RuntimeError):
                    return
                request_settings = _settings_with_request_options(request_settings, payload)
            user_text = str(payload.get("text") or payload.get("userText") or "").strip()
            if not user_text:
                raise RuntimeError("Text turn requires non-empty user text")

            async def wait_for_disconnect() -> None:
                try:
                    while True:
                        await websocket.receive()
                except (WebSocketDisconnect, RuntimeError):
                    return

            turn_task = asyncio.create_task(
                run_text_turn_impl(
                    request_settings,
                    user_text,
                    progress_callback=progress,
                    vad_provider=str(payload.get("vadProvider") or "browser-vad"),
                    asr_provider=str(payload.get("asrProvider") or "azure-embedded"),
                    vad_ms=_optional_float(payload.get("vadLatencyMs")),
                    asr_ms=_optional_float(payload.get("asrLatencyMs")),
                    llm_prompt=llm_prompt,
                    llm_context=llm_context,
                    prepared_llm_turn=prepared_llm_turn,
                )
            )
            disconnect_task = asyncio.create_task(wait_for_disconnect())
            done, pending = await asyncio.wait({turn_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
                return
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            result = await turn_task
            result_payload = result.as_dict()
            result_payload["tts"]["audioBase64"] = None
            await safe_send_text({"event": "result", "mode": f"{turn_mode}-text-turn", "result": result_payload})
            await safe_send_text({"event": "done"})
        except asyncio.CancelledError:
            if turn_task is not None:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
            if disconnect_task is not None:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
            return
        except WebSocketDisconnect:
            if turn_task is not None:
                turn_task.cancel()
                await asyncio.gather(turn_task, return_exceptions=True)
            return
        except Exception as exc:
            try:
                await safe_send_text({"event": "error", "type": type(exc).__name__, "message": str(exc)})
                await safe_send_text({"event": "done"})
            except WebSocketDisconnect:
                return

    @app.post("/api/session/backend-check")
    async def backend_real_chain_check(payload: dict[str, str] | None = Body(default=None)) -> dict[str, object]:
        request_settings = _settings_with_request_options(resolved_settings, payload or {})
        try:
            audio_bytes = synthesize_sapi_wav("Hello, how are you today?")
            result = await run_audio_turn(
                request_settings,
                audio_bytes,
                filename="backend-check.wav",
                media_type="audio/wav",
                llm_prompt=_turn_llm_prompt((payload or {}).get("llmPrompt"), llm_config),
                llm_context=_turn_llm_context((payload or {}).get("llmContext"), llm_config),
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
        return {"mode": f"backend-{turn_mode}-chain-check", "result": result.as_dict()}

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


def _turn_llm_prompt(value: object, llm_config: dict[str, str]) -> str:
    prompt = str(value or "").strip()
    return prompt or llm_config.get("prompt") or DEFAULT_SYSTEM_PROMPT


def _turn_llm_context(value: object, llm_config: dict[str, str]) -> str:
    if value is None:
        return llm_config.get("context", "")
    return str(value)


def _settings_with_request_options(settings: Settings, options: dict[str, str | None]) -> Settings:
    updated = _settings_with_tts_provider(settings, options.get("ttsProvider"))
    asr_provider = options.get("asrProvider")
    asr_locale = options.get("asrLocale")
    asr_model = options.get("asrModel")
    asr_language = options.get("asrLanguage")
    llm_provider = options.get("llmProvider")
    llm_model = options.get("llmModel")
    if llm_provider:
        providers = replace(updated.providers, llm=_normalize_llm_provider(llm_provider))
        providers.validate()
        updated = replace(updated, providers=providers)
    if llm_model and updated.providers.llm == "foundry-local":
        foundry = replace(updated.foundry, llm_model=llm_model)
        updated = replace(updated, foundry=foundry)
    if llm_model and updated.providers.llm == "llama-cpp":
        llama_cpp = replace(updated.llama_cpp, model=llm_model)
        updated = replace(updated, llama_cpp=llama_cpp)
    if asr_provider:
        providers = replace(updated.providers, asr=_normalize_asr_provider(asr_provider))
        providers.validate()
        updated = replace(updated, providers=providers)
    if asr_model and updated.providers.asr == "foundry-local":
        foundry = replace(updated.foundry, asr_model=asr_model)
        updated = replace(updated, foundry=foundry)
    if asr_model and updated.providers.asr == "faster-whisper":
        audio = replace(updated.audio, faster_whisper_model=asr_model)
        updated = replace(updated, audio=audio)
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


def _normalize_llm_provider(llm_provider: str) -> str:
    aliases = {
        "foundry-local": "foundry-local",
        "foundry": "foundry-local",
        "llama-cpp": "llama-cpp",
        "llamacpp": "llama-cpp",
    }
    normalized = aliases.get(llm_provider)
    if normalized is None:
        raise ValueError(f"Unsupported LLM provider selection: {llm_provider}")
    return normalized


def _llm_client_for_settings(settings: Settings):
    if settings.providers.llm == "llama-cpp":
        return LlamaCppLLM(settings.llama_cpp)
    return FoundryLocalLLM(settings.foundry)


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
