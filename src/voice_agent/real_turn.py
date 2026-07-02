from __future__ import annotations

import base64
import inspect
import asyncio
import os
import socket
from time import perf_counter
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Protocol

from voice_agent.config import Settings
from voice_agent.pipecat_runtime.events import ProgressCallback, emit_progress, progress_event
from voice_agent.providers.asr import ASRTranscript, AzureEmbeddedASR, FasterWhisperASR, FoundryLocalASR, warm_faster_whisper, warm_foundry_streaming_asr
from voice_agent.providers.llm_foundry import ChatMessage, FoundryLocalLLM
from voice_agent.providers.llm_llama_cpp import LlamaCppLLM
from voice_agent.providers.tts_windows import AsyncAzureEmbeddedTTSGrpcClient, synthesize_azure_embedded_wav, synthesize_edge_mp3, synthesize_sapi_wav
from voice_agent.providers.vad import SileroVad, VadDecision, warm_silero_vad


class ASRClient(Protocol):
    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        language: str = "auto",
    ) -> ASRTranscript:
        ...


class LLMClient(Protocol):
    async def complete(self, messages: list[ChatMessage]) -> str:
        ...


class PreparedLLMTurn(Protocol):
    def with_user_text(self, user_text: str) -> "PreparedLLMTurn":
        ...


class TTSClient(Protocol):
    def synthesize(self, text: str) -> bytes:
        ...


class VADClient(Protocol):
    async def analyze_audio(self, audio_bytes: bytes, filename: str, media_type: str) -> VadDecision:
        ...


DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."


@dataclass(frozen=True)
class SapiTTSClient:
    voice: str | None = None

    def synthesize(self, text: str) -> bytes:
        return synthesize_sapi_wav(text, self.voice)


@dataclass(frozen=True)
class EdgeTTSClient:
    voice: str = "zh-CN-XiaoxiaoNeural"

    async def synthesize(self, text: str) -> bytes:
        return await synthesize_edge_mp3(text, self.voice)


@dataclass
class AzureEmbeddedTTSClient:
    settings: Settings
    _client: AsyncAzureEmbeddedTTSGrpcClient | None = None

    def synthesize(self, text: str) -> bytes:
        return synthesize_azure_embedded_wav(text, self.settings.audio)

    async def synthesize_async(self, text: str) -> bytes:
        if self._client is None:
            self._client = AsyncAzureEmbeddedTTSGrpcClient(self.settings.audio)
        return await self._client.synthesize(text)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


@dataclass(frozen=True)
class RealTurnResult:
    status: str
    user_text: str
    assistant_text: str
    vad_provider: str
    asr_provider: str
    llm_provider: str
    tts_provider: str
    audio_media_type: str | None
    audio_base64: str | None
    browser_tts_fallback: bool
    timings_ms: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "userText": self.user_text,
            "assistantText": self.assistant_text,
            "vad": {"provider": self.vad_provider},
            "asr": {"provider": self.asr_provider},
            "llm": {"provider": self.llm_provider},
            "tts": {
                "provider": self.tts_provider,
                "audioMediaType": self.audio_media_type,
                "audioBase64": self.audio_base64,
                "browserFallback": self.browser_tts_fallback,
            },
            "timingsMs": self.timings_ms,
        }


@dataclass(frozen=True)
class AssistantTurnPayload:
    assistant_text: str
    tts_provider: str
    audio_media_type: str | None
    audio_base64: str | None
    browser_tts_fallback: bool
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class PreparedMessagesLLMTurn:
    llm: object
    messages: list[ChatMessage]

    def with_user_text(self, user_text: str) -> "PreparedMessagesLLMTurn":
        return PreparedMessagesLLMTurn(self.llm, [*self.messages, ChatMessage("user", user_text)])

    async def complete(self) -> str:
        return await self.llm.complete(self.messages)

    async def stream(self) -> AsyncIterator[str]:
        async for token in self.llm.stream(self.messages):
            yield token


async def run_real_turn(
    settings: Settings,
    audio_bytes: bytes,
    filename: str,
    media_type: str,
    asr_client: ASRClient | None = None,
    llm_client: LLMClient | None = None,
    tts_client: TTSClient | None = None,
    vad_client: VADClient | None = None,
    progress_callback: ProgressCallback | None = None,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
) -> RealTurnResult:
    total_started = perf_counter()
    if len(audio_bytes) < 128:
        raise RuntimeError("Recorded audio is too small to transcribe")

    asr = asr_client or _default_asr_client(settings)
    llm = llm_client or _default_llm_client(settings)
    owns_tts = tts_client is None
    tts = tts_client or _default_tts_client(settings)

    vad_started = perf_counter()
    vad = vad_client or _default_vad_client(settings)
    await _emit_progress(progress_callback, "vad", "running")
    vad_decision = await vad.analyze_audio(audio_bytes, filename, media_type)
    vad_ms = _elapsed_ms(vad_started)
    if not vad_decision.is_speech:
        await _emit_progress(progress_callback, "vad", "failed", latency_ms=vad_ms)
        raise RuntimeError("Silero VAD detected no speech")
    await _emit_progress(progress_callback, "vad", "idle", latency_ms=vad_ms)

    llm_prepare_task = asyncio.create_task(prepare_llm_turn(llm, llm_prompt=llm_prompt, llm_context=llm_context))
    asr_started = perf_counter()
    await _emit_progress(progress_callback, "asr", "running")
    try:
        transcript = await _transcribe_with_local_fallback(settings, asr, audio_bytes, filename, media_type)
        asr_ms = _elapsed_ms(asr_started)
        if not transcript.text.strip():
            await _emit_progress(progress_callback, "asr", "failed", latency_ms=asr_ms)
            raise RuntimeError("ASR returned empty text")
        await _emit_progress(progress_callback, "asr", "idle", latency_ms=asr_ms, text=transcript.text)
    except Exception:
        llm_prepare_task.cancel()
        await asyncio.gather(llm_prepare_task, return_exceptions=True)
        raise

    try:
        prepared_llm_turn = await llm_prepare_task
        assistant_turn = await _complete_assistant_turn(
            settings,
            transcript.text,
            llm,
            tts,
            progress_callback,
            llm_prompt=llm_prompt,
            llm_context=llm_context,
            prepared_llm_turn=prepared_llm_turn,
        )
    finally:
        if owns_tts and hasattr(tts, "close"):
            close_result = tts.close()
            if inspect.isawaitable(close_result):
                await close_result

    return RealTurnResult(
        status="passed",
        user_text=transcript.text,
        assistant_text=assistant_turn.assistant_text,
        vad_provider=vad_decision.provider,
        asr_provider=transcript.provider,
        llm_provider=settings.providers.llm,
        tts_provider=assistant_turn.tts_provider,
        audio_media_type=assistant_turn.audio_media_type,
        audio_base64=assistant_turn.audio_base64,
        browser_tts_fallback=assistant_turn.browser_tts_fallback,
        timings_ms={
            "vad": vad_ms,
            "asr": asr_ms,
            **assistant_turn.timings_ms,
            "backendTotal": _elapsed_ms(total_started),
        },
    )


async def run_text_turn(
    settings: Settings,
    user_text: str,
    llm_client: LLMClient | None = None,
    tts_client: TTSClient | None = None,
    progress_callback: ProgressCallback | None = None,
    vad_provider: str = "browser-vad",
    asr_provider: str = "azure-embedded",
    vad_ms: float = 0.0,
    asr_ms: float = 0.0,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
    prepared_llm_turn: PreparedLLMTurn | None = None,
) -> RealTurnResult:
    total_started = perf_counter()
    normalized_text = user_text.strip()
    if not normalized_text:
        raise RuntimeError("Text turn requires non-empty user text")

    llm = llm_client or _default_llm_client(settings)
    owns_tts = tts_client is None
    tts = tts_client or _default_tts_client(settings)
    try:
        if prepared_llm_turn is None:
            prepared_llm_turn = await prepare_llm_turn(llm, llm_prompt=llm_prompt, llm_context=llm_context)
        assistant_turn = await _complete_assistant_turn(settings, normalized_text, llm, tts, progress_callback, llm_prompt=llm_prompt, llm_context=llm_context, prepared_llm_turn=prepared_llm_turn)
    finally:
        if owns_tts and hasattr(tts, "close"):
            close_result = tts.close()
            if inspect.isawaitable(close_result):
                await close_result
    return RealTurnResult(
        status="passed",
        user_text=normalized_text,
        assistant_text=assistant_turn.assistant_text,
        vad_provider=vad_provider,
        asr_provider=asr_provider,
        llm_provider=settings.providers.llm,
        tts_provider=assistant_turn.tts_provider,
        audio_media_type=assistant_turn.audio_media_type,
        audio_base64=assistant_turn.audio_base64,
        browser_tts_fallback=assistant_turn.browser_tts_fallback,
        timings_ms={
            "vad": vad_ms,
            "asr": asr_ms,
            **assistant_turn.timings_ms,
            "backendTotal": _elapsed_ms(total_started),
        },
    )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 1)


async def _emit_progress(
    progress_callback: ProgressCallback | None,
    module: str,
    phase: str,
    *,
    text: str | None = None,
    latency_ms: float | None = None,
    total_ms: float | None = None,
    audio_base64: str | None = None,
    audio_media_type: str | None = None,
) -> None:
    await emit_progress(
        progress_callback,
        progress_event(
            module,
            phase,
            text=text,
            latency_ms=latency_ms,
            total_ms=total_ms,
            audio_base64=audio_base64,
            audio_media_type=audio_media_type,
        ),
    )


def _default_asr_client(settings: Settings) -> ASRClient:
    if settings.providers.asr == "foundry-local":
        return FoundryLocalASR(settings.foundry)
    if settings.providers.asr == "faster-whisper":
        return FasterWhisperASR(settings.audio.faster_whisper_model)
    if settings.providers.asr == "azure-embedded":
        return AzureEmbeddedASR(settings.audio)
    raise RuntimeError(f"Real turn ASR provider is not implemented: {settings.providers.asr}")


def _default_vad_client(settings: Settings) -> VADClient:
    if settings.providers.vad == "silero":
        return SileroVad()
    return SileroVad()


def _default_tts_client(settings: Settings):
    if settings.providers.tts == "edge-tts":
        return EdgeTTSClient(settings.audio.edge_tts_voice)
    if settings.providers.tts == "azure-embedded":
        return AzureEmbeddedTTSClient(settings)
    return SapiTTSClient(settings.audio.windows_tts_voice)


def _default_llm_client(settings: Settings):
    if settings.providers.llm == "llama-cpp":
        return LlamaCppLLM(settings.llama_cpp)
    if settings.providers.llm == "foundry-local":
        return FoundryLocalLLM(settings.foundry)
    raise RuntimeError(f"Real turn LLM provider is not implemented: {settings.providers.llm}")


async def prepare_llm_turn(
    llm,
    *,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
) -> PreparedLLMTurn:
    messages = _base_llm_messages(llm_prompt=llm_prompt, llm_context=llm_context)
    if hasattr(llm, "prepare_turn"):
        prepared = llm.prepare_turn(messages)
        if inspect.isawaitable(prepared):
            return await prepared
        return prepared
    return PreparedMessagesLLMTurn(llm, messages)


def _base_llm_messages(*, llm_prompt: str | None = None, llm_context: str | None = None) -> list[ChatMessage]:
    system_prompt = (llm_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    messages = [ChatMessage("system", system_prompt)]
    context = (llm_context or "").strip()
    if context:
        messages.append(ChatMessage("system", f"Context:\n{context}"))
    return messages


async def _synthesize_tts(tts, text: str) -> tuple[bytes, str]:
    if hasattr(tts, "synthesize_async"):
        audio_bytes = await tts.synthesize_async(text)
    elif isinstance(tts, EdgeTTSClient):
        output = tts.synthesize(text)
        if inspect.isawaitable(output):
            audio_bytes = await output
        else:
            audio_bytes = output
    else:
        audio_bytes = await asyncio.to_thread(tts.synthesize, text)
    media_type = "audio/mpeg" if isinstance(tts, EdgeTTSClient) else "audio/wav"
    return audio_bytes, media_type


async def _complete_assistant_turn(
    settings: Settings,
    user_text: str,
    llm,
    tts,
    progress_callback: ProgressCallback | None = None,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
    prepared_llm_turn: PreparedLLMTurn | None = None,
) -> AssistantTurnPayload:
    llm_turn = prepared_llm_turn or await prepare_llm_turn(llm, llm_prompt=llm_prompt, llm_context=llm_context)
    llm_turn = llm_turn.with_user_text(user_text)
    await _emit_progress(progress_callback, "llm", "running")
    assistant_text, llm_ms, llm_first_sentence_ms, llm_total_ms, tts_payload = await _run_llm_and_tts(
        llm_turn,
        tts,
        progress_callback,
    )
    if not assistant_text.strip():
        await _emit_progress(progress_callback, "llm", "failed")
        raise RuntimeError("LLM returned empty text")
    await _emit_progress(progress_callback, "llm", "idle", latency_ms=llm_ms, total_ms=llm_total_ms, text=assistant_text)

    audio_base64 = None
    audio_media_type = None
    browser_tts_fallback = False
    tts_provider = settings.providers.tts
    try:
        audio_bytes, audio_media_type, tts_ms, tts_total_ms = tts_payload
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        await _emit_progress(progress_callback, "tts", "idle", latency_ms=tts_ms, total_ms=tts_total_ms)
    except Exception:
        browser_tts_fallback = True
        tts_provider = "browser-speechSynthesis-fallback"
        tts_started = perf_counter()
        await _emit_progress(progress_callback, "tts", "running")
        audio_bytes, audio_media_type = await _synthesize_tts(tts, assistant_text)
        tts_ms = _elapsed_ms(tts_started)
        tts_total_ms = tts_ms
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        await _emit_progress(progress_callback, "tts", "idle", latency_ms=tts_ms, total_ms=tts_total_ms)

    return AssistantTurnPayload(
        assistant_text=assistant_text,
        tts_provider=tts_provider,
        audio_media_type=audio_media_type,
        audio_base64=audio_base64,
        browser_tts_fallback=browser_tts_fallback,
        timings_ms={
            "llm": llm_ms,
            "llmFirstSentence": llm_first_sentence_ms,
            "llmTotal": llm_total_ms,
            "tts": tts_ms,
            "ttsTotal": tts_total_ms,
        },
    )


async def _run_llm_and_tts(
    llm_turn,
    tts,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, float, float, float, tuple[bytes, str, float, float]]:
    if hasattr(llm_turn, "stream"):
        try:
            return await _stream_llm_and_tts(llm_turn, tts, progress_callback)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    llm_started = perf_counter()
    assistant_text = await llm_turn.complete()
    llm_ms = _elapsed_ms(llm_started)
    await _emit_progress(progress_callback, "llm", "ttft", latency_ms=llm_ms)
    await _emit_progress(progress_callback, "llm", "first_sentence", latency_ms=llm_ms)
    tts_started = perf_counter()
    await _emit_progress(progress_callback, "tts", "running")
    audio_bytes, audio_media_type = await _synthesize_tts(tts, assistant_text)
    tts_ms = _elapsed_ms(tts_started)
    return assistant_text, llm_ms, llm_ms, llm_ms, (audio_bytes, audio_media_type, tts_ms, tts_ms)


async def _stream_llm_and_tts(
    llm_turn,
    tts,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, float, float, float, tuple[bytes, str, float, float]]:
    llm_started = perf_counter()
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    assistant_parts: list[str] = []
    pending_text = ""
    tts_tasks: list[asyncio.Task[tuple[bytes, str, float]]] = []

    try:
        async for token in llm_turn.stream():
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(llm_started)
                await _emit_progress(progress_callback, "llm", "ttft", latency_ms=first_token_ms)
            await _emit_progress(progress_callback, "llm", "token", text=token)
            assistant_parts.append(token)
            pending_text += token
            while True:
                split_index = _first_sentence_boundary(pending_text)
                if split_index is None:
                    break
                segment = pending_text[: split_index + 1].strip()
                pending_text = pending_text[split_index + 1 :]
                if segment:
                    if first_sentence_ms is None:
                        first_sentence_ms = _elapsed_ms(llm_started)
                        await _emit_progress(progress_callback, "llm", "first_sentence", latency_ms=first_sentence_ms)
                    await _emit_progress(progress_callback, "tts", "running", text=segment)
                    tts_tasks.append(asyncio.create_task(_timed_synthesize_tts(tts, segment, progress_callback)))

        if pending_text.strip():
            if first_sentence_ms is None:
                first_sentence_ms = _elapsed_ms(llm_started)
                await _emit_progress(progress_callback, "llm", "first_sentence", latency_ms=first_sentence_ms)
            tts_tasks.append(asyncio.create_task(_timed_synthesize_tts(tts, pending_text.strip(), progress_callback)))

        assistant_text = "".join(assistant_parts).strip()
        if not assistant_text:
            raise RuntimeError("LLM stream returned empty text")
        if not tts_tasks:
            tts_tasks.append(asyncio.create_task(_timed_synthesize_tts(tts, assistant_text, progress_callback)))

        tts_results = [await task for task in tts_tasks]
        first_tts_ms = tts_results[0][2]
        segment_tts_total_ms = round(sum(result[2] for result in tts_results), 1)
        full_tts_started = perf_counter()
        audio_bytes, audio_media_type = await _synthesize_tts(tts, assistant_text)
        full_tts_ms = _elapsed_ms(full_tts_started)
        total_tts_ms = round(segment_tts_total_ms + full_tts_ms, 1)
        return assistant_text, first_token_ms or _elapsed_ms(llm_started), first_sentence_ms or _elapsed_ms(llm_started), _elapsed_ms(llm_started), (
            audio_bytes,
            audio_media_type,
            first_tts_ms,
            total_tts_ms,
        )
    except asyncio.CancelledError:
        for task in tts_tasks:
            task.cancel()
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)
        raise


async def _timed_synthesize_tts(
    tts,
    text: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bytes, str, float]:
    started = perf_counter()
    audio_bytes, audio_media_type = await _synthesize_tts(tts, text)
    latency_ms = _elapsed_ms(started)
    await _emit_progress(
        progress_callback,
        "tts",
        "audio",
        latency_ms=latency_ms,
        audio_media_type=audio_media_type,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
        text=text,
    )
    return audio_bytes, audio_media_type, latency_ms


def _first_sentence_boundary(text: str) -> int | None:
    for index, char in enumerate(text):
        if char in ".!?。！？；;，,":
            return index
    return None


async def _transcribe_with_local_fallback(
    settings: Settings,
    asr: ASRClient,
    audio_bytes: bytes,
    filename: str,
    media_type: str,
) -> ASRTranscript:
    try:
        return await asr.transcribe_audio(
            audio_bytes,
            filename=filename,
            media_type=media_type or "application/octet-stream",
            language=settings.audio.asr_language,
        )
    except Exception:
        if settings.providers.asr != "azure-embedded":
            raise
        fallback = FasterWhisperASR(settings.audio.faster_whisper_model)
        return await fallback.transcribe_audio(
            audio_bytes,
            filename=filename,
            media_type=media_type or "application/octet-stream",
            language=settings.audio.asr_language,
        )


async def check_foundry_ready(settings: Settings) -> dict[str, object]:
    try:
        import httpx
    except ImportError:
        return {"ready": False, "error": "httpx is not installed"}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.foundry.endpoint.rstrip('/')}/models")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return {"ready": False, "endpoint": settings.foundry.endpoint, "error": str(exc)}

    model_ids = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str):
            model_ids.append(model_id)
    return {
        "ready": True,
        "endpoint": settings.foundry.endpoint,
        "models": model_ids,
        "llmModelConfigured": settings.foundry.llm_model,
        "asrModelConfigured": settings.foundry.asr_model,
    }


async def check_llama_cpp_ready(settings: Settings) -> dict[str, object]:
    return await LlamaCppLLM(settings.llama_cpp).health()


async def warm_real_chain(settings: Settings) -> dict[str, object]:
    timings: dict[str, float] = {}
    statuses: dict[str, object] = {}

    async for event in iter_warm_real_chain(settings):
        if event.get("event") != "model_loaded":
            continue
        stage = str(event["stage"])
        timings[stage] = float(event.get("latencyMs") or 0.0)
        details = event.get("details")
        statuses[stage] = details if isinstance(details, dict) else {}

    return {"status": "ready", "timingsMs": timings, "models": statuses}


async def iter_warm_real_chain(settings: Settings) -> AsyncIterator[dict[str, object]]:
    async def emit_loaded(
        stage: str,
        details: dict[str, object],
        started: float,
        memory_before: int,
        process_id: int | None = None,
    ) -> dict[str, object]:
        memory_after = _rss_bytes_for_pid(process_id) if process_id is not None else None
        if memory_after is None:
            memory_after = _current_rss_bytes()
        return {
            "event": "model_loaded",
            "stage": stage,
            "status": "loaded",
            "latencyMs": _elapsed_ms(started),
            "memoryRssMb": _bytes_to_mb(memory_after),
            "memoryDeltaMb": _bytes_to_mb(memory_after - memory_before),
            "memorySource": "sidecar" if process_id is not None else "server",
            "details": details,
        }

    started = perf_counter()
    memory_before = _current_rss_bytes()
    await asyncio.to_thread(warm_silero_vad)
    yield await emit_loaded("vad", {"provider": "silero"}, started, memory_before)

    started = perf_counter()
    memory_before = _current_rss_bytes()
    if settings.providers.asr == "foundry-local":
        ready = await check_foundry_ready(settings)
        if not ready.get("ready"):
            raise RuntimeError(f"Foundry Local ASR is not ready: {ready.get('error')}")
        asr_status = await asyncio.to_thread(warm_foundry_streaming_asr, settings.foundry.asr_model)
        yield await emit_loaded("asr", {**asr_status, "language": settings.audio.asr_language, "endpoint": settings.foundry.endpoint}, started, memory_before)
    elif settings.providers.asr == "azure-embedded":
        sidecar_pid = _pid_for_grpc_url(settings.audio.azure_embedded_grpc_url)
        memory_before = _rss_bytes_for_pid(sidecar_pid) or memory_before
        asr_status = await asyncio.to_thread(_warm_azure_embedded_asr_sidecar, settings)
        yield await emit_loaded("asr", {**asr_status, "processId": sidecar_pid}, started, memory_before, process_id=sidecar_pid)
    else:
        await asyncio.to_thread(warm_faster_whisper, settings.audio.faster_whisper_model, "cpu", "int8")
        yield await emit_loaded("asr", {"provider": "faster-whisper", "model": settings.audio.faster_whisper_model}, started, memory_before)

    started = perf_counter()
    memory_before = _current_rss_bytes()
    if settings.providers.llm == "llama-cpp":
        ready = await check_llama_cpp_ready(settings)
        if not ready.get("ready"):
            raise RuntimeError(f"llama.cpp LLM is not ready: {ready.get('error')}")
        await LlamaCppLLM(settings.llama_cpp).complete(
            [ChatMessage("system", "Reply with OK only."), ChatMessage("user", "OK?")]
        )
        yield await emit_loaded("llm", {"provider": "llama-cpp", "model": settings.llama_cpp.model, "endpoint": settings.llama_cpp.endpoint, "slotId": settings.llama_cpp.slot_id}, started, memory_before)
    else:
        ready = await check_foundry_ready(settings)
        if not ready.get("ready"):
            raise RuntimeError(f"Foundry Local LLM is not ready: {ready.get('error')}")
        await FoundryLocalLLM(settings.foundry).complete(
            [ChatMessage("system", "Reply with OK only."), ChatMessage("user", "OK?")]
        )
        yield await emit_loaded("llm", {"provider": "foundry-local", "model": settings.foundry.llm_model, "endpoint": settings.foundry.endpoint}, started, memory_before)

    started = perf_counter()
    memory_before = _current_rss_bytes()
    if settings.providers.tts == "azure-embedded":
        sidecar_pid = _pid_for_grpc_url(settings.audio.azure_embedded_tts_grpc_url)
        memory_before = _rss_bytes_for_pid(sidecar_pid) or memory_before
        tts_status = await asyncio.to_thread(_check_azure_embedded_health, settings.audio.azure_embedded_tts_grpc_url, "tts", settings.audio.azure_embedded_tts_voice)
        await _synthesize_tts(_default_tts_client(settings), "Ready.")
        yield await emit_loaded("tts", {**tts_status, "voice": settings.audio.azure_embedded_tts_voice, "processId": sidecar_pid}, started, memory_before, process_id=sidecar_pid)
    else:
        await _synthesize_tts(_default_tts_client(settings), "Ready.")
        tts_voice = settings.audio.edge_tts_voice if settings.providers.tts == "edge-tts" else settings.audio.windows_tts_voice
        yield await emit_loaded("tts", {"provider": settings.providers.tts, "voice": tts_voice}, started, memory_before)


def _current_rss_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _rss_bytes_for_pid(process_id: int | None) -> int | None:
    if process_id is None:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process(process_id).memory_info().rss)
    except Exception:
        return None


def _bytes_to_mb(value: int) -> float | None:
    if value == 0:
        return None
    return round(value / (1024 * 1024), 1)


def _check_azure_embedded_health(grpc_url: str, kind: str, expected: str) -> dict[str, object]:
    try:
        import grpc
        from voice_agent.providers import azure_embedded_pb2 as pb
        from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc
    except ImportError as exc:
        raise RuntimeError("Install grpcio and generate Azure Embedded gRPC stubs to use azure-embedded") from exc

    channel = grpc.insecure_channel(grpc_url)
    try:
        grpc.channel_ready_future(channel).result(timeout=2.0)
        response = pb_grpc.AzureEmbeddedSpeechStub(channel).Health(pb.HealthRequest(), timeout=5)
    except Exception as exc:
        message = _grpc_error_message(exc)
        suffix = f": {message}" if message else ""
        raise RuntimeError(f"Azure Embedded {kind.upper()} sidecar is not reachable at {grpc_url}{suffix}. Start the native sidecar for this port before loading models.") from exc
    finally:
        channel.close()

    models = response.asr_models if kind == "asr" else response.tts_models
    statuses = [
        {
            "id": model.id,
            "locale": model.locale,
            "path": model.path,
            "loaded": bool(model.loaded),
            "detail": model.detail,
        }
        for model in models
    ]
    match = next((model for model in statuses if model["id"] == expected or model["locale"] == expected), None)
    if match is None:
        known = ", ".join(str(model["id"]) for model in statuses) or "none"
        raise RuntimeError(f"Azure Embedded {kind.upper()} sidecar at {grpc_url} does not expose {expected}; available: {known}")
    if match.get("loaded") is False and "missing" in str(match.get("detail", "")):
        raise RuntimeError(f"Azure Embedded {kind.upper()} model {expected} is not loadable: {match.get('detail')}")
    return {"provider": "azure-embedded", "url": grpc_url, "model": expected, "sidecarStatus": response.status, "modelStatus": match}


def _warm_azure_embedded_asr_sidecar(settings: Settings) -> dict[str, object]:
    status = _check_azure_embedded_health(settings.audio.azure_embedded_grpc_url, "asr", settings.audio.azure_embedded_asr_locale)
    pcm16 = b"\x00\x00" * 3200
    try:
        import grpc
        from voice_agent.providers import azure_embedded_pb2 as pb
        from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc
    except ImportError as exc:
        raise RuntimeError("Install grpcio and generate Azure Embedded gRPC stubs to use azure-embedded ASR") from exc

    def requests():
        yield pb.AsrRequest(
            config=pb.AsrConfig(
                locale=settings.audio.azure_embedded_asr_locale,
                sample_rate_hz=16000,
                channels=1,
                bits_per_sample=16,
            )
        )
        yield pb.AsrRequest(pcm16=pcm16)
        yield pb.AsrRequest(end=True)

    channel = grpc.insecure_channel(settings.audio.azure_embedded_grpc_url)
    try:
        grpc.channel_ready_future(channel).result(timeout=2.0)
        for event in pb_grpc.AzureEmbeddedSpeechStub(channel).Recognize(requests(), timeout=15):
            event_type = getattr(event, "type", "")
            if event_type == "error":
                raise RuntimeError(getattr(event, "detail", "Azure Embedded ASR warmup failed"))
            if event_type == "final":
                status["warmup"] = {"bytes": len(pcm16), "elapsedMs": getattr(event, "elapsed_ms", 0)}
                return status
    except Exception as exc:
        raise RuntimeError(f"Azure Embedded ASR warmup failed at {settings.audio.azure_embedded_grpc_url}: {_grpc_error_message(exc)}") from exc
    finally:
        channel.close()
    status["warmup"] = {"bytes": len(pcm16)}
    return status


def _pid_for_grpc_url(grpc_url: str) -> int | None:
    host, _, port_text = grpc_url.rpartition(":")
    if not host or not port_text.isdigit():
        return None
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return None
    try:
        import psutil
    except ImportError:
        return None
    port = int(port_text)
    for connection in psutil.net_connections(kind="tcp"):
        local = connection.laddr
        if getattr(local, "port", None) == port and connection.status == psutil.CONN_LISTEN:
            return connection.pid
    return None


def _grpc_error_message(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    if callable(details):
        try:
            return str(details())
        except Exception:
            pass
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, socket.timeout)):
        return str(exc)
    message = str(exc)
    if not message:
        code = getattr(exc, "code", None)
        if callable(code):
            try:
                message = str(code())
            except Exception:
                message = ""
    return message