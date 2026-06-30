from __future__ import annotations

import base64
import inspect
import asyncio
from time import perf_counter
from dataclasses import dataclass
from typing import Protocol

from voice_agent.config import Settings
from voice_agent.pipecat_runtime.events import ProgressCallback, emit_progress, progress_event
from voice_agent.providers.asr import ASRTranscript, AzureEmbeddedASR, FasterWhisperASR, FoundryLocalASR, warm_faster_whisper, warm_foundry_streaming_asr
from voice_agent.providers.llm_foundry import ChatMessage, FoundryLocalLLM
from voice_agent.providers.tts_windows import synthesize_azure_embedded_wav, synthesize_edge_mp3, synthesize_sapi_wav
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


class TTSClient(Protocol):
    def synthesize(self, text: str) -> bytes:
        ...


class VADClient(Protocol):
    async def analyze_audio(self, audio_bytes: bytes, filename: str, media_type: str) -> VadDecision:
        ...


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


@dataclass(frozen=True)
class AzureEmbeddedTTSClient:
    settings: Settings

    def synthesize(self, text: str) -> bytes:
        return synthesize_azure_embedded_wav(text, self.settings.audio)


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
) -> RealTurnResult:
    total_started = perf_counter()
    if len(audio_bytes) < 128:
        raise RuntimeError("Recorded audio is too small to transcribe")

    asr = asr_client or _default_asr_client(settings)
    llm = llm_client or FoundryLocalLLM(settings.foundry)
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

    asr_started = perf_counter()
    await _emit_progress(progress_callback, "asr", "running")
    transcript = await _transcribe_with_local_fallback(settings, asr, audio_bytes, filename, media_type)
    asr_ms = _elapsed_ms(asr_started)
    if not transcript.text.strip():
        await _emit_progress(progress_callback, "asr", "failed", latency_ms=asr_ms)
        raise RuntimeError("ASR returned empty text")
    await _emit_progress(progress_callback, "asr", "idle", latency_ms=asr_ms, text=transcript.text)

    assistant_turn = await _complete_assistant_turn(settings, transcript.text, llm, tts, progress_callback)

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
) -> RealTurnResult:
    total_started = perf_counter()
    normalized_text = user_text.strip()
    if not normalized_text:
        raise RuntimeError("Text turn requires non-empty user text")

    llm = llm_client or FoundryLocalLLM(settings.foundry)
    tts = tts_client or _default_tts_client(settings)
    assistant_turn = await _complete_assistant_turn(settings, normalized_text, llm, tts, progress_callback)
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


async def _synthesize_tts(tts, text: str) -> tuple[bytes, str]:
    if isinstance(tts, EdgeTTSClient):
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
) -> AssistantTurnPayload:
    messages = [
        ChatMessage(
            "system",
            "You are a concise local voice assistant. Reply in the user's language when possible. Do not ask for credentials, tokens, session IDs, or other secrets.",
        ),
        ChatMessage("user", user_text),
    ]
    await _emit_progress(progress_callback, "llm", "running")
    assistant_text, llm_ms, llm_first_sentence_ms, llm_total_ms, tts_payload = await _run_llm_and_tts(
        llm,
        tts,
        messages,
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
    llm,
    tts,
    messages: list[ChatMessage],
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, float, float, float, tuple[bytes, str, float, float]]:
    if hasattr(llm, "stream"):
        try:
            return await _stream_llm_and_tts(llm, tts, messages, progress_callback)
        except Exception:
            pass

    llm_started = perf_counter()
    assistant_text = await llm.complete(messages)
    llm_ms = _elapsed_ms(llm_started)
    await _emit_progress(progress_callback, "llm", "ttft", latency_ms=llm_ms)
    await _emit_progress(progress_callback, "llm", "first_sentence", latency_ms=llm_ms)
    tts_started = perf_counter()
    await _emit_progress(progress_callback, "tts", "running")
    audio_bytes, audio_media_type = await _synthesize_tts(tts, assistant_text)
    tts_ms = _elapsed_ms(tts_started)
    return assistant_text, llm_ms, llm_ms, llm_ms, (audio_bytes, audio_media_type, tts_ms, tts_ms)


async def _stream_llm_and_tts(
    llm,
    tts,
    messages: list[ChatMessage],
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, float, float, float, tuple[bytes, str, float, float]]:
    llm_started = perf_counter()
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    assistant_parts: list[str] = []
    pending_text = ""
    tts_tasks: list[asyncio.Task[tuple[bytes, str, float]]] = []

    async for token in llm.stream(messages):
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


async def warm_real_chain(settings: Settings) -> dict[str, object]:
    timings: dict[str, float] = {}
    statuses: dict[str, object] = {}

    started = perf_counter()
    warm_silero_vad()
    timings["vad"] = _elapsed_ms(started)
    statuses["vad"] = {"provider": "silero"}

    started = perf_counter()
    ready = await check_foundry_ready(settings)
    if not ready.get("ready"):
        raise RuntimeError(f"Foundry Local is not ready: {ready.get('error')}")

    asr_started = perf_counter()
    if settings.providers.asr == "foundry-local":
        asr_status = await asyncio.to_thread(warm_foundry_streaming_asr, settings.foundry.asr_model)
        timings["asr"] = _elapsed_ms(asr_started)
        statuses["asr"] = {**asr_status, "language": settings.audio.asr_language}
    elif settings.providers.asr == "azure-embedded":
        timings["asr"] = _elapsed_ms(asr_started)
        statuses["asr"] = {
            "provider": "azure-embedded",
            "locale": settings.audio.azure_embedded_asr_locale,
        }
    else:
        warm_faster_whisper(settings.audio.faster_whisper_model, "cpu", "int8")
        timings["asr"] = _elapsed_ms(asr_started)
        statuses["asr"] = {"provider": "faster-whisper", "model": settings.audio.faster_whisper_model}

    await FoundryLocalLLM(settings.foundry).complete(
        [ChatMessage("system", "Reply with OK only."), ChatMessage("user", "OK?")]
    )
    timings["llm"] = _elapsed_ms(started)
    statuses["llm"] = {"provider": "foundry-local", "model": settings.foundry.llm_model}

    started = perf_counter()
    await _synthesize_tts(_default_tts_client(settings), "Ready.")
    timings["tts"] = _elapsed_ms(started)
    tts_voice = settings.audio.edge_tts_voice
    if settings.providers.tts == "azure-embedded":
        tts_voice = settings.audio.azure_embedded_tts_voice
    statuses["tts"] = {"provider": settings.providers.tts, "voice": tts_voice}

    return {"status": "ready", "timingsMs": timings, "models": statuses}