from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from voice_agent.config import Settings
from voice_agent.pipecat_runtime.events import ProgressCallback
from voice_agent.providers.asr import ASRTranscript, AzureEmbeddedASR, FasterWhisperASR, FoundryLocalASR
from voice_agent.providers.llm_foundry import FoundryLocalLLM
from voice_agent.providers.llm_llama_cpp import LlamaCppLLM
from voice_agent.providers.vad import SileroVad, VadDecision
from voice_agent.real_turn import (
    AzureEmbeddedTTSClient,
    EdgeTTSClient,
    PreparedLLMTurn,
    RealTurnResult,
    SapiTTSClient,
    prepare_llm_turn,
    run_text_turn,
)


class ASRClient(Protocol):
    async def transcribe_audio(self, audio_bytes: bytes, filename: str, media_type: str, language: str = "auto") -> ASRTranscript:
        ...


class VADClient(Protocol):
    async def analyze_audio(self, audio_bytes: bytes, filename: str, media_type: str) -> VadDecision:
        ...


@dataclass(frozen=True)
class PurePipecatTurnContext:
    settings: Settings
    audio_bytes: bytes | None = None
    filename: str = "recording.webm"
    media_type: str = "audio/webm"
    user_text: str | None = None
    vad_provider: str = "browser-vad"
    asr_provider: str = "azure-embedded"
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    llm_prompt: str | None = None
    llm_context: str | None = None
    asr_client: ASRClient | None = None
    vad_client: VADClient | None = None
    llm_client: object | None = None
    tts_client: object | None = None
    prepared_llm_turn: PreparedLLMTurn | None = None
    progress_callback: ProgressCallback | None = None


@dataclass
class PurePipecatTurnState:
    vad_decision: VadDecision | None = None
    transcript: ASRTranscript | None = None
    prepared_llm_turn: PreparedLLMTurn | None = None
    result: RealTurnResult | None = None


class PurePipecatRuntimeError(RuntimeError):
    pass


async def run_pure_pipecat_audio_turn(
    settings: Settings,
    audio_bytes: bytes,
    filename: str,
    media_type: str,
    *,
    asr_client: ASRClient | None = None,
    vad_client: VADClient | None = None,
    llm_client: object | None = None,
    tts_client: object | None = None,
    prepared_llm_turn: PreparedLLMTurn | None = None,
    progress_callback: ProgressCallback | None = None,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
) -> RealTurnResult:
    context = PurePipecatTurnContext(
        settings=settings,
        audio_bytes=audio_bytes,
        filename=filename,
        media_type=media_type,
        llm_prompt=llm_prompt,
        llm_context=llm_context,
        asr_client=asr_client,
        vad_client=vad_client,
        llm_client=llm_client,
        tts_client=tts_client,
        prepared_llm_turn=prepared_llm_turn,
        progress_callback=progress_callback,
    )
    return await _run_pipecat_context(context)


async def run_pure_pipecat_text_turn(
    settings: Settings,
    user_text: str,
    *,
    llm_client: object | None = None,
    tts_client: object | None = None,
    progress_callback: ProgressCallback | None = None,
    vad_provider: str = "browser-vad",
    asr_provider: str = "azure-embedded",
    vad_ms: float = 0.0,
    asr_ms: float = 0.0,
    llm_prompt: str | None = None,
    llm_context: str | None = None,
) -> RealTurnResult:
    context = PurePipecatTurnContext(
        settings=settings,
        user_text=user_text,
        vad_provider=vad_provider,
        asr_provider=asr_provider,
        vad_ms=vad_ms,
        asr_ms=asr_ms,
        llm_prompt=llm_prompt,
        llm_context=llm_context,
        llm_client=llm_client,
        tts_client=tts_client,
        progress_callback=progress_callback,
    )
    return await _run_pipecat_context(context)


async def _run_pipecat_context(context: PurePipecatTurnContext) -> RealTurnResult:
    try:
        from pipecat.frames.frames import EndFrame, TextFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.workers.runner import WorkerRunner
    except ImportError as exc:
        raise PurePipecatRuntimeError("Install pipecat-ai to run the pure Pipecat fork") from exc

    state = PurePipecatTurnState()
    result_ready = asyncio.get_running_loop().create_future()

    class VadAsrProcessor(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if not isinstance(frame, TextFrame):
                await self.push_frame(frame, direction)
                return
            if context.user_text is not None:
                await self.push_frame(frame, direction)
                return
            if context.audio_bytes is None or len(context.audio_bytes) < 128:
                raise RuntimeError("Recorded audio is too small to transcribe")
            vad_started = perf_counter()
            await _emit(context.progress_callback, "vad", "running")
            vad = context.vad_client or SileroVad()
            state.vad_decision = await vad.analyze_audio(context.audio_bytes, context.filename, context.media_type)
            vad_ms = _elapsed_ms(vad_started)
            if not state.vad_decision.is_speech:
                await _emit(context.progress_callback, "vad", "failed", latency_ms=vad_ms)
                raise RuntimeError("Silero VAD detected no speech")
            await _emit(context.progress_callback, "vad", "idle", latency_ms=vad_ms)

            asr_started = perf_counter()
            await _emit(context.progress_callback, "asr", "running")
            asr = context.asr_client or _default_asr_client(context.settings)
            state.transcript = await _transcribe_with_local_fallback(context.settings, asr, context.audio_bytes, context.filename, context.media_type)
            asr_ms = _elapsed_ms(asr_started)
            if not state.transcript.text.strip():
                await _emit(context.progress_callback, "asr", "failed", latency_ms=asr_ms)
                raise RuntimeError("ASR returned empty text")
            await _emit(context.progress_callback, "asr", "idle", latency_ms=asr_ms, text=state.transcript.text)
            await self.push_frame(TextFrame(text=state.transcript.text), direction)

    class PromptProcessor(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if not isinstance(frame, TextFrame):
                await self.push_frame(frame, direction)
                return
            if context.prepared_llm_turn is not None:
                state.prepared_llm_turn = context.prepared_llm_turn
            else:
                llm = context.llm_client or _default_llm_client(context.settings)
                state.prepared_llm_turn = await prepare_llm_turn(llm, llm_prompt=context.llm_prompt, llm_context=context.llm_context)
            await self.push_frame(frame, direction)

    class AssistantProcessor(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if not isinstance(frame, TextFrame) or not frame.text.strip():
                await self.push_frame(frame, direction)
                return
            tts = context.tts_client or _default_tts_client(context.settings)
            owns_tts = context.tts_client is None
            try:
                if context.user_text is not None:
                    state.result = await run_text_turn(
                        context.settings,
                        frame.text,
                        llm_client=context.llm_client,
                        tts_client=tts,
                        progress_callback=context.progress_callback,
                        vad_provider=context.vad_provider,
                        asr_provider=context.asr_provider,
                        vad_ms=context.vad_ms,
                        asr_ms=context.asr_ms,
                        llm_prompt=context.llm_prompt,
                        llm_context=context.llm_context,
                        prepared_llm_turn=state.prepared_llm_turn,
                    )
                else:
                    state.result = await _complete_audio_assistant_turn(context, state, tts)
            finally:
                if owns_tts and hasattr(tts, "close"):
                    close_result = tts.close()
                    if inspect.isawaitable(close_result):
                        await close_result
            if not result_ready.done():
                result_ready.set_result(None)
            await self.push_frame(frame, direction)

    pipeline = Pipeline([VadAsrProcessor(), PromptProcessor(), AssistantProcessor()])
    task = PipelineWorker(pipeline, idle_timeout_secs=None, enable_rtvi=False, enable_turn_tracking=False)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    runner_task = asyncio.create_task(runner.run())
    try:
        await task.queue_frame(TextFrame(text=context.user_text or ""))
        done, _ = await asyncio.wait({result_ready, runner_task}, return_when=asyncio.FIRST_COMPLETED)
        if runner_task in done:
            await runner_task
        await result_ready
        if state.result is None:
            raise RuntimeError("Pure Pipecat pipeline finished without a turn result")
        await task.queue_frame(EndFrame())
        await runner_task
        return state.result
    finally:
        if not runner_task.done():
            await task.cancel(reason="pure Pipecat turn cleanup")
            await runner_task


async def _complete_audio_assistant_turn(context: PurePipecatTurnContext, state: PurePipecatTurnState, tts: object) -> RealTurnResult:
    if state.transcript is None or state.vad_decision is None:
        raise RuntimeError("Pure Pipecat audio turn reached LLM/TTS before ASR completed")
    return await run_text_turn(
        context.settings,
        state.transcript.text,
        llm_client=context.llm_client,
        tts_client=tts,
        progress_callback=context.progress_callback,
        vad_provider=state.vad_decision.provider,
        asr_provider=state.transcript.provider,
        vad_ms=0.0,
        asr_ms=0.0,
        llm_prompt=context.llm_prompt,
        llm_context=context.llm_context,
        prepared_llm_turn=state.prepared_llm_turn,
    )


async def _emit(progress_callback: ProgressCallback | None, stage: str, status: str, **payload: object) -> None:
    if progress_callback is not None:
        await progress_callback({"stage": stage, "status": status, **payload})


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 1)


def _default_asr_client(settings: Settings):
    if settings.providers.asr == "foundry-local":
        return FoundryLocalASR(settings.foundry)
    if settings.providers.asr == "faster-whisper":
        return FasterWhisperASR(settings.audio.faster_whisper_model)
    if settings.providers.asr == "azure-embedded":
        return AzureEmbeddedASR(settings.audio)
    raise RuntimeError(f"Pure Pipecat ASR provider is not implemented: {settings.providers.asr}")


def _default_llm_client(settings: Settings):
    if settings.providers.llm == "llama-cpp":
        return LlamaCppLLM(settings.llama_cpp)
    if settings.providers.llm == "foundry-local":
        return FoundryLocalLLM(settings.foundry)
    raise RuntimeError(f"Pure Pipecat LLM provider is not implemented: {settings.providers.llm}")


def _default_tts_client(settings: Settings):
    if settings.providers.tts == "edge-tts":
        return EdgeTTSClient(settings.audio.edge_tts_voice)
    if settings.providers.tts == "azure-embedded":
        return AzureEmbeddedTTSClient(settings)
    return SapiTTSClient(settings.audio.windows_tts_voice)


async def _transcribe_with_local_fallback(settings: Settings, asr, audio_bytes: bytes, filename: str, media_type: str) -> ASRTranscript:
    try:
        return await asr.transcribe_audio(audio_bytes, filename=filename, media_type=media_type or "application/octet-stream", language=settings.audio.asr_language)
    except Exception:
        if settings.providers.asr != "azure-embedded":
            raise
        fallback = FasterWhisperASR(settings.audio.faster_whisper_model)
        return await fallback.transcribe_audio(audio_bytes, filename=filename, media_type=media_type or "application/octet-stream", language=settings.audio.asr_language)
