from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from voice_agent.config import LFM2AudioSettings, Settings
from voice_agent.pipecat_runtime.events import ProgressCallback, emit_progress, progress_event


@dataclass(frozen=True)
class _LFM2AudioComponents:
    processor: Any
    model: Any
    chat_state: Any
    modality: Any
    torch: Any


class LFM2AudioVoiceClient:
    _components_cache: dict[tuple[str, str, bool], _LFM2AudioComponents] = {}

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_settings = settings.lfm2_audio

    def warm(self) -> dict[str, object]:
        cache_key = self._components_cache_key()
        cached = cache_key in self._components_cache
        self._load_components()
        return {
            "provider": "lfm2-audio",
            "model": self.model_settings.model_id,
            "runtime": "liquid-audio",
            "mode": "speech-to-speech",
            "modelDir": str(_model_source(self.model_settings)),
            "cached": cached,
        }

    async def run_audio_turn(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        *,
        progress_callback: ProgressCallback | None = None,
        llm_prompt: str | None = None,
        llm_context: str | None = None,
        vad_provider: str = "browser-vad",
        vad_ms: float = 0.0,
        emit_vad_progress: bool = True,
    ):
        from voice_agent.real_turn import RealTurnResult

        total_started = perf_counter()
        if emit_vad_progress:
            await _emit(progress_callback, "vad", "idle", latency_ms=vad_ms)
        await _emit(progress_callback, "llm", "running")

        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue[tuple[str, str, dict[str, object]]] = asyncio.Queue()
        first_streamed_audio = False

        def stream_audio_chunk(audio_wav: bytes, text: str, latency_ms: float, total_ms: float) -> None:
            nonlocal first_streamed_audio
            if not first_streamed_audio:
                first_streamed_audio = True
                loop.call_soon_threadsafe(progress_queue.put_nowait, ("llm", "ttft", {"latency_ms": latency_ms}))
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                (
                    "tts",
                    "audio",
                    {
                        "latency_ms": 0.0,
                        "total_ms": 0.0,
                        "text": text,
                        "audio_base64": base64.b64encode(audio_wav).decode("ascii"),
                        "audio_media_type": "audio/wav",
                    },
                ),
            )

        worker = asyncio.create_task(
            asyncio.to_thread(
                self._run_audio_turn_sync,
                audio_bytes,
                filename,
                media_type,
                llm_prompt,
                llm_context,
                stream_audio_chunk,
            )
        )
        while not worker.done() or not progress_queue.empty():
            try:
                stage, status, payload = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await _emit(progress_callback, stage, status, **payload)

        user_text, assistant_text, audio_wav, timings = await worker
        audio_base64 = base64.b64encode(audio_wav).decode("ascii")
        if not timings.get("streamedAudioChunks"):
            await _emit(progress_callback, "llm", "ttft", latency_ms=timings["llm"])
        await _emit(progress_callback, "llm", "idle", latency_ms=timings["llm"], total_ms=timings["llmTotal"], text=assistant_text)
        if not timings.get("streamedAudioChunks"):
            await _emit(
                progress_callback,
                "tts",
                "audio",
                latency_ms=0.0,
                total_ms=0.0,
                text=assistant_text,
                audio_base64=audio_base64,
                audio_media_type="audio/wav",
            )
        await _emit(progress_callback, "tts", "idle", latency_ms=0.0, total_ms=0.0)
        return RealTurnResult(
            status="passed",
            user_text=user_text,
            assistant_text=assistant_text,
            vad_provider=vad_provider,
            asr_provider="direct-audio",
            llm_provider="lfm2-audio",
            tts_provider="lfm2-audio",
            audio_media_type="audio/wav",
            audio_base64=audio_base64,
            browser_tts_fallback=False,
            timings_ms={**timings, "vad": vad_ms, "backendTotal": _elapsed_ms(total_started)},
            tts_voice=self.model_settings.model_id,
        )

    async def run_text_turn(
        self,
        user_text: str,
        *,
        progress_callback: ProgressCallback | None = None,
        llm_prompt: str | None = None,
        llm_context: str | None = None,
        vad_provider: str = "browser-vad",
        asr_provider: str = "browser-text",
        vad_ms: float = 0.0,
        asr_ms: float = 0.0,
    ):
        from voice_agent.real_turn import RealTurnResult

        total_started = perf_counter()
        normalized_text = user_text.strip()
        if not normalized_text:
            raise RuntimeError("Text turn requires non-empty user text")
        assistant_text, audio_wav, timings = await asyncio.to_thread(
            self._run_text_turn_sync,
            normalized_text,
            llm_prompt,
            llm_context,
        )
        audio_base64 = base64.b64encode(audio_wav).decode("ascii")
        await _emit(progress_callback, "llm", "idle", latency_ms=timings["llm"], total_ms=timings["llmTotal"], text=assistant_text)
        await _emit(
            progress_callback,
            "tts",
            "audio",
            latency_ms=timings["tts"],
            total_ms=timings["ttsTotal"],
            text=assistant_text,
            audio_base64=audio_base64,
            audio_media_type="audio/wav",
        )
        await _emit(progress_callback, "tts", "idle", latency_ms=timings["tts"], total_ms=timings["ttsTotal"])
        return RealTurnResult(
            status="passed",
            user_text=normalized_text,
            assistant_text=assistant_text,
            vad_provider=vad_provider,
            asr_provider=asr_provider,
            llm_provider="lfm2-audio",
            tts_provider="lfm2-audio",
            audio_media_type="audio/wav",
            audio_base64=audio_base64,
            browser_tts_fallback=False,
            timings_ms={**timings, "vad": vad_ms, "asr": asr_ms, "backendTotal": _elapsed_ms(total_started)},
            tts_voice=self.model_settings.model_id,
        )

    def _run_audio_turn_sync(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        llm_prompt: str | None,
        llm_context: str | None,
        stream_audio_chunk: Callable[[bytes, str, float, float], None] | None = None,
    ):
        components = self._load_components()
        waveform, sample_rate = _decode_audio_bytes(audio_bytes, filename, media_type, components.torch)
        assistant_text, audio_wav, response_timings = self._generate_interleaved_response(
            components,
            audio_waveform=waveform,
            sample_rate=sample_rate,
            user_text=None,
            llm_prompt=llm_prompt,
            llm_context=llm_context,
            stream_audio_chunk=stream_audio_chunk,
        )
        return "Audio input", assistant_text, audio_wav, {"asr": 0.0, **response_timings}

    def _run_text_turn_sync(self, user_text: str, llm_prompt: str | None, llm_context: str | None):
        components = self._load_components()
        return self._generate_interleaved_response(
            components,
            audio_waveform=None,
            sample_rate=None,
            user_text=user_text,
            llm_prompt=llm_prompt,
            llm_context=llm_context,
        )

    def _generate_asr_text(self, components: _LFM2AudioComponents, waveform, sample_rate: int) -> str:
        chat = components.chat_state(components.processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(waveform, sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        text_parts: list[str] = []
        with components.torch.inference_mode():
            for token in components.model.generate_sequential(**chat, max_new_tokens=self.model_settings.max_new_tokens):
                if token.numel() == 1:
                    text_parts.append(components.processor.text.decode(token))
        return "".join(text_parts)

    def _generate_interleaved_response(
        self,
        components: _LFM2AudioComponents,
        *,
        audio_waveform,
        sample_rate: int | None,
        user_text: str | None,
        llm_prompt: str | None,
        llm_context: str | None,
        stream_audio_chunk: Callable[[bytes, str, float, float], None] | None = None,
    ):
        chat = components.chat_state(components.processor)
        chat.new_turn("system")
        chat.add_text(_system_prompt(llm_prompt, llm_context))
        chat.end_turn()
        chat.new_turn("user")
        if audio_waveform is not None and sample_rate is not None:
            chat.add_audio(audio_waveform, sample_rate)
        else:
            chat.add_text(user_text or "")
        chat.end_turn()
        chat.new_turn("assistant")

        text_tokens = []
        audio_tokens = []
        pending_audio_tokens = []
        text_parts: list[str] = []
        started = perf_counter()
        first_text_ms: float | None = None
        first_audio_ms: float | None = None
        streamed_audio_chunks = 0
        with components.torch.inference_mode():
            for token in components.model.generate_interleaved(
                **chat,
                max_new_tokens=self.model_settings.max_new_tokens,
                audio_temperature=self.model_settings.audio_temperature,
                audio_top_k=self.model_settings.audio_top_k,
            ):
                if token.numel() == 1:
                    if first_text_ms is None:
                        first_text_ms = _elapsed_ms(started)
                    text_tokens.append(token)
                    text_parts.append(components.processor.text.decode(token))
                else:
                    is_eos = _is_audio_eos_token(components, token)
                    if first_audio_ms is None:
                        first_audio_ms = _elapsed_ms(started)
                        if stream_audio_chunk is not None and not is_eos:
                            stream_audio_chunk(_decode_audio_output(components, [token]), _clean_generated_text("".join(text_parts)), first_audio_ms, first_audio_ms)
                            streamed_audio_chunks += 1
                    audio_tokens.append(token)
                    if not is_eos and not (streamed_audio_chunks == 1 and len(audio_tokens) == 1):
                        pending_audio_tokens.append(token)
                    if len(pending_audio_tokens) >= 24:
                        if stream_audio_chunk is not None:
                            stream_audio_chunk(_decode_audio_output(components, pending_audio_tokens), _clean_generated_text("".join(text_parts)), first_audio_ms or _elapsed_ms(started), _elapsed_ms(started))
                            streamed_audio_chunks += 1
                        pending_audio_tokens = []
        llm_total_ms = _elapsed_ms(started)
        assistant_text = _clean_generated_text("".join(text_parts))
        if not assistant_text:
            raise RuntimeError("LFM2.5 Audio returned empty assistant text")
        if pending_audio_tokens and stream_audio_chunk is not None:
            stream_audio_chunk(_decode_audio_output(components, pending_audio_tokens), assistant_text, first_audio_ms or llm_total_ms, llm_total_ms)
            streamed_audio_chunks += 1
        audio_wav = _decode_audio_output(components, audio_tokens)
        tts_total_ms = 0.0
        return assistant_text, audio_wav, {
            "llm": first_audio_ms or first_text_ms or llm_total_ms,
            "llmFirstSentence": first_audio_ms or first_text_ms or llm_total_ms,
            "llmTotal": llm_total_ms,
            "tts": 0.0,
            "ttsTotal": tts_total_ms,
            "streamedAudioChunks": streamed_audio_chunks,
        }

    def _load_components(self) -> _LFM2AudioComponents:
        model_source = _model_source(self.model_settings)
        key = self._components_cache_key(model_source)
        cached = self._components_cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality
        except ImportError as exc:
            raise RuntimeError('Install LFM2 Audio dependencies with `python -m pip install -e ".[lfm2-audio]"`') from exc
        if self.model_settings.torch_threads > 0:
            torch.set_num_threads(self.model_settings.torch_threads)
        processor = LFM2AudioProcessor.from_pretrained(model_source, device="cpu").eval()
        _ensure_cpu_audio_detokenizer(processor)
        model = LFM2AudioModel.from_pretrained(model_source, device="cpu").eval()
        components = _LFM2AudioComponents(processor=processor, model=model, chat_state=ChatState, modality=LFMModality, torch=torch)
        self._components_cache[key] = components
        return components

    def _components_cache_key(self, model_source: str | Path | None = None) -> tuple[str, str, bool]:
        if model_source is None:
            model_source = _model_source(self.model_settings)
        return (self.model_settings.model_id, str(model_source), self.model_settings.allow_download)


def _ensure_cpu_audio_detokenizer(processor) -> None:
    detokenizer_path = getattr(processor, "detokenizer_path", None)
    if not detokenizer_path or getattr(processor, "_audio_detokenizer", None) is not None:
        return
    from pathlib import Path

    from liquid_audio.detokenizer import LFM2AudioDetokenizer
    from safetensors.torch import load_file
    from transformers import Lfm2Config

    detokenizer_dir = Path(detokenizer_path)
    detokenizer_config = Lfm2Config.from_pretrained(detokenizer_dir / "config.json")
    if isinstance(detokenizer_config.layer_types, list):
        detokenizer_config.layer_types = ["full_attention" if layer == "sliding_attention" else layer for layer in detokenizer_config.layer_types]
    detokenizer = LFM2AudioDetokenizer(detokenizer_config).eval().to(device=processor.device)
    detokenizer_weights = load_file(detokenizer_dir / "model.safetensors", device=str(processor.device))
    detokenizer.load_state_dict(detokenizer_weights)
    processor._audio_detokenizer = detokenizer.eval()


def _model_source(settings: LFM2AudioSettings) -> str | Path:
    if settings.allow_download:
        return settings.model_id
    if not settings.model_dir.exists():
        raise RuntimeError(
            "LFM2.5 Audio model files are not available. Set VOICE_AGENT_LFM2_AUDIO_MODEL_DIR to a local snapshot "
            "or set VOICE_AGENT_LFM2_AUDIO_ALLOW_DOWNLOAD=true."
        )
    return settings.model_dir


def _decode_audio_bytes(audio_bytes: bytes, filename: str, media_type: str, torch):
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if samples.shape[1] > 1:
            samples = samples.mean(axis=1, keepdims=True)
        return torch.from_numpy(samples[:, 0]).unsqueeze(0), int(sample_rate)
    except Exception as soundfile_error:
        try:
            return _decode_audio_with_av(audio_bytes, torch)
        except Exception as av_error:
            raise RuntimeError(
                f"LFM2.5 Audio could not decode {media_type or filename}; install PyAV/ffmpeg support or send WAV audio. "
                f"soundfile error: {soundfile_error}; av error: {av_error}"
            ) from av_error


def _decode_audio_with_av(audio_bytes: bytes, torch):
    import av
    import numpy as np

    container = av.open(BytesIO(audio_bytes))
    chunks = []
    sample_rate = None
    for frame in container.decode(audio=0):
        sample_rate = int(frame.sample_rate)
        array = frame.to_ndarray()
        source_dtype = array.dtype
        array = array.astype("float32")
        if np.issubdtype(source_dtype, np.integer):
            array /= float(np.iinfo(source_dtype).max)
        if array.ndim == 2:
            array = array.mean(axis=0 if array.shape[0] <= array.shape[1] else 1)
        chunks.append(array.reshape(-1))
    if not chunks or sample_rate is None:
        raise RuntimeError("No audio frames were decoded")
    samples = np.concatenate(chunks).astype("float32")
    return torch.from_numpy(samples).unsqueeze(0), sample_rate


def _decode_audio_output(components: _LFM2AudioComponents, audio_tokens: list[Any]) -> bytes:
    audio_frames = list(audio_tokens)
    while audio_frames and bool(components.torch.all(audio_frames[-1] == 2048).item()):
        audio_frames.pop()
    if not audio_frames:
        raise RuntimeError("LFM2.5 Audio returned no audio tokens")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install soundfile to encode LFM2.5 Audio output") from exc
    audio_codes = components.torch.stack(audio_frames, 1).unsqueeze(0)
    waveform = components.processor.decode(audio_codes)
    samples = waveform.detach().cpu()
    if samples.ndim == 2:
        samples = samples[0]
    elif samples.ndim == 3:
        samples = samples[0, 0]
    output = BytesIO()
    sf.write(output, samples.float().numpy(), 24_000, format="WAV")
    return output.getvalue()


def _is_audio_eos_token(components: _LFM2AudioComponents, token: Any) -> bool:
    return bool(components.torch.all(token == 2048).item())


def _system_prompt(llm_prompt: str | None, llm_context: str | None) -> str:
    parts = []
    prompt = (llm_prompt or "").strip()
    if prompt:
        parts.append(prompt)
    parts.append("Respond with interleaved text and audio.")
    context = (llm_context or "").strip()
    if context:
        parts.append(f"Context:\n{context}")
    return "\n".join(parts)


async def _emit(progress_callback: ProgressCallback | None, stage: str, status: str, **payload: object) -> None:
    await emit_progress(
        progress_callback,
        progress_event(
            stage,
            status,
            text=_optional_str(payload.get("text")),
            voice=_optional_str(payload.get("voice")),
            latency_ms=_optional_float(payload.get("latency_ms")),
            total_ms=_optional_float(payload.get("total_ms")),
            audio_base64=_optional_str(payload.get("audio_base64")),
            audio_media_type=_optional_str(payload.get("audio_media_type")),
        ),
    )


def _clean_generated_text(text: str) -> str:
    cleaned = text
    for token in ("<|im_end|>", "<|text_end|>", "<|audio_start|>", "<|audio_end|>", "<|endoftext|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 1)
