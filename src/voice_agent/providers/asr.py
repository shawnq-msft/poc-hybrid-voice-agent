from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from voice_agent.config import AudioSettings, FOUNDRY_STREAMING_ASR_MODELS, LOCAL_ASR_PROVIDERS, FoundrySettings


def validate_asr_provider(provider: str) -> str:
    if provider not in LOCAL_ASR_PROVIDERS:
        raise ValueError(f"Unsupported ASR provider: {provider}")
    return provider


@dataclass(frozen=True)
class ASRTranscript:
    text: str
    language: str | None
    provider: str


@dataclass(frozen=True)
class FoundryLocalASR:
    settings: FoundrySettings

    @property
    def transcription_url(self) -> str:
        return f"{self.settings.endpoint.rstrip('/')}/audio/transcriptions"

    async def transcribe_wav(self, wav_bytes: bytes, language: str = "auto") -> ASRTranscript:
        return await self.transcribe_audio(
            wav_bytes,
            filename="audio.wav",
            media_type="audio/wav",
            language=language,
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        language: str = "auto",
    ) -> ASRTranscript:
        if not audio_bytes:
            raise RuntimeError("Cannot transcribe empty audio")
        pcm16 = await asyncio.to_thread(_decode_audio_to_pcm16, audio_bytes, filename, 16000)
        text = await asyncio.to_thread(_transcribe_pcm16_with_foundry_live_session, self.settings.asr_model, pcm16, language)
        if not text:
            raise RuntimeError("Foundry Local streaming ASR returned empty text")
        return ASRTranscript(text=text, language=language, provider="foundry-local")


@dataclass(frozen=True)
class FasterWhisperASR:
    model_name: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        language: str = "auto",
    ) -> ASRTranscript:
        if not audio_bytes:
            raise RuntimeError("Cannot transcribe empty audio")
        suffix = Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)
        try:
            model = _get_faster_whisper_model(self.model_name, self.device, self.compute_type)
            language_arg = None if language == "auto" else language
            segments, info = model.transcribe(str(temp_path), language=language_arg, vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        finally:
            temp_path.unlink(missing_ok=True)
        if not text:
            raise RuntimeError("faster-whisper returned empty text")
        return ASRTranscript(text=text, language=getattr(info, "language", None), provider="faster-whisper")


@dataclass(frozen=True)
class AzureEmbeddedASR:
    audio_settings: AudioSettings

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        media_type: str,
        language: str = "auto",
    ) -> ASRTranscript:
        self.validate_assets()
        locale = language if language not in {"auto", ""} else self.audio_settings.azure_embedded_asr_locale
        pcm16 = _decode_audio_to_pcm16(audio_bytes, filename, 16000)
        try:
            final_text = await asyncio.to_thread(
                _transcribe_pcm16_with_azure_embedded_grpc,
                self.audio_settings.azure_embedded_grpc_url,
                pcm16,
                locale,
            )
        except Exception:
            final_text = await _transcribe_pcm16_with_azure_embedded_websocket(
                self.audio_settings.azure_embedded_asr_sidecar_url,
                pcm16,
                locale,
            )
        if not final_text:
            raise RuntimeError(
                "AzureEmbeddedSpeech sidecar returned empty text. "
                "Confirm the C++ gRPC sidecar is running and linked against Azure Speech SDK 1.47."
            )
        return ASRTranscript(final_text, locale, "azure-embedded")

    def validate_assets(self) -> None:
        if not self.audio_settings.pasco_model_key:
            raise RuntimeError("PASCO_MODEL_KEY is not configured")
        model_dir = self.model_dir_for_locale(self.audio_settings.azure_embedded_asr_locale)
        required = ["sr.ini", "model_onnx.config", "tokens.list"]
        missing = [name for name in required if not (model_dir / name).exists()]
        onnx_files = list(model_dir.glob("*.onnx"))
        if missing or len(onnx_files) < 4:
            raise RuntimeError(
                f"Azure Embedded ASR model assets are incomplete in {model_dir}: "
                f"missing={missing}, onnx_count={len(onnx_files)}"
            )

    def model_dir_for_locale(self, locale: str) -> Path:
        if locale.lower() == "en-gb":
            return self.audio_settings.azure_embedded_asr_en_gb_model_dir
        return self.audio_settings.azure_embedded_asr_zh_cn_model_dir


def _transcribe_pcm16_with_azure_embedded_grpc(grpc_url: str, pcm16: bytes, locale: str) -> str:
    try:
        import grpc
        from voice_agent.providers import azure_embedded_pb2 as pb
        from voice_agent.providers import azure_embedded_pb2_grpc as pb_grpc
    except ImportError as exc:
        raise RuntimeError("Install grpcio and generate Azure Embedded gRPC stubs to use azure-embedded ASR") from exc

    def requests():
        yield pb.AsrRequest(
            config=pb.AsrConfig(
                locale=locale,
                sample_rate_hz=16000,
                channels=1,
                bits_per_sample=16,
            )
        )
        chunk_size = 3200
        for offset in range(0, len(pcm16), chunk_size):
            yield pb.AsrRequest(pcm16=pcm16[offset : offset + chunk_size])
        yield pb.AsrRequest(end=True)

    channel = grpc.insecure_channel(grpc_url)
    try:
        grpc.channel_ready_future(channel).result(timeout=1.5)
    except Exception as exc:
        channel.close()
        raise RuntimeError(f"Azure Embedded gRPC sidecar is not reachable at {grpc_url}") from exc
    stub = pb_grpc.AzureEmbeddedSpeechStub(channel)
    final_text = ""
    try:
        for event in stub.Recognize(requests(), timeout=15):
            event_type = getattr(event, "type", "")
            if event_type == "error":
                raise RuntimeError(getattr(event, "detail", "Azure Embedded ASR failed"))
            if event_type == "final":
                final_text = getattr(event, "text", "").strip()
                break
    finally:
        channel.close()
    return final_text


async def _transcribe_pcm16_with_azure_embedded_websocket(sidecar_url: str, pcm16: bytes, locale: str) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install websockets or run the Azure Embedded gRPC sidecar to use azure-embedded ASR") from exc

    async with websockets.connect(sidecar_url, max_size=16 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"type": "start", "locale": locale}))
        await websocket.send(pcm16)
        await websocket.send(json.dumps({"type": "end"}))
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                continue
            payload = json.loads(message)
            if payload.get("type") == "error":
                raise RuntimeError(str(payload.get("message") or payload.get("detail") or "Azure Embedded ASR sidecar failed"))
            if payload.get("type") == "final":
                return str(payload.get("text") or "").strip()


def _decode_audio_to_pcm16(audio_bytes: bytes, filename: str, sampling_rate: int) -> bytes:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install PyAV to decode audio for AzureEmbeddedSpeech sidecar") from exc

    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)
    try:
        chunks: list[bytes] = []
        with av.open(str(temp_path)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1).tobytes())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1).tobytes())
    finally:
        temp_path.unlink(missing_ok=True)
    if not chunks:
        raise RuntimeError("No audio frames decoded for AzureEmbeddedSpeech sidecar")
    return b"".join(chunks)


_foundry_sdk_lock = threading.Lock()
_foundry_sdk_initialized = False


def warm_foundry_streaming_asr(model_alias: str) -> dict[str, object]:
    model = _get_foundry_streaming_asr_model(model_alias)
    return {
        "provider": "foundry-local",
        "model": model_alias,
        "resolvedModelId": getattr(model, "id", model_alias),
        "cached": bool(getattr(model, "is_cached", False)),
        "loaded": bool(getattr(model, "is_loaded", False)),
    }


def _transcribe_pcm16_with_foundry_live_session(model_alias: str, pcm16: bytes, language: str) -> str:
    if not pcm16:
        raise RuntimeError("Cannot transcribe empty PCM audio")
    model = _get_foundry_streaming_asr_model(model_alias)
    audio_client = model.get_audio_client()
    session = audio_client.create_live_transcription_session()
    session.settings.sample_rate = 16000
    session.settings.channels = 1
    session.settings.bits_per_sample = 16
    session.settings.language = language or "auto"

    final_parts: list[str] = []
    partial_text = ""
    errors: list[BaseException] = []

    def read_results() -> None:
        nonlocal partial_text
        try:
            for result in session.get_stream():
                content = getattr(result, "content", None) or []
                text = getattr(content[0], "text", "") if content else ""
                text = text.strip() if isinstance(text, str) else ""
                if not text:
                    continue
                if getattr(result, "is_final", False):
                    final_parts.append(text)
                else:
                    partial_text = text
        except BaseException as exc:
            errors.append(exc)

    session.start()
    read_thread = threading.Thread(target=read_results, daemon=True)
    read_thread.start()
    try:
        chunk_size = 3200
        for offset in range(0, len(pcm16), chunk_size):
            session.append(pcm16[offset : offset + chunk_size])
    finally:
        session.stop()
        read_thread.join(timeout=5)

    if errors:
        raise RuntimeError(f"Foundry Local streaming ASR failed: {errors[0]}") from errors[0]
    return " ".join(final_parts).strip() or partial_text.strip()


def _get_foundry_streaming_asr_model(model_alias: str):
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError as exc:
        raise RuntimeError("Install foundry-local-sdk-winml to use Foundry Local streaming ASR") from exc

    global _foundry_sdk_initialized
    with _foundry_sdk_lock:
        if not _foundry_sdk_initialized:
            FoundryLocalManager.initialize(Configuration(app_name="hybrid_voice_agent"))
            FoundryLocalManager.instance.download_and_register_eps()
            _foundry_sdk_initialized = True
        manager = FoundryLocalManager.instance
        model_id = _resolve_foundry_streaming_asr_model_id(model_alias)
        if model_id == model_alias:
            model = manager.catalog.get_model(model_alias)
        else:
            model = manager.catalog.get_model_variant(model_id)
        if model is None:
            raise RuntimeError(f'Foundry Local streaming ASR model "{model_alias}" was not found in the catalog')
        if not getattr(model, "is_cached", False):
            model.download()
        if not getattr(model, "is_loaded", False):
            model.load()
        return model


def _resolve_foundry_streaming_asr_model_id(model_alias: str) -> str:
    model_info = FOUNDRY_STREAMING_ASR_MODELS.get(model_alias)
    if model_info is None:
        return model_alias
    model_id = model_info.get("modelId")
    return str(model_id or model_alias)


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to use the local ASR fallback") from exc

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def warm_faster_whisper(model_name: str = "tiny", device: str = "cpu", compute_type: str = "int8") -> None:
    _get_faster_whisper_model(model_name, device, compute_type)


@dataclass(frozen=True)
class ASRProviderPlan:
    primary: str
    fallback: str | None
    language: str


def plan_asr_provider(primary: str, language: str, cloud_fallback_enabled: bool) -> ASRProviderPlan:
    validate_asr_provider(primary)
    fallback = None
    if cloud_fallback_enabled:
        fallback = "cloud-asr"
    return ASRProviderPlan(primary=primary, fallback=fallback, language=language)
