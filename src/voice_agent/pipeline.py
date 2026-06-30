from __future__ import annotations

from dataclasses import dataclass

from voice_agent.config import ASR_PROVIDER_CAPABILITIES, Settings


@dataclass(frozen=True)
class PipelineStage:
    name: str
    provider: str
    role: str
    fallback_provider: str | None = None


@dataclass(frozen=True)
class PipelineBlueprint:
    stages: tuple[PipelineStage, ...]
    full_duplex: bool
    barge_in_enabled: bool
    transport: str
    asr_inference_mode: str
    asr_transport_mode: str
    vad_role: str

    def as_dict(self) -> dict[str, object]:
        return {
            "transport": self.transport,
            "fullDuplex": self.full_duplex,
            "bargeInEnabled": self.barge_in_enabled,
            "asrInferenceMode": self.asr_inference_mode,
            "asrTransportMode": self.asr_transport_mode,
            "vadRole": self.vad_role,
            "stages": [stage.__dict__ for stage in self.stages],
        }


def build_pipeline_blueprint(settings: Settings) -> PipelineBlueprint:
    asr_capabilities = ASR_PROVIDER_CAPABILITIES.get(settings.providers.asr, {})
    asr_inference_mode = str(asr_capabilities.get("inferenceMode", "batch"))
    asr_transport_mode = str(asr_capabilities.get("transportMode", "batch"))
    vad_role = str(asr_capabilities.get("vadRole", "start-end"))
    asr_fallback = None
    if settings.providers.cloud_fallback_enabled:
        asr_fallback = "cloud-asr"

    tts_fallback = "windows-sapi" if settings.providers.tts in {"windows-winrt", "azure-embedded"} else None
    if settings.providers.cloud_fallback_enabled and tts_fallback is None:
        tts_fallback = "cloud-tts"

    llm_fallback = "cloud-llm" if settings.providers.cloud_fallback_enabled else None

    stages = (
        PipelineStage("transport-input", "browser-webrtc", "audio-input"),
        PipelineStage("vad", settings.providers.vad, vad_role),
        PipelineStage("asr", settings.providers.asr, "speech-to-text", asr_fallback),
        PipelineStage("dialogue", settings.providers.llm, "conversation", llm_fallback),
        PipelineStage("tool-router", "allowlisted-copilot-tools", "local-action"),
        PipelineStage("tts", settings.providers.tts, "text-to-speech", tts_fallback),
        PipelineStage("transport-output", "browser-webrtc", "audio-output"),
    )
    return PipelineBlueprint(
        stages=stages,
        full_duplex=True,
        barge_in_enabled=True,
        transport="browser-streaming" if asr_transport_mode == "streaming" else "browser-batch",
        asr_inference_mode=asr_inference_mode,
        asr_transport_mode=asr_transport_mode,
        vad_role=vad_role,
    )
