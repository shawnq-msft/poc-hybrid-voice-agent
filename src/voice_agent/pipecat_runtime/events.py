from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


ProgressCallback = Callable[[dict[str, object]], Awaitable[None]]


@dataclass(frozen=True)
class PipelineEvent:
    module: str
    phase: str
    text: str | None = None
    voice: str | None = None
    latency_ms: float | None = None
    total_ms: float | None = None
    audio_base64: str | None = None
    audio_media_type: str | None = None

    def as_progress(self) -> dict[str, object]:
        payload: dict[str, object] = {"stage": self.module, "status": self.phase}
        if self.text is not None:
            payload["text"] = self.text
        if self.voice is not None:
            payload["voice"] = self.voice
        if self.latency_ms is not None:
            payload["latencyMs"] = self.latency_ms
        if self.total_ms is not None:
            payload["totalMs"] = self.total_ms
        if self.audio_base64 is not None:
            payload["audioBase64"] = self.audio_base64
        if self.audio_media_type is not None:
            payload["audioMediaType"] = self.audio_media_type
        return payload


async def emit_progress(progress_callback: ProgressCallback | None, event: PipelineEvent) -> None:
    if progress_callback is not None:
        await progress_callback(event.as_progress())


def progress_event(
    module: str,
    phase: str,
    *,
    text: str | None = None,
    voice: str | None = None,
    latency_ms: float | None = None,
    total_ms: float | None = None,
    audio_base64: str | None = None,
    audio_media_type: str | None = None,
) -> PipelineEvent:
    return PipelineEvent(
        module=module,
        phase=phase,
        text=text,
        voice=voice,
        latency_ms=latency_ms,
        total_ms=total_ms,
        audio_base64=audio_base64,
        audio_media_type=audio_media_type,
    )
