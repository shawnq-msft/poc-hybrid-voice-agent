from __future__ import annotations

import json
import os
from array import array
from dataclasses import dataclass

from voice_agent.config import Settings
from voice_agent.pipeline import build_pipeline_blueprint
from voice_agent.providers.asr import ASRTranscript
from voice_agent.providers.vad import EnergyVad
from voice_agent.tools.copilot_agent import CopilotAgentBridge, ToolResult
from voice_agent.tools.policy import ToolRequest


@dataclass(frozen=True)
class SmokeResult:
    status: str
    user_text: str
    assistant_text: str
    vad: dict[str, object]
    asr: dict[str, object]
    llm: dict[str, object]
    tts: dict[str, object]
    tool: dict[str, object] | None
    pipeline: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "userText": self.user_text,
            "assistantText": self.assistant_text,
            "vad": self.vad,
            "asr": self.asr,
            "llm": self.llm,
            "tts": self.tts,
            "tool": self.tool,
            "pipeline": self.pipeline,
        }


def _synthetic_pcm16_frame() -> bytes:
    samples = array("h", [12000 if index % 2 == 0 else -12000 for index in range(320)])
    return samples.tobytes()


def run_smoke_turn(settings: Settings, user_text: str = "Open Copilot and summarize this workspace") -> SmokeResult:
    blueprint = build_pipeline_blueprint(settings)

    vad_decision = EnergyVad(threshold=0.01).analyze_pcm16(_synthetic_pcm16_frame())
    if not vad_decision.is_speech:
        raise RuntimeError("Smoke VAD did not detect synthetic speech")

    transcript = ASRTranscript(text=user_text, language=settings.audio.asr_language, provider="mock-asr")
    assistant_text = f"Smoke reply via {settings.foundry.llm_model}: {transcript.text}"
    tts_bytes = assistant_text.encode("utf-8")

    tool_result: ToolResult | None = None
    if settings.copilot_tools.enabled:
        bridge = CopilotAgentBridge(settings.copilot_tools)
        tool_result = bridge.invoke(
            ToolRequest(
                "copilot.submit-prompt",
                {"prompt": f"Smoke test request: {transcript.text}"},
            )
        )

    return SmokeResult(
        status="passed",
        user_text=transcript.text,
        assistant_text=assistant_text,
        vad={
            "provider": vad_decision.provider,
            "isSpeech": vad_decision.is_speech,
            "confidence": round(vad_decision.confidence, 4),
        },
        asr={
            "provider": transcript.provider,
            "configuredProvider": settings.providers.asr,
            "model": settings.foundry.asr_model if settings.providers.asr == "foundry-local" else None,
            "language": transcript.language,
        },
        llm={
            "provider": "mock-llm",
            "configuredProvider": settings.providers.llm,
            "model": settings.foundry.llm_model,
        },
        tts={
            "provider": "mock-tts",
            "configuredProvider": settings.providers.tts,
            "sampleRate": settings.audio.tts_sample_rate,
            "bytes": len(tts_bytes),
        },
        tool=None
        if tool_result is None
        else {
            "action": tool_result.action,
            "dryRun": tool_result.dry_run,
            "status": tool_result.status,
            "detail": tool_result.detail,
        },
        pipeline=blueprint.as_dict(),
    )


def main() -> None:
    settings = Settings.from_env(os.environ)
    print(json.dumps(run_smoke_turn(settings).as_dict(), indent=2))


if __name__ == "__main__":
    main()
