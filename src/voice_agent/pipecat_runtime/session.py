from __future__ import annotations

from dataclasses import dataclass

from voice_agent.config import Settings
from voice_agent.pipeline import PipelineBlueprint, build_pipeline_blueprint


@dataclass(frozen=True)
class PipecatRuntimePlan:
    blueprint: PipelineBlueprint
    input_lane: str
    vad_starts_stream: bool
    vad_ends_turn: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "inputLane": self.input_lane,
            "vadStartsStream": self.vad_starts_stream,
            "vadEndsTurn": self.vad_ends_turn,
            "blueprint": self.blueprint.as_dict(),
        }


def build_runtime_plan(settings: Settings) -> PipecatRuntimePlan:
    blueprint = build_pipeline_blueprint(settings)
    return PipecatRuntimePlan(
        blueprint=blueprint,
        input_lane="streaming-asr" if blueprint.asr_transport_mode == "streaming" else "batch-asr",
        vad_starts_stream=blueprint.asr_transport_mode == "streaming",
        vad_ends_turn=blueprint.vad_role == "start-end",
    )
