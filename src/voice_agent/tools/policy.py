from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


class ToolPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    kind: str
    command: str
    allowed_arguments: tuple[str, ...]


@dataclass(frozen=True)
class ToolRequest:
    action: str
    arguments: Mapping[str, str]


@dataclass(frozen=True)
class ToolPolicy:
    workspace_root: Path
    tools: Mapping[str, ToolDefinition]

    @classmethod
    def default(cls, workspace_root: Path) -> "ToolPolicy":
        tools = {
            "vscode.open-file": ToolDefinition(
                name="vscode.open-file",
                kind="vscode-command",
                command="vscode.open",
                allowed_arguments=("path",),
            ),
            "vscode.run-task": ToolDefinition(
                name="vscode.run-task",
                kind="vscode-command",
                command="workbench.action.tasks.runTask",
                allowed_arguments=("task",),
            ),
            "copilot.submit-prompt": ToolDefinition(
                name="copilot.submit-prompt",
                kind="copilot-request",
                command="github.copilot.chat.submit",
                allowed_arguments=("prompt",),
            ),
        }
        return cls(workspace_root=workspace_root.resolve(), tools=tools)

    def validate(self, request: ToolRequest) -> ToolDefinition:
        definition = self.tools.get(request.action)
        if definition is None:
            raise ToolPolicyError(f"Tool action is not allowlisted: {request.action}")

        unexpected = set(request.arguments) - set(definition.allowed_arguments)
        if unexpected:
            raise ToolPolicyError(f"Unexpected tool arguments: {sorted(unexpected)}")

        missing = set(definition.allowed_arguments) - set(request.arguments)
        if missing:
            raise ToolPolicyError(f"Missing tool arguments: {sorted(missing)}")

        for name, value in request.arguments.items():
            if not isinstance(value, str) or not value.strip():
                raise ToolPolicyError(f"Tool argument must be a non-empty string: {name}")
            if len(value) > 4000:
                raise ToolPolicyError(f"Tool argument is too long: {name}")
            if name == "path":
                self._validate_workspace_path(value)

        return definition

    def _validate_workspace_path(self, value: str) -> None:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            candidate.resolve().relative_to(self.workspace_root)
        except ValueError as exc:
            raise ToolPolicyError("Tool path argument must stay inside the workspace") from exc


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        enriched = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")
