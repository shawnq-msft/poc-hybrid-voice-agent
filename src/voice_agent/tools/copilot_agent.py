from __future__ import annotations

import subprocess
from dataclasses import dataclass

from voice_agent.config import CopilotToolSettings
from voice_agent.tools.policy import AuditLog, ToolPolicy, ToolRequest


@dataclass(frozen=True)
class ToolResult:
    action: str
    dry_run: bool
    status: str
    detail: str


class CopilotAgentBridge:
    def __init__(self, settings: CopilotToolSettings, policy: ToolPolicy | None = None) -> None:
        self.settings = settings
        self.policy = policy or ToolPolicy.default(settings.workspace_root)
        self.audit_log = AuditLog(settings.audit_log_path)

    def invoke(self, request: ToolRequest) -> ToolResult:
        if not self.settings.enabled:
            return ToolResult(request.action, True, "disabled", "Copilot tools are disabled")

        definition = self.policy.validate(request)
        self.audit_log.write(
            {
                "action": request.action,
                "arguments": dict(request.arguments),
                "dryRun": self.settings.dry_run,
                "kind": definition.kind,
            }
        )

        if self.settings.dry_run:
            return ToolResult(request.action, True, "dry-run", definition.command)

        if definition.kind == "vscode-command":
            completed = subprocess.run(
                ["code", "--reuse-window", "--command", definition.command],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            status = "completed" if completed.returncode == 0 else "failed"
            detail = completed.stdout.strip() or completed.stderr.strip() or definition.command
            return ToolResult(request.action, False, status, detail)

        return ToolResult(
            request.action,
            False,
            "queued",
            "Copilot prompt request was audited for a VS Code bridge to consume.",
        )
