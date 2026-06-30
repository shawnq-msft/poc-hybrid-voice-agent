import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from voice_agent.config import CopilotToolSettings
from voice_agent.tools.copilot_agent import CopilotAgentBridge
from voice_agent.tools.policy import ToolPolicy, ToolPolicyError, ToolRequest


class ToolPolicyTests(unittest.TestCase):
    def test_rejects_unknown_action(self):
        policy = ToolPolicy.default(Path("C:/workspace"))

        with self.assertRaises(ToolPolicyError):
            policy.validate(ToolRequest("shell.run", {"command": "del *"}))

    def test_rejects_path_outside_workspace(self):
        policy = ToolPolicy.default(Path("C:/workspace"))

        with self.assertRaises(ToolPolicyError):
            policy.validate(ToolRequest("vscode.open-file", {"path": "C:/Windows/System32/drivers/etc/hosts"}))

    def test_dry_run_writes_audit_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = CopilotToolSettings(
                enabled=True,
                dry_run=True,
                policy_path=root / "policy.json",
                audit_log_path=root / "audit.jsonl",
                workspace_root=root,
            )
            bridge = CopilotAgentBridge(settings)

            result = bridge.invoke(ToolRequest("copilot.submit-prompt", {"prompt": "Summarize workspace"}))

            self.assertEqual(result.status, "dry-run")
            self.assertTrue(settings.audit_log_path.exists())
            self.assertIn("copilot.submit-prompt", settings.audit_log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
