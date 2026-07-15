from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
BASE_MODEL_ID = "google/gemma-4-E2B"
DEFAULT_LOCAL_DIR = Path("models/llm/gemma-4-e2b")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    data: dict[str, Any] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check Gemma 4 E2B local runtime prerequisites.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id to check; default is the official QAT mobile 8-bit instruction variant.")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Local download directory for optional HF snapshot checks.")
    parser.add_argument("--download", action="store_true", help="Download the HF snapshot into --local-dir. The default QAT mobile snapshot is about 2.3GB.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    results = [
        check_foundry_model(args.model_id),
        check_huggingface_model(args.model_id),
        check_transformers_support(args.model_id),
        check_local_snapshot(local_dir),
    ]
    if args.download:
        results.append(download_huggingface_snapshot(args.model_id, local_dir))
        results.append(check_local_snapshot(local_dir))

    print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
    required = [result for result in results if result.name in {"huggingface", "transformers"}]
    return 0 if all(result.ok for result in required) else 1


def check_foundry_model(model_id: str) -> CheckResult:
    try:
        completed = subprocess.run(
            ["foundry", "model", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("foundry", False, f"Foundry CLI is unavailable: {exc}")
    output = f"{completed.stdout}\n{completed.stderr}"
    matches = [line.strip() for line in output.splitlines() if "gemma" in line.lower() or "e2b" in line.lower()]
    if model_id in output or matches:
        return CheckResult("foundry", True, "Foundry catalog lists a Gemma/E2B candidate.", {"matches": matches})
    return CheckResult("foundry", False, "Foundry catalog does not list Gemma 4 E2B; use HF fallback or load a supported Foundry alias.")


def check_huggingface_model(model_id: str) -> CheckResult:
    try:
        from huggingface_hub import model_info
    except ImportError as exc:
        return CheckResult("huggingface", False, "Install huggingface_hub to inspect Gemma 4 E2B.", {"error": str(exc)})
    try:
        info = model_info(model_id, files_metadata=True)
    except Exception as exc:
        return CheckResult("huggingface", False, f"HF model lookup failed for {model_id}: {exc}")
    siblings = getattr(info, "siblings", []) or []
    total_bytes = sum((getattr(sibling, "size", 0) or 0) for sibling in siblings)
    return CheckResult(
        "huggingface",
        True,
        f"HF model {model_id} is available.",
        {
            "modelId": model_id,
            "baseModelId": BASE_MODEL_ID,
            "pipeline": getattr(info, "pipeline_tag", None),
            "library": getattr(info, "library_name", None),
            "fileCount": len(siblings),
            "totalBytes": total_bytes,
            "totalGb": round(total_bytes / 1024**3, 2),
        },
    )


def check_transformers_support(model_id: str) -> CheckResult:
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError as exc:
        return CheckResult("transformers", False, "Install transformers to load Gemma 4 E2B.", {"error": str(exc)})
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        return CheckResult(
            "transformers",
            False,
            f"Transformers {transformers.__version__} cannot load {model_id}: {exc}",
            {"version": transformers.__version__},
        )
    return CheckResult(
        "transformers",
        True,
        f"Transformers {transformers.__version__} recognizes Gemma 4 E2B config.",
        {
            "version": transformers.__version__,
            "configClass": type(config).__name__,
            "modelType": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
        },
    )


def check_local_snapshot(local_dir: Path) -> CheckResult:
    config_path = local_dir / "config.json"
    model_path = local_dir / "model.safetensors"
    if config_path.exists() and model_path.exists():
        return CheckResult("localSnapshot", True, f"Gemma 4 E2B snapshot exists at {local_dir}.", {"path": str(local_dir)})
    return CheckResult("localSnapshot", False, f"No complete local Gemma 4 E2B snapshot found at {local_dir}.")


def download_huggingface_snapshot(model_id: str, local_dir: Path) -> CheckResult:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return CheckResult("download", False, "Install huggingface_hub to download Gemma 4 E2B.", {"error": str(exc)})
    try:
        path = snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as exc:
        return CheckResult("download", False, f"Download failed for {model_id}: {exc}")
    return CheckResult("download", True, f"Downloaded {model_id} to {path}.", {"path": path})


if __name__ == "__main__":
    sys.exit(main())
