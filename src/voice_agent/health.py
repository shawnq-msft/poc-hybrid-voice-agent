from __future__ import annotations

import json
import os

from voice_agent.config import Settings
from voice_agent.pipeline import build_pipeline_blueprint


def collect_health(settings: Settings) -> dict[str, object]:
    return {
        "status": "ok",
        "settings": settings.public_summary(),
        "pipeline": build_pipeline_blueprint(settings).as_dict(),
    }


def main() -> None:
    settings = Settings.from_env(os.environ)
    print(json.dumps(collect_health(settings), indent=2))


if __name__ == "__main__":
    main()
