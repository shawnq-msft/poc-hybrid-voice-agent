from __future__ import annotations

import os
from pathlib import Path

from voice_agent.config import Settings
from voice_agent.pipecat_runtime.pure_turn import run_pure_pipecat_audio_turn, run_pure_pipecat_text_turn
from voice_agent.server import create_app


def create_pipecat_app(settings: Settings | None = None):
    return create_app(
        settings,
        audio_turn_runner=run_pure_pipecat_audio_turn,
        text_turn_runner=run_pure_pipecat_text_turn,
        app_title="Pure Pipecat Voice Agent",
        turn_mode="pure-pipecat",
    )


def main() -> None:
    try:
        import uvicorn
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install uvicorn with `python -m pip install -e .`") from exc

    load_dotenv(dotenv_path=Path.cwd() / ".env")
    settings = Settings.from_env(os.environ, base_dir=Path.cwd())
    uvicorn.run(create_pipecat_app(settings), host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
