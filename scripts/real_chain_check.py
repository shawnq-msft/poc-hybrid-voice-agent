from __future__ import annotations

import asyncio
import json

from voice_agent.config import Settings
from voice_agent.providers.tts_windows import synthesize_sapi_wav
from voice_agent.real_turn import run_real_turn


async def main() -> None:
    settings = Settings.from_env({})
    audio = synthesize_sapi_wav("Hello, how are you today?")
    result = await run_real_turn(settings, audio, "input.wav", "audio/wav")
    payload = result.as_dict()
    summary = {
        "status": payload["status"],
        "endpoint": settings.foundry.endpoint,
        "userText": payload["userText"],
        "assistantText": payload["assistantText"],
        "asr": payload["asr"],
        "llm": payload["llm"],
        "ttsProvider": payload["tts"]["provider"],
        "ttsHasAudio": bool(payload["tts"]["audioBase64"]),
        "ttsBrowserFallback": payload["tts"]["browserFallback"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
