from __future__ import annotations

import json

import httpx

from voice_agent.providers.tts_windows import synthesize_sapi_wav


def main() -> None:
    audio = synthesize_sapi_wav("Hello, how are you today?")
    response = httpx.post(
        "http://127.0.0.1:8787/api/session/turn",
        files={"audio": ("input.wav", audio, "audio/wav")},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload["result"]
    summary = {
        "mode": payload["mode"],
        "status": result["status"],
        "userText": result["userText"],
        "assistantText": result["assistantText"],
        "asr": result["asr"],
        "llm": result["llm"],
        "ttsProvider": result["tts"]["provider"],
        "ttsHasAudio": bool(result["tts"]["audioBase64"]),
        "ttsBrowserFallback": result["tts"]["browserFallback"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
