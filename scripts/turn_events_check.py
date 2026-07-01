from __future__ import annotations

import json

import httpx

from voice_agent.providers.tts_windows import synthesize_sapi_wav


def main() -> None:
    audio = synthesize_sapi_wav("Hello, how are you today?")
    with httpx.stream(
        "POST",
        "http://127.0.0.1:8787/api/session/turn-events",
        files={
            "audio": ("input.wav", audio, "audio/wav"),
            "tts_provider": (None, "azure-embedded"),
        },
        timeout=240,
    ) as response:
        response.raise_for_status()
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    summary = {
        "eventTypes": [event.get("event") for event in events],
        "progress": [
            {"stage": event.get("stage"), "status": event.get("status"), "latencyMs": event.get("latencyMs")}
            for event in events
            if event.get("event") == "progress"
        ],
        "result": next((event.get("result", {}) for event in events if event.get("event") == "result"), {}),
    }
    result = summary["result"]
    compact = {
        "eventTypes": summary["eventTypes"],
        "progress": summary["progress"],
        "audioChunks": len(
            [event for event in events if event.get("event") == "progress" and event.get("stage") == "tts" and event.get("status") == "audio"]
        ),
        "llmTokenEvents": len(
            [event for event in events if event.get("event") == "progress" and event.get("stage") == "llm" and event.get("status") == "token"]
        ),
        "firstAudioBytes": len(
            next(
                (
                    event.get("audioBase64", "")
                    for event in events
                    if event.get("event") == "progress" and event.get("stage") == "tts" and event.get("status") == "audio"
                ),
                "",
            )
        ),
        "ttsProvider": result.get("tts", {}).get("provider"),
        "llmTtftMs": result.get("timingsMs", {}).get("llm"),
        "ttsTtfbMs": result.get("timingsMs", {}).get("tts"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
