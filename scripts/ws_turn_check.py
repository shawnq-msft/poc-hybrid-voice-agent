from __future__ import annotations

import asyncio
import json

import websockets

from voice_agent.providers.tts_windows import synthesize_sapi_wav


async def main() -> None:
    audio = synthesize_sapi_wav("Hello, how are you today?")
    events: list[dict[str, object]] = []
    audio_chunks = 0
    audio_bytes = 0

    async with websockets.connect("ws://127.0.0.1:8787/api/session/turn-ws", max_size=32 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"type": "config", "ttsProvider": "azure-embedded", "filename": "input.wav", "mediaType": "audio/wav"}))
        await websocket.send(audio)
        await websocket.send(json.dumps({"type": "end"}))

        async for message in websocket:
            if isinstance(message, bytes):
                audio_chunks += 1
                audio_bytes += len(message)
                continue
            event = json.loads(message)
            events.append(event)
            if event.get("event") == "done":
                break

    result = next((event.get("result", {}) for event in events if event.get("event") == "result"), {})
    summary = {
        "eventTypes": [event.get("event") for event in events],
        "audioChunks": audio_chunks,
        "audioBytes": audio_bytes,
        "ttsProvider": result.get("tts", {}).get("provider"),
        "llmTtftMs": result.get("timingsMs", {}).get("llm"),
        "ttsTtfbMs": result.get("timingsMs", {}).get("tts"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
