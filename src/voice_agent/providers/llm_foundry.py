from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

from voice_agent.config import FoundrySettings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class FoundryLocalLLM:
    settings: FoundrySettings

    @property
    def chat_url(self) -> str:
        return f"{self.settings.endpoint.rstrip('/')}/chat/completions"

    async def complete(self, messages: Iterable[ChatMessage]) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx to use Foundry Local LLM") from exc

        payload = {
            "model": self.settings.llm_model,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(self.chat_url, json=payload)
            response.raise_for_status()
            body = response.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Foundry Local LLM response did not include assistant content") from exc

    async def stream(self, messages: Iterable[ChatMessage]) -> AsyncIterator[str]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx to use Foundry Local LLM") from exc

        payload = {
            "model": self.settings.llm_model,
            "messages": [message.__dict__ for message in messages],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            async with client.stream("POST", self.chat_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is not None:
                        yield chunk


def _parse_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("data:"):
        stripped = stripped.removeprefix("data:").strip()
    if stripped == "[DONE]":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    try:
        choice = payload["choices"][0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is None:
            content = choice.get("message", {}).get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return content if isinstance(content, str) and content else None
