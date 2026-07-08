from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

from voice_agent.config import LlamaCppSettings
from voice_agent.providers.llm_foundry import ChatMessage


@dataclass
class PreparedLlamaCppTurn:
    llm: "LlamaCppLLM"
    messages: list[ChatMessage]

    def with_user_text(self, user_text: str) -> "PreparedLlamaCppTurn":
        return PreparedLlamaCppTurn(self.llm, [*self.messages, ChatMessage("user", user_text)])

    async def complete(self) -> str:
        return await self.llm.complete(self.messages)

    async def stream(self) -> AsyncIterator[str]:
        async for token in self.llm.stream(self.messages):
            yield token


@dataclass(frozen=True)
class LlamaCppLLM:
    settings: LlamaCppSettings

    @property
    def completion_url(self) -> str:
        return f"{self.settings.endpoint.rstrip('/')}/completion"

    @property
    def health_url(self) -> str:
        return f"{self.settings.endpoint.rstrip('/')}/health"

    async def prepare_turn(self, messages: Iterable[ChatMessage]) -> PreparedLlamaCppTurn:
        prepared_messages = list(messages)
        await self._completion(_prompt_from_messages(prepared_messages, add_generation_prompt=False), stream=False, max_tokens=0)
        return PreparedLlamaCppTurn(self, prepared_messages)

    async def complete(self, messages: Iterable[ChatMessage]) -> str:
        payload = await self._completion(_prompt_from_messages(messages), stream=False)
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp response did not include content")
        return content

    async def stream(self, messages: Iterable[ChatMessage]) -> AsyncIterator[str]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx to use llama.cpp LLM") from exc

        payload = self._payload(_prompt_from_messages(messages), stream=True)
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            async with client.stream("POST", self.completion_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    token = _parse_completion_line(line)
                    if token:
                        yield token

    async def health(self) -> dict[str, object]:
        try:
            import httpx
        except ImportError:
            return {"ready": False, "error": "httpx is not installed"}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(self.health_url)
                if response.status_code == 404:
                    response = await client.get(f"{self.settings.endpoint.rstrip('/')}/props")
                response.raise_for_status()
                payload = response.json() if response.content else {}
        except Exception as exc:
            return {"ready": False, "endpoint": self.settings.endpoint, "model": self.settings.model, "modelPath": str(self.settings.model_path), "error": str(exc)}
        return {"ready": True, "endpoint": self.settings.endpoint, "model": self.settings.model, "modelPath": str(self.settings.model_path), "slotId": self.settings.slot_id, "details": payload}

    async def _completion(self, prompt: str, *, stream: bool, max_tokens: int | None = None) -> dict[str, object]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx to use llama.cpp LLM") from exc

        payload = self._payload(prompt, stream=stream, max_tokens=max_tokens)
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(self.completion_url, json=payload)
            response.raise_for_status()
            return response.json()

    def _payload(self, prompt: str, *, stream: bool, max_tokens: int | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "prompt": prompt,
            "stream": stream,
            "cache_prompt": True,
            "id_slot": self.settings.slot_id,
        }
        if max_tokens is not None:
            payload["n_predict"] = max_tokens
        return payload


def _prompt_from_messages(messages: Iterable[ChatMessage], *, add_generation_prompt: bool = True) -> str:
    parts: list[str] = []
    for message in messages:
        role = "model" if message.role == "assistant" else message.role
        parts.append(f"<start_of_turn>{role}\n{message.content.strip()}\n<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _parse_completion_line(line: str) -> str | None:
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
    content = payload.get("content") if isinstance(payload, dict) else None
    return content if isinstance(content, str) and content else None