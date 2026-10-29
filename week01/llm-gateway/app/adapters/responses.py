from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.adapters.base import JSON_ONLY_INSTRUCTION, ProviderAdapter
from app.core.config import ModelConfig
from app.core.errors import GatewayError
from app.domain import UnifiedRequest, UnifiedResponse, UnifiedStreamEvent
from app.schemas import TokenUsage


class ResponsesAdapter(ProviderAdapter):
    def __init__(self, config: ModelConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config, client)

    def _url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/responses"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _payload(self, request: UnifiedRequest, *, stream: bool) -> dict[str, Any]:
        systems = [message.content for message in request.messages if message.role == "system"]
        if self._json_instruction_required(request):
            systems.append(JSON_ONLY_INSTRUCTION)

        input_messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role != "system"
        ]
        if not input_messages:
            input_messages = [{"role": "user", "content": "Please follow the system instructions."}]

        payload: dict[str, Any] = {
            "model": self.config.upstream_model,
            "input": input_messages,
            "stream": stream,
        }
        if systems:
            payload["instructions"] = "\n\n".join(systems)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        return payload

    async def complete(self, request: UnifiedRequest) -> UnifiedResponse:
        self._ensure_configured()
        try:
            response = await self.client.post(
                self._url(),
                headers=self._headers(),
                json=self._payload(request, stream=False),
                timeout=self._timeout(),
            )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                "UPSTREAM_TIMEOUT",
                "Upstream Responses request timed out",
                status_code=504,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc
        except httpx.RequestError as exc:
            raise GatewayError(
                "UPSTREAM_CONNECTION_FAILED",
                "Could not connect to the Responses upstream",
                status_code=502,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc

        await self._raise_for_upstream_error(response)
        data = self._decode_json(response)
        return UnifiedResponse(
            content=self._extract_text(data),
            finish_reason=self._finish_reason(data),
            usage=self._normalise_usage(data.get("usage")),
        )

    async def stream(self, request: UnifiedRequest) -> AsyncIterator[UnifiedStreamEvent]:
        self._ensure_configured()
        try:
            async with self.client.stream(
                "POST",
                self._url(),
                headers={**self._headers(), "Accept": "text/event-stream"},
                json=self._payload(request, stream=True),
                timeout=self._timeout(),
            ) as response:
                await self._raise_for_upstream_error(response)
                done_sent = False
                async for event_name, raw_data in self._iter_sse(response):
                    if raw_data == "[DONE]":
                        break
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError as exc:
                        raise GatewayError(
                            "UPSTREAM_BAD_RESPONSE",
                            "Responses upstream emitted invalid SSE JSON",
                            status_code=502,
                            retryable=True,
                        ) from exc
                    if not isinstance(event, dict):
                        continue
                    event_type = str(event.get("type") or event_name)
                    stream_error = self._stream_error(event_type, event)
                    if stream_error is not None:
                        raise stream_error

                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            yield UnifiedStreamEvent(type="text_delta", content=str(delta))
                        continue

                    if event_type == "response.completed":
                        completed = event.get("response") or event
                        yield UnifiedStreamEvent(
                            type="done",
                            finish_reason=self._finish_reason(completed),
                            usage=self._normalise_usage(completed.get("usage")),
                        )
                        done_sent = True
                        break

                    # Some Responses-compatible servers emit chat-style deltas.
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices:
                        candidate = choices[0] or {}
                        delta = candidate.get("delta", {}).get("content")
                        if delta:
                            yield UnifiedStreamEvent(type="text_delta", content=str(delta))

                if not done_sent:
                    yield UnifiedStreamEvent(type="done", finish_reason="stop", usage=TokenUsage())
        except httpx.TimeoutException as exc:
            raise GatewayError(
                "UPSTREAM_TIMEOUT",
                "Upstream Responses stream timed out",
                status_code=504,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc
        except httpx.RequestError as exc:
            raise GatewayError(
                "UPSTREAM_CONNECTION_FAILED",
                "Could not connect to the Responses upstream",
                status_code=502,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        top_level = data.get("output_text")
        if isinstance(top_level, str):
            return top_level

        parts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if not parts:
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = (choices[0] or {}).get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "".join(parts)

    @staticmethod
    def _finish_reason(data: dict[str, Any]) -> str | None:
        status = data.get("status")
        if status == "completed":
            return "stop"
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            return "length" if reason == "max_output_tokens" else str(reason or "incomplete")
        if data.get("finish_reason"):
            return str(data["finish_reason"])
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            return (choices[0] or {}).get("finish_reason")
        return "stop"
