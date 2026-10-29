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


class MessagesAdapter(ProviderAdapter):
    def __init__(
        self,
        config: ModelConfig,
        client: httpx.AsyncClient,
        *,
        anthropic_version: str,
        default_max_tokens: int,
    ) -> None:
        super().__init__(config, client)
        self.anthropic_version = anthropic_version
        self.default_max_tokens = default_max_tokens

    def _url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _payload(self, request: UnifiedRequest, *, stream: bool) -> dict[str, Any]:
        systems = [message.content for message in request.messages if message.role == "system"]
        if self._json_instruction_required(request):
            systems.append(JSON_ONLY_INSTRUCTION)

        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role != "system"
        ]
        if not messages:
            messages = [{"role": "user", "content": "Please follow the system instructions."}]

        payload: dict[str, Any] = {
            "model": self.config.upstream_model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.default_max_tokens,
            "stream": stream,
        }
        if systems:
            payload["system"] = "\n\n".join(systems)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
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
                "Upstream Messages request timed out",
                status_code=504,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc
        except httpx.RequestError as exc:
            raise GatewayError(
                "UPSTREAM_CONNECTION_FAILED",
                "Could not connect to the Messages upstream",
                status_code=502,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc

        await self._raise_for_upstream_error(response)
        data = self._decode_json(response)
        return UnifiedResponse(
            content=self._extract_text(data),
            finish_reason=self._finish_reason(data.get("stop_reason")),
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
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                reasoning_tokens = 0
                finish_reason: str | None = None
                done_sent = False

                async for event_name, raw_data in self._iter_sse(response):
                    if raw_data == "[DONE]":
                        break
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError as exc:
                        raise GatewayError(
                            "UPSTREAM_BAD_RESPONSE",
                            "Messages upstream emitted invalid SSE JSON",
                            status_code=502,
                            retryable=True,
                        ) from exc
                    if not isinstance(event, dict):
                        continue
                    event_type = str(event.get("type") or event_name)
                    stream_error = self._stream_error(event_type, event)
                    if stream_error is not None:
                        raise stream_error

                    if event_type == "message_start":
                        usage = (event.get("message") or {}).get("usage") or event.get("usage") or {}
                        input_tokens = int(usage.get("input_tokens", input_tokens) or 0)
                        cached_tokens = int(
                            usage.get("cache_read_input_tokens", cached_tokens) or 0
                        )
                        continue

                    if event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") in {"text_delta", "text"}:
                            text = delta.get("text", "")
                            if text:
                                yield UnifiedStreamEvent(type="text_delta", content=str(text))
                        continue

                    if event_type == "message_delta":
                        delta = event.get("delta") or {}
                        finish_reason = self._finish_reason(delta.get("stop_reason"))
                        usage = event.get("usage") or {}
                        output_tokens = int(usage.get("output_tokens", output_tokens) or 0)
                        input_tokens = int(usage.get("input_tokens", input_tokens) or 0)
                        cached_tokens = int(
                            usage.get("cache_read_input_tokens", cached_tokens) or 0
                        )
                        continue

                    if event_type == "message_stop":
                        usage = TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                            cached_input_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                        )
                        yield UnifiedStreamEvent(
                            type="done",
                            finish_reason=finish_reason or "stop",
                            usage=usage,
                        )
                        done_sent = True
                        break

                if not done_sent:
                    usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        cached_input_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                    yield UnifiedStreamEvent(
                        type="done",
                        finish_reason=finish_reason or "stop",
                        usage=usage,
                    )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                "UPSTREAM_TIMEOUT",
                "Upstream Messages stream timed out",
                status_code=504,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc
        except httpx.RequestError as exc:
            raise GatewayError(
                "UPSTREAM_CONNECTION_FAILED",
                "Could not connect to the Messages upstream",
                status_code=502,
                retryable=True,
                details={"model": self.config.alias},
            ) from exc

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        content = data.get("content", [])
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _finish_reason(reason: Any) -> str | None:
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        return mapping.get(reason, str(reason) if reason else None)
