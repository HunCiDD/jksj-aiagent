from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import ModelConfig
from app.core.errors import GatewayError
from app.domain import UnifiedRequest, UnifiedResponse, UnifiedStreamEvent
from app.schemas import TokenUsage


JSON_ONLY_INSTRUCTION = (
    "Return exactly one valid JSON object and nothing else. "
    "Do not use Markdown fences, comments, or explanatory text."
)


class ProviderAdapter(ABC):
    def __init__(self, config: ModelConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client

    @abstractmethod
    async def complete(self, request: UnifiedRequest) -> UnifiedResponse:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: UnifiedRequest) -> AsyncIterator[UnifiedStreamEvent]:
        raise NotImplementedError

    def _ensure_configured(self) -> None:
        if not self.config.base_url or not self.config.api_key:
            raise GatewayError(
                "UPSTREAM_NOT_CONFIGURED",
                f"Model {self.config.alias} is missing base URL or API key",
                status_code=503,
                retryable=False,
                details={"model": self.config.alias},
            )

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.config.timeout_seconds)

    async def _raise_for_upstream_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        await response.aread()
        status = response.status_code
        if status in (400, 413, 422):
            code = "UPSTREAM_INVALID_REQUEST"
            gateway_status = 400
        elif status in (401, 403):
            code = "UPSTREAM_AUTH_ERROR"
            gateway_status = 502
        elif status == 404:
            code = "UPSTREAM_MODEL_NOT_FOUND"
            gateway_status = 502
        elif status == 408:
            code = "UPSTREAM_TIMEOUT"
            gateway_status = 504
        elif status == 429:
            code = "UPSTREAM_RATE_LIMITED"
            gateway_status = 429
        elif status >= 500:
            code = "UPSTREAM_UNAVAILABLE"
            gateway_status = 503
        else:
            code = "UPSTREAM_BAD_RESPONSE"
            gateway_status = 502

        retry_after_header = response.headers.get("retry-after")
        retry_after = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else None
        raise GatewayError(
            code,
            f"Upstream model request failed with HTTP {status}",
            status_code=gateway_status,
            retryable=status == 408 or status == 429 or status >= 500,
            details={"upstream_status": status, "model": self.config.alias},
            retry_after=retry_after,
        )

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise GatewayError(
                "UPSTREAM_BAD_RESPONSE",
                "Upstream returned a non-JSON response",
                status_code=502,
                retryable=True,
            ) from exc
        if not isinstance(data, dict):
            raise GatewayError(
                "UPSTREAM_BAD_RESPONSE",
                "Upstream JSON response must be an object",
                status_code=502,
                retryable=True,
            )
        return data

    @staticmethod
    def _normalise_usage(raw: dict[str, Any] | None) -> TokenUsage:
        raw = raw or {}
        input_details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
        output_details = raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}

        input_tokens = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
        output_tokens = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
        cached_tokens = int(
            raw.get(
                "cache_read_input_tokens",
                input_details.get("cached_tokens", 0),
            )
            or 0
        )
        reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)
        total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @staticmethod
    def _json_instruction_required(request: UnifiedRequest) -> bool:
        return request.response_format is not None

    @staticmethod
    async def _iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
        event_name = ""
        data_lines: list[str] = []
        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\r")
            if line == "":
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = ""
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                continue
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            yield event_name, "\n".join(data_lines)

    @staticmethod
    def _stream_error(event_type: str, data: dict[str, Any]) -> GatewayError | None:
        if event_type not in {"error", "response.failed", "response.error"} and data.get("type") not in {
            "error",
            "response.failed",
            "response.error",
        }:
            return None
        message = data.get("message") or data.get("error") or "Upstream streaming request failed"
        return GatewayError(
            "UPSTREAM_STREAM_ERROR",
            "Upstream streaming request failed",
            status_code=502,
            retryable=False,
            details={"upstream_error": str(message)[:500]},
        )
