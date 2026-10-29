from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from app.core.config import ModelConfig, Settings


class AsyncBodyStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content

    async def aclose(self) -> None:
        return None


def json_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        stream=AsyncBodyStream(json.dumps(data).encode()),
    )


def sse_response(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    )
    body += "data: [DONE]\n\n"
    return httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=AsyncBodyStream(body.encode()),
    )


class MockUpstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.pro_failures_remaining = 0
        self.invalid_structured_output = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/responses"):
            if self.pro_failures_remaining > 0:
                self.pro_failures_remaining -= 1
                return json_response({"error": {"message": "temporary"}}, status_code=503)
            return self._responses(request, body)
        if request.url.path.endswith("/messages"):
            return self._messages(request, body)
        return json_response({"error": "unknown path"}, status_code=404)

    def _responses(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer pro-key"
        content = self._content(body.get("instructions", ""))
        if body.get("stream"):
            return sse_response(
                [
                    {"type": "response.created"},
                    {"type": "response.output_text.delta", "delta": content[:5]},
                    {"type": "response.output_text.delta", "delta": content[5:]},
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "usage": {
                                "input_tokens": 11,
                                "output_tokens": 7,
                                "total_tokens": 18,
                                "input_tokens_details": {"cached_tokens": 3},
                                "output_tokens_details": {"reasoning_tokens": 2},
                            },
                        },
                    },
                ]
            )
        return json_response(
            {
                "id": "resp_mock",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            }
        )

    def _messages(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        assert request.headers["x-api-key"] == "flash-key"
        content = self._content(body.get("system", ""))
        if body.get("stream"):
            return sse_response(
                [
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 13, "cache_read_input_tokens": 4}},
                    },
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": content[:5]},
                    },
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": content[5:]},
                    },
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 9},
                    },
                    {"type": "message_stop"},
                ]
            )
        return json_response(
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 13,
                    "output_tokens": 9,
                    "cache_read_input_tokens": 4,
                },
            }
        )

    def _content(self, instructions: str) -> str:
        if "Return exactly one valid JSON object" in instructions:
            if self.invalid_structured_output:
                return "This is not JSON"
            return '{"ok": true}'
        return instructions or "mock response"


def make_settings(
    tmp_path: Path,
    *,
    pro_rpm: int = 100,
    pro_burst: int = 10,
    flash_rpm: int = 100,
    flash_burst: int = 10,
    retry_base_delay: float = 0.001,
) -> Settings:
    pro = ModelConfig(
        alias="deepseek-v4-pro",
        provider="deepseek",
        protocol="openai_responses",
        upstream_model="deepseek-v4-pro-upstream",
        base_url="http://upstream.test/v1",
        api_key="pro-key",
        timeout_seconds=5,
        rate_limit_rpm=pro_rpm,
        rate_limit_burst=pro_burst,
    )
    flash = ModelConfig(
        alias="deepseek-v4-flash",
        provider="deepseek",
        protocol="anthropic_messages",
        upstream_model="deepseek-v4-flash-upstream",
        base_url="http://upstream.test/v1",
        api_key="flash-key",
        timeout_seconds=5,
        rate_limit_rpm=flash_rpm,
        rate_limit_burst=flash_burst,
    )
    return Settings(
        gateway_api_key="test-gateway-key",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'gateway.db').as_posix()}",
        models={pro.alias: pro, flash.alias: flash},
        max_retries=3,
        retry_base_delay=retry_base_delay,
        retry_max_delay=0.01,
        default_max_tokens=512,
    )
