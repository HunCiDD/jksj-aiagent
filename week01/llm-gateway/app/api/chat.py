from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_gateway_service, get_tenant_id, require_gateway_key
from app.domain import UnifiedStreamEvent
from app.schemas import ChatCompletionRequest, TokenUsage
from app.services.gateway import ExecutionResult, GatewayService, PreparedInvocation


router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(require_gateway_key)],
)


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
    service: GatewayService = Depends(get_gateway_service),
):
    request_id = request.state.request_id
    invocation = await service.prepare(
        body,
        request_id=request_id,
        tenant_id=tenant_id,
    )

    if body.stream:
        events = service.stream(invocation)
        try:
            first_event = await anext(events)
        except StopAsyncIteration:
            await events.aclose()
            from app.core.errors import GatewayError

            raise GatewayError(
                "UPSTREAM_EMPTY_STREAM",
                "Upstream returned an empty stream",
                status_code=502,
            )
        return StreamingResponse(
            _sse_response(invocation, first_event, events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "x-request-id": request_id,
            },
        )

    result = await service.complete(invocation)
    return _completion_response(invocation, result)


def _usage_payload(usage: TokenUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
    }


def _gateway_payload(
    invocation: PreparedInvocation,
    *,
    ttft_ms: float | None,
    total_latency_ms: float | None,
    retry_count: int,
    attempt_count: int,
) -> dict[str, Any]:
    return {
        "request_id": invocation.request_id,
        "provider": invocation.config.provider,
        "protocol": invocation.config.protocol,
        "prompt_name": invocation.prompt_name,
        "prompt_version": invocation.prompt_version,
        "retry_count": retry_count,
        "attempt_count": attempt_count,
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "total_latency_ms": round(total_latency_ms, 3)
        if total_latency_ms is not None
        else None,
    }


def _completion_response(
    invocation: PreparedInvocation,
    result: ExecutionResult,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{invocation.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": invocation.requested_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response.content,
                },
                "finish_reason": result.response.finish_reason,
            }
        ],
        "usage": _usage_payload(result.response.usage),
        "gateway": _gateway_payload(
            invocation,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,
            retry_count=result.retry_count,
            attempt_count=result.attempt_count,
        ),
    }


def _chunk(
    invocation: PreparedInvocation,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: TokenUsage | None = None,
    gateway: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{invocation.request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": invocation.requested_model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = _usage_payload(usage)
    if gateway is not None:
        payload["gateway"] = gateway
    return payload


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def _sse_response(
    invocation: PreparedInvocation,
    first_event: UnifiedStreamEvent,
    events: AsyncIterator[UnifiedStreamEvent],
) -> AsyncIterator[str]:
    yield _sse(_chunk(invocation, delta={"role": "assistant"}))

    try:
        current = first_event
        while True:
            if current.type == "text_delta":
                yield _sse(
                    _chunk(
                        invocation,
                        delta={"content": current.content or ""},
                    )
                )
            elif current.type == "done":
                yield _sse(
                    _chunk(
                        invocation,
                        delta={},
                        finish_reason=current.finish_reason,
                        usage=current.usage or TokenUsage(),
                        gateway=_gateway_payload(
                            invocation,
                            ttft_ms=current.ttft_ms,
                            total_latency_ms=current.total_latency_ms,
                            retry_count=current.retry_count,
                            attempt_count=current.attempt_count,
                        ),
                    )
                )
            elif current.type == "error":
                yield _sse(current.error or {"error": {"code": "STREAM_ERROR"}})

            current = await anext(events)
    except StopAsyncIteration:
        pass
    finally:
        await events.aclose()
        yield "data: [DONE]\n\n"
