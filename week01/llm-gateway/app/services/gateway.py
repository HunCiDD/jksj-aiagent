from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from app.adapters.base import ProviderAdapter
from app.core.config import ModelConfig, Settings
from app.core.errors import GatewayError
from app.core.router import ModelRouter
from app.database import Database, PromptTemplate
from app.domain import UnifiedRequest, UnifiedResponse, UnifiedStreamEvent
from app.rate_limit import ModelRateLimiter
from app.schemas import ChatCompletionRequest, Message, PromptReference, Scalar, TokenUsage


VARIABLE_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


@dataclass(slots=True)
class PreparedInvocation:
    request_id: str
    tenant_id: str
    requested_model: str
    config: ModelConfig
    adapter: ProviderAdapter
    request: UnifiedRequest
    prompt_name: str | None
    prompt_version: int | None
    created_at: datetime
    started_clock: float


@dataclass(slots=True)
class ExecutionResult:
    response: UnifiedResponse
    ttft_ms: float | None
    total_latency_ms: float
    attempt_count: int
    retry_count: int


@dataclass(slots=True)
class _CallState:
    attempts: int = 0
    retries: int = 0
    ttft_ms: float | None = None
    content_parts: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


class GatewayService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        router: ModelRouter,
        adapters: dict[str, ProviderAdapter],
        rate_limiter: ModelRateLimiter,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings
        self.database = database
        self.router = router
        self.adapters = adapters
        self.rate_limiter = rate_limiter
        self.sleep = sleep
        self.clock = clock

    async def prepare(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str,
        tenant_id: str,
    ) -> PreparedInvocation:
        config = self.router.resolve(request.model)
        started_clock = self.clock()
        created_at = datetime.now(UTC)

        prompt_name: str | None = None
        prompt_version: int | None = None
        messages = list(request.messages)
        try:
            if request.prompt is not None:
                prompt, rendered = await self._render_prompt(request.prompt)
                prompt_name = prompt.name
                prompt_version = prompt.version
                if messages:
                    messages.insert(0, Message(role="system", content=rendered))
                else:
                    messages.append(Message(role="user", content=rendered))
        except GatewayError as exc:
            await self._record(
                request_id=request_id,
                tenant_id=tenant_id,
                requested_model=request.model,
                config=config,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                state=_CallState(),
                status="error",
                error=exc,
                total_latency_ms=(self.clock() - started_clock) * 1000,
            )
            raise

        decision = await self.rate_limiter.acquire(request.model)
        if not decision.allowed:
            error = GatewayError(
                "MODEL_RATE_LIMITED",
                f"Rate limit exceeded for model {request.model}",
                status_code=429,
                retryable=False,
                retry_after=decision.retry_after,
                details={"model": request.model},
            )
            await self._record(
                request_id=request_id,
                tenant_id=tenant_id,
                requested_model=request.model,
                config=config,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                state=_CallState(),
                status="error",
                error=error,
                total_latency_ms=(self.clock() - started_clock) * 1000,
            )
            raise error

        adapter = self.adapters.get(config.protocol)
        if adapter is None:
            raise GatewayError(
                "PROTOCOL_NOT_CONFIGURED",
                f"No adapter is registered for protocol {config.protocol}",
                status_code=503,
            )

        unified = UnifiedRequest(
            model=request.model,
            messages=messages,
            stream=request.stream,
            response_format=request.response_format,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return PreparedInvocation(
            request_id=request_id,
            tenant_id=tenant_id,
            requested_model=request.model,
            config=config,
            adapter=adapter,
            request=unified,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            created_at=created_at,
            started_clock=started_clock,
        )

    async def complete(self, invocation: PreparedInvocation) -> ExecutionResult:
        state = _CallState()

        while True:
            state.attempts += 1
            try:
                response = await invocation.adapter.complete(invocation.request)
                self._validate_structured_output(invocation.request, response.content)
                state.usage = response.usage
                total_latency = (self.clock() - invocation.started_clock) * 1000
                await self._record(
                    request_id=invocation.request_id,
                    tenant_id=invocation.tenant_id,
                    requested_model=invocation.requested_model,
                    config=invocation.config,
                    prompt_name=invocation.prompt_name,
                    prompt_version=invocation.prompt_version,
                    state=state,
                    status="success",
                    error=None,
                    total_latency_ms=total_latency,
                )
                return ExecutionResult(
                    response=response,
                    ttft_ms=None,
                    total_latency_ms=total_latency,
                    attempt_count=state.attempts,
                    retry_count=state.retries,
                )
            except GatewayError as exc:
                if self._can_retry(exc, state):
                    state.retries += 1
                    await self.sleep(self._retry_delay(state.retries))
                    continue
                await self._record_failure(invocation, state, exc)
                raise
            except Exception as exc:
                error = GatewayError(
                    "INTERNAL_ERROR",
                    "An unexpected gateway error occurred",
                    status_code=500,
                )
                await self._record_failure(invocation, state, error)
                raise error from exc

    async def stream(self, invocation: PreparedInvocation) -> AsyncIterator[UnifiedStreamEvent]:
        state = _CallState()

        while True:
            state.attempts += 1
            try:
                async for event in invocation.adapter.stream(invocation.request):
                    if event.type == "text_delta":
                        if event.content:
                            state.content_parts.append(event.content)
                            if state.ttft_ms is None:
                                state.ttft_ms = (
                                    self.clock() - invocation.started_clock
                                ) * 1000
                            yield event
                        continue

                    if event.type == "done":
                        state.usage = event.usage or TokenUsage()
                        self._validate_structured_output(
                            invocation.request,
                            "".join(state.content_parts),
                        )
                        total_latency = (self.clock() - invocation.started_clock) * 1000
                        await self._record(
                            request_id=invocation.request_id,
                            tenant_id=invocation.tenant_id,
                            requested_model=invocation.requested_model,
                            config=invocation.config,
                            prompt_name=invocation.prompt_name,
                            prompt_version=invocation.prompt_version,
                            state=state,
                            status="success",
                            error=None,
                            total_latency_ms=total_latency,
                        )
                        yield UnifiedStreamEvent(
                            type="done",
                            finish_reason=event.finish_reason,
                            usage=state.usage,
                            ttft_ms=state.ttft_ms,
                            total_latency_ms=total_latency,
                            retry_count=state.retries,
                            attempt_count=state.attempts,
                        )
                        return

                error = GatewayError(
                    "UPSTREAM_BAD_RESPONSE",
                    "Upstream stream ended without a completion event",
                    status_code=502,
                    retryable=True,
                )
                raise error
            except GatewayError as exc:
                if self._can_retry(exc, state, stream_committed=bool(state.content_parts)):
                    state.retries += 1
                    await self.sleep(self._retry_delay(state.retries))
                    continue
                await self._record_failure(invocation, state, exc)
                if state.content_parts:
                    yield UnifiedStreamEvent(
                        type="error",
                        error=exc.to_payload(invocation.request_id),
                    )
                    return
                raise
            except Exception as exc:
                error = GatewayError(
                    "INTERNAL_ERROR",
                    "An unexpected gateway error occurred",
                    status_code=500,
                )
                await self._record_failure(invocation, state, error)
                if state.content_parts:
                    yield UnifiedStreamEvent(
                        type="error",
                        error=error.to_payload(invocation.request_id),
                    )
                    return
                raise error from exc

    async def _render_prompt(
        self,
        reference: PromptReference,
    ) -> tuple[PromptTemplate, str]:
        prompt = await self.database.get_prompt(reference.name, reference.version)
        required = set(VARIABLE_PATTERN.findall(prompt.template))
        missing = sorted(required - reference.variables.keys())
        if missing:
            raise GatewayError(
                "PROMPT_VARIABLE_MISSING",
                "Prompt variables are missing",
                status_code=422,
                details={
                    "prompt": reference.name,
                    "version": reference.version,
                    "missing_variables": missing,
                },
            )

        def replace(match: re.Match[str]) -> str:
            return self._stringify(reference.variables[match.group(1)])

        return prompt, VARIABLE_PATTERN.sub(replace, prompt.template)

    @staticmethod
    def _stringify(value: Scalar) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _validate_structured_output(request: UnifiedRequest, content: str) -> None:
        if request.response_format is None:
            return
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise GatewayError(
                "INVALID_STRUCTURED_OUTPUT",
                "Model response is not valid JSON",
                status_code=422,
            ) from exc
        if not isinstance(parsed, dict):
            raise GatewayError(
                "INVALID_STRUCTURED_OUTPUT",
                "Model response must be a JSON object",
                status_code=422,
            )

    def _can_retry(
        self,
        error: GatewayError,
        state: _CallState,
        *,
        stream_committed: bool = False,
    ) -> bool:
        return (
            error.retryable
            and not stream_committed
            and state.retries < self.settings.max_retries
        )

    def _retry_delay(self, retry_number: int) -> float:
        delay = self.settings.retry_base_delay * (2 ** (retry_number - 1))
        return min(delay, self.settings.retry_max_delay)

    async def _record_failure(
        self,
        invocation: PreparedInvocation,
        state: _CallState,
        error: GatewayError,
    ) -> None:
        await self._record(
            request_id=invocation.request_id,
            tenant_id=invocation.tenant_id,
            requested_model=invocation.requested_model,
            config=invocation.config,
            prompt_name=invocation.prompt_name,
            prompt_version=invocation.prompt_version,
            state=state,
            status="error",
            error=error,
            total_latency_ms=(self.clock() - invocation.started_clock) * 1000,
        )

    async def _record(
        self,
        *,
        request_id: str,
        tenant_id: str,
        requested_model: str,
        config: ModelConfig,
        prompt_name: str | None,
        prompt_version: int | None,
        state: _CallState,
        status: str,
        error: GatewayError | None,
        total_latency_ms: float,
    ) -> None:
        await self.database.record_call(
            request_id=request_id,
            tenant_id=tenant_id,
            requested_model=requested_model,
            resolved_model=config.upstream_model,
            provider=config.provider,
            protocol=config.protocol,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            status=status,
            error_code=error.code if error else None,
            usage=state.usage,
            ttft_ms=state.ttft_ms,
            total_latency_ms=total_latency_ms,
            attempt_count=state.attempts,
            retry_count=state.retries,
        )
