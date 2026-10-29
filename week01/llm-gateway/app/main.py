from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.adapters.messages import MessagesAdapter
from app.adapters.responses import ResponsesAdapter
from app.api import calls, chat, health, prompts
from app.core.config import Settings
from app.core.errors import GatewayError
from app.core.router import ModelRouter
from app.database import Database
from app.rate_limit import ModelRateLimiter, TokenBucket
from app.services.gateway import GatewayService


logger = logging.getLogger("llm_gateway")


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database = Database(resolved_settings.database_url)
        await database.initialize()
        client = httpx.AsyncClient(transport=upstream_transport)

        adapters = {
            "openai_responses": ResponsesAdapter(
                resolved_settings.models["deepseek-v4-pro"],
                client,
            ),
            "anthropic_messages": MessagesAdapter(
                resolved_settings.models["deepseek-v4-flash"],
                client,
                anthropic_version=resolved_settings.anthropic_version,
                default_max_tokens=resolved_settings.default_max_tokens,
            ),
        }
        rate_limiter = ModelRateLimiter(
            {
                model.alias: TokenBucket(
                    model.rate_limit_rpm,
                    model.rate_limit_burst,
                )
                for model in resolved_settings.models.values()
            }
        )
        application.state.settings = resolved_settings
        application.state.database = database
        application.state.upstream_client = client
        application.state.gateway_service = GatewayService(
            resolved_settings,
            database,
            ModelRouter(resolved_settings.models),
            adapters,
            rate_limiter,
            sleep=sleep,
        )
        try:
            yield
        finally:
            await client.aclose()
            await database.close()

    application = FastAPI(
        title="LLM Unified Gateway",
        version="0.1.0",
        description="Two-protocol LLM gateway with prompt versions and observability.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @application.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex}")
        headers = {"x-request-id": request_id}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_payload(request_id),
            headers=headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex}")
        errors = [
            {
                "location": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=GatewayError(
                "INVALID_REQUEST",
                "Request validation failed",
                status_code=422,
                details={"errors": errors},
            ).to_payload(request_id),
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex}")
        logger.exception("Unhandled gateway error request_id=%s", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=GatewayError(
                "INTERNAL_ERROR",
                "An unexpected gateway error occurred",
                status_code=500,
            ).to_payload(request_id),
        )

    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(prompts.router)
    application.include_router(calls.router)
    return application


app = create_app()
