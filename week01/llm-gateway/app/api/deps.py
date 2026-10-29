from __future__ import annotations

import secrets

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import GatewayError
from app.services.gateway import GatewayService


bearer_scheme = HTTPBearer(auto_error=False)


async def require_gateway_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    settings = request.app.state.settings
    supplied = credentials.credentials if credentials else ""
    if not supplied or not secrets.compare_digest(supplied, settings.gateway_api_key):
        raise GatewayError(
            "INVALID_GATEWAY_KEY",
            "A valid Gateway API key is required",
            status_code=401,
        )


def get_gateway_service(request: Request) -> GatewayService:
    return request.app.state.gateway_service


def get_tenant_id(request: Request) -> str:
    return request.headers.get("x-tenant-id", "default")[:100]
