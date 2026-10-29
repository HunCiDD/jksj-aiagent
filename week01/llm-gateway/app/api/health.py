from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    database = getattr(request.app.state, "database", None)
    service = getattr(request.app.state, "gateway_service", None)
    is_ready = database is not None and service is not None
    return {
        "status": "ready" if is_ready else "starting",
        "database": "ready" if database is not None else "unavailable",
    }
