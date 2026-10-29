from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import require_gateway_key
from app.schemas import CallRecordResponse


router = APIRouter(
    prefix="/v1/calls",
    tags=["observability"],
    dependencies=[Depends(require_gateway_key)],
)


@router.get("", response_model=list[CallRecordResponse])
async def list_calls(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[CallRecordResponse]:
    records = await request.app.state.database.recent_calls(limit)
    return [CallRecordResponse.model_validate(record) for record in records]


@router.get("/{request_id}", response_model=CallRecordResponse)
async def get_call(request_id: str, request: Request) -> CallRecordResponse:
    record = await request.app.state.database.get_call(request_id)
    return CallRecordResponse.model_validate(record)
