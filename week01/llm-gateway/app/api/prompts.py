from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_gateway_key
from app.schemas import (
    PromptCreateRequest,
    PromptResponse,
    PromptVersionCreateRequest,
)


router = APIRouter(
    prefix="/v1/prompts",
    tags=["prompts"],
    dependencies=[Depends(require_gateway_key)],
)


@router.post("", response_model=PromptResponse, status_code=201)
async def create_prompt(body: PromptCreateRequest, request: Request) -> PromptResponse:
    prompt = await request.app.state.database.create_prompt(body.name, body.template)
    return PromptResponse.model_validate(prompt)


@router.post("/{name}/versions", response_model=PromptResponse, status_code=201)
async def create_prompt_version(
    name: str,
    body: PromptVersionCreateRequest,
    request: Request,
) -> PromptResponse:
    prompt = await request.app.state.database.create_prompt_version(name, body.template)
    return PromptResponse.model_validate(prompt)


@router.get("/{name}/versions/{version}", response_model=PromptResponse)
async def get_prompt(name: str, version: int, request: Request) -> PromptResponse:
    prompt = await request.app.state.database.get_prompt(name, version)
    return PromptResponse.model_validate(prompt)
