from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.errors import GatewayError
from app.schemas import TokenUsage


class Base(DeclarativeBase):
    pass


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer)
    template: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class LLMCall(Base):
    __tablename__ = "llm_calls"

    request_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(100), default="default")
    requested_model: Mapped[str] = mapped_column(String(100))
    resolved_model: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(100))
    protocol: Mapped[str] = mapped_column(String(100))
    prompt_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class Database:
    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def create_prompt(self, name: str, template: str) -> PromptTemplate:
        async with self.sessions() as session:
            existing = await session.scalar(select(PromptTemplate.id).where(PromptTemplate.name == name))
            if existing is not None:
                raise GatewayError(
                    "PROMPT_ALREADY_EXISTS",
                    f"Prompt {name} already exists; create a new version instead",
                    status_code=409,
                )
            prompt = PromptTemplate(name=name, version=1, template=template)
            session.add(prompt)
            await self._commit_or_conflict(session, "PROMPT_ALREADY_EXISTS")
            return prompt

    async def create_prompt_version(self, name: str, template: str) -> PromptTemplate:
        async with self.sessions() as session:
            current = await session.scalar(
                select(func.max(PromptTemplate.version)).where(PromptTemplate.name == name)
            )
            if current is None:
                raise GatewayError(
                    "PROMPT_NOT_FOUND",
                    f"Prompt {name} was not found",
                    status_code=404,
                    details={"prompt": name},
                )
            prompt = PromptTemplate(name=name, version=int(current) + 1, template=template)
            session.add(prompt)
            await self._commit_or_conflict(session, "PROMPT_VERSION_CONFLICT")
            return prompt

    async def get_prompt(self, name: str, version: int) -> PromptTemplate:
        async with self.sessions() as session:
            prompt = await session.scalar(
                select(PromptTemplate).where(
                    PromptTemplate.name == name,
                    PromptTemplate.version == version,
                )
            )
            if prompt is None:
                raise GatewayError(
                    "PROMPT_NOT_FOUND",
                    f"Prompt {name} version {version} was not found",
                    status_code=404,
                    details={"prompt": name, "version": version},
                )
            return prompt

    async def record_call(
        self,
        *,
        request_id: str,
        tenant_id: str,
        requested_model: str,
        resolved_model: str,
        provider: str,
        protocol: str,
        prompt_name: str | None,
        prompt_version: int | None,
        status: str,
        error_code: str | None,
        usage: TokenUsage,
        ttft_ms: float | None,
        total_latency_ms: float | None,
        attempt_count: int,
        retry_count: int,
    ) -> None:
        record = LLMCall(
            request_id=request_id,
            tenant_id=tenant_id,
            requested_model=requested_model,
            resolved_model=resolved_model,
            provider=provider,
            protocol=protocol,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            status=status,
            error_code=error_code,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            attempt_count=attempt_count,
            retry_count=retry_count,
        )
        async with self.sessions() as session:
            session.add(record)
            await session.commit()

    async def get_call(self, request_id: str) -> LLMCall:
        async with self.sessions() as session:
            record = await session.get(LLMCall, request_id)
            if record is None:
                raise GatewayError(
                    "CALL_NOT_FOUND",
                    f"Call {request_id} was not found",
                    status_code=404,
                    details={"request_id": request_id},
                )
            return record

    async def recent_calls(self, limit: int = 20) -> list[LLMCall]:
        async with self.sessions() as session:
            result = await session.scalars(
                select(LLMCall).order_by(LLMCall.created_at.desc()).limit(limit)
            )
            return list(result)

    @staticmethod
    async def _commit_or_conflict(session: AsyncSession, code: str) -> None:
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise GatewayError(
                code,
                "The requested Prompt version conflicts with an existing version",
                status_code=409,
            ) from exc
