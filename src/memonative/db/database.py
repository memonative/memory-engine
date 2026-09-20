from collections.abc import AsyncGenerator
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from memonative.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession, expire_on_commit=False)


async def set_rls_tenant(session: AsyncSession, tenant_id: UUID) -> None:
    """Set the Postgres session variable used by RLS policies.

    Uses set_config() with a bind parameter — eliminates any SQL
    injection surface even if tenant_id were ever a raw string.
    The third arg `true` scopes to the current transaction (= SET LOCAL).
    """
    await session.execute(
        text("SELECT set_config('app.current_tenant_id', :tid, true)"),
        {"tid": str(tenant_id)},
    )


# Backwards-compatible alias for any existing imports
_set_rls_tenant = set_rls_tenant


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
