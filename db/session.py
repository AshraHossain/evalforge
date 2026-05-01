"""
db/session.py — Async SQLAlchemy engine & session factory

WHY THIS FILE EXISTS:
    Creates the engine once and exposes `get_db()` — an async generator FastAPI
    uses as a dependency injection target.  Every router function that needs the
    database just declares `db: AsyncSession = Depends(get_db)`.

    RELATIONSHIP TO OTHER FILES:
    ┌─ db/session.py ─────────────────────────────────────────────────────────┐
    │  Exports: engine, AsyncSessionLocal, get_db                             │
    │  Used by:                                                               │
    │    api/routers/*.py         ← FastAPI Depends(get_db)                  │
    │    memory/store.py          ← direct async session usage               │
    │    db/migrations/env.py     ← Alembic uses engine for migrations       │
    └─────────────────────────────────────────────────────────────────────────┘

    WHY ASYNC:
    FastAPI is async-first.  Using an async SQLAlchemy session means DB queries
    don't block the event loop — the server can handle other WebSocket messages
    while waiting for a slow Postgres query.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

# create_async_engine uses asyncpg under the hood (declared in DATABASE_URL).
# pool_pre_ping=True sends a cheap "SELECT 1" before each connection to detect
# stale connections after a Postgres restart.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,         # Set True to log all SQL — useful for debugging
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Don't re-fetch attributes after commit
)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency.  Usage:

        @router.get("/targets")
        async def list_targets(db: AsyncSession = Depends(get_db)):
            ...

    The `async with` ensures the session is always closed even if an exception
    is raised mid-route — no connection leaks.
    """
    async with AsyncSessionLocal() as session:
        yield session
