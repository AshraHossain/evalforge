"""
db/migrations/env.py — Alembic runtime environment

WHY THIS FILE EXISTS:
    When you run `alembic upgrade head`, Alembic imports this file to find:
      1. The DB connection string (we read from settings, not hardcoded)
      2. The metadata of all models (so autogenerate can diff them)

    RELATIONSHIP TO OTHER FILES:
    ┌─ db/migrations/env.py ──────────────────────────────────────────────────┐
    │  Imports: db/models.py (Base.metadata), config.py (settings)           │
    │  If you add a new table to db/models.py, this file picks it up         │
    │  automatically because it imports Base.metadata.                       │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import models so Alembic sees them in metadata
from db.models import Base  # noqa: F401
from config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
