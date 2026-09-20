#!/usr/bin/env bash
set -euo pipefail

# Run as a one-shot release step (init container or pre-deploy job).
# Waits for Postgres to accept connections, then runs migrations.

echo "migrate: waiting for database..."

# Wait up to 30s for Postgres to be reachable
for i in $(seq 1 30); do
    if python -c "
import sys, asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from memonative.config import settings

async def check():
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text('SELECT 1'))
    finally:
        await engine.dispose()

asyncio.run(check())
" 2>/dev/null; then
        echo "migrate: database is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "migrate: ERROR — database not reachable after 30s" >&2
        exit 1
    fi
    sleep 1
done

echo "migrate: running alembic upgrade head..."
alembic upgrade head

echo "migrate: done"
