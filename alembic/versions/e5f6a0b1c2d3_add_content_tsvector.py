"""Add generated tsvector column + GIN index on memories.content

Revision ID: e5f6a0b1c2d3
Revises: d4e5f9a0b1c2
Create Date: 2026-04-25

Pure vector search misses exact-token queries (order IDs, SKUs, error
codes, names). Adding a Postgres-generated ``content_tsv`` column lets
us BM25-equivalent-rank with ``ts_rank_cd``, and fuse those hits with
the vector candidates via RRF in retrieve_memories.

We use ``GENERATED ALWAYS AS ... STORED`` so the column stays in sync
with content automatically — no trigger to maintain, no write path to
thread the tsvector through.
"""
from alembic import op


revision = 'e5f6a0b1c2d3'
down_revision = 'd4e5f9a0b1c2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE memories ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute(
        "CREATE INDEX ix_memories_content_tsv "
        "ON memories USING GIN(content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_content_tsv")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS content_tsv")
