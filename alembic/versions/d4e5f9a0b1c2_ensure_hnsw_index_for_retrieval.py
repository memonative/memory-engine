"""Ensure HNSW index on memories.embedding for in-DB candidate generation

Revision ID: d4e5f9a0b1c2
Revises: c3d4e5f8a9b0
Create Date: 2026-04-25

retrieve_memories used to load every (tenant, user) memory into Python
and score in a loop. It now pushes candidate generation into Postgres
via ``ORDER BY embedding <=> :q LIMIT :k``, which only behaves as O(log N)
when an HNSW index with vector_cosine_ops exists.

A previous migration (b2c3d4e5f7a8) already added this index, but that
was before retrieval actually relied on it — so this migration re-asserts
the index with ``IF NOT EXISTS`` as a guarantee for any deployment that
somehow lost it (manual drop, partial restore, older fork). No-op on
healthy installs.
"""
from alembic import op


revision = 'd4e5f9a0b1c2'
down_revision = 'c3d4e5f8a9b0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # Intentionally does not drop the index — migration b2c3d4e5f7a8 owns
    # its lifecycle, and dropping it here would make downgrade of this
    # revision silently remove an index the earlier revision created.
    pass
