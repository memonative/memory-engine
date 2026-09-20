"""Add missing edgetype enum values, vector + lookup indexes

Revision ID: b2c3d4e5f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-04-18

Adds:
  - 'corrects' and 'enriches' to edgetype enum (the model emits these
    but the original migration never added them, so any insert with
    those values throws "invalid input value for enum edgetype")
  - HNSW index on memories.embedding for cosine_distance
  - btree on memories(user_id, decay_state) — every retrieval/write
    filters on these
  - btree on memory_edges(from_memory_id) — used by walk_associations
"""
from alembic import op


revision = 'b2c3d4e5f7a8'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE edgetype ADD VALUE IF NOT EXISTS 'corrects'")
    op.execute("ALTER TYPE edgetype ADD VALUE IF NOT EXISTS 'enriches'")

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_user_decay "
        "ON memories (user_id, decay_state)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memory_edges_from "
        "ON memory_edges (from_memory_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_memory_edges_from")
    op.execute("DROP INDEX IF EXISTS ix_memories_user_decay")
