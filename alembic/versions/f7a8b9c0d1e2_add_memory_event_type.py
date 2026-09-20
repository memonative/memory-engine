"""Add Memory.event_type for per-memory ADD/UPDATE/DELETE semantics

Revision ID: f7a8b9c0d1e2
Revises: e5f6a0b1c2d3
Create Date: 2026-05-19

Adds a per-memory event_type tag the reader can use to disambiguate
state changes vs static facts without inferring from prose. Complements
the existing MemoryRevision chain (within-slot history): event_type
sits on the memory row, revision_type stays on each revision.

Defaults all existing rows to 'none' so retrieval keeps working.
"""
from alembic import op


revision = 'f7a8b9c0d1e2'
down_revision = 'e5f6a0b1c2d3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TYPE memoryeventtype AS ENUM ('none', 'add', 'update', 'delete')"
    )
    op.execute(
        "ALTER TABLE memories ADD COLUMN event_type memoryeventtype "
        "NOT NULL DEFAULT 'none'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS event_type")
    op.execute("DROP TYPE IF EXISTS memoryeventtype")
