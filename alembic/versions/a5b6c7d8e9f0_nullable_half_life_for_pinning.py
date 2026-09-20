"""Make Memory.half_life_hours nullable — NULL means pinned

Revision ID: a5b6c7d8e9f0
Revises: f4a5b6c7d8e9
Create Date: 2026-08-29

A pinned memory is one the user has told us never to forget, so it needs a
representation that the decay job cannot walk over. NULL is that
representation: `set_retention(memory_id, None)` writes it, and the decay
job's WHERE clause skips those rows entirely.

The alternative was a sentinel half-life large enough to outlive the user.
That stores a falsehood in a column other code does arithmetic on, and it
degrades rather than stops — a pin that merely fades very slowly is still a
pin that eventually breaks.

Existing rows keep their values; nothing is pinned by this migration.
"""
from alembic import op


revision = 'a5b6c7d8e9f0'
down_revision = 'f4a5b6c7d8e9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memories ALTER COLUMN half_life_hours DROP NOT NULL")


def downgrade() -> None:
    # Pinned rows have no half-life to restore, so they take the column
    # default. They stop being pinned, which is the honest outcome: the
    # concept does not exist in the schema being downgraded to.
    op.execute(
        "UPDATE memories SET half_life_hours = 72.0 WHERE half_life_hours IS NULL"
    )
    op.execute("ALTER TABLE memories ALTER COLUMN half_life_hours SET NOT NULL")
