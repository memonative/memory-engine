"""Add Memory.event_date for extraction-time resolved event date

Revision ID: a9b0c1d2e3f4
Revises: f7a8b9c0d1e2
Create Date: 2026-05-19

Adds a nullable DATE column for the resolved event date — when the
described event actually happened, distinct from when the user told
us about it. Extraction resolves relative phrases ("a month ago",
"last week") against the conversation date and writes the absolute
date here. NULL means the event date is the same as conversation
date (the default case for most memories).

The reader uses event_date for "how many days between X and Y" math;
created_at remains the row's insert/conversation timestamp.
"""
from alembic import op


revision = 'a9b0c1d2e3f4'
down_revision = 'f7a8b9c0d1e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE memories ADD COLUMN event_date DATE NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS event_date")
