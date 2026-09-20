"""Add memory_revisions table

Revision ID: a1b2c3d4e5f6
Revises: f4f2d4e3c61b
Create Date: 2025-03-22

This migration adds the memory_revisions table which tracks the full
history of semantic memory changes — replacing the single compressed_content
field with a proper revision chain.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = 'a1b2c3d4e5f6'
down_revision = '7a82c4f1c911'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Safely create enum if it doesn't exist
    op.execute(
        "DO $$ BEGIN CREATE TYPE revisiontype AS ENUM "
        "('initial', 'retraction', 'temporal_update', 'enrichment'); "
        "EXCEPTION WHEN duplicate_object THEN null; END $$;"
    )

    revision_type_enum = postgresql.ENUM(
        'initial', 'retraction', 'temporal_update', 'enrichment',
        name='revisiontype',
        create_type=False,
    )

    # Create the memory_revisions table
    op.create_table(
        'memory_revisions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column('memory_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('memories.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('revision_type', revision_type_enum, nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('trigger_message', sa.Text(), nullable=True),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('attribute_slot', sa.String(50), nullable=True),
        sa.Column('sequence_number', sa.Integer(), nullable=False, default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    # Create the composite index for fast slot history queries
    op.create_index(
        'ix_revisions_user_slot',
        'memory_revisions',
        ['user_id', 'attribute_slot', 'sequence_number'],
    )


def downgrade() -> None:
    op.drop_index('ix_revisions_user_slot', table_name='memory_revisions')
    op.drop_table('memory_revisions')
    op.execute("DROP TYPE IF EXISTS revisiontype")
