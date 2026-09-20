"""Add engine_model to tenant_credentials

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-06-07
"""
from alembic import op
import sqlalchemy as sa

revision = 'e3f4a5b6c7d8'
down_revision = 'd2e3f4a5b6c7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'tenant_credentials',
        sa.Column('engine_model', sa.String(200), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('tenant_credentials', 'engine_model')
