"""Add tenant_credentials table for per-tenant encrypted BYOK keys

Revision ID: b0c1d2e3f4a5
Revises: a9b0c1d2e3f4
Create Date: 2026-05-31

Stores Fernet-encrypted LLM keys per tenant. Engine LLM key is fully
BYOK; embedding key is optional (platform absorbs cost or requires
an OpenAI key specifically for text-embedding-3-small).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = 'b0c1d2e3f4a5'
down_revision = 'a9b0c1d2e3f4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'tenant_credentials',
        sa.Column('id', UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column('tenant_id', UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('engine_api_key_enc', sa.Text(), nullable=True),
        sa.Column('engine_base_url', sa.String(500), nullable=True),
        sa.Column('embedding_api_key_enc', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_tenant_credentials_tenant_id', 'tenant_credentials', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('ix_tenant_credentials_tenant_id')
    op.drop_table('tenant_credentials')
