"""Add dashboard_users table for web UI authentication

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-05-31

Dashboard users authenticate via email+password to the web UI.
Each user maps to exactly one tenant (org). Separate from the
API-key auth used for programmatic /v1/* access.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID


revision = 'd2e3f4a5b6c7'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'dashboard_users',
        sa.Column('id', PGUUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', PGUUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.String(20), nullable=False,
                  server_default='owner'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index('ix_dashboard_users_tenant_id', 'dashboard_users', ['tenant_id'])
    op.create_index('ix_dashboard_users_email', 'dashboard_users', ['email'])


def downgrade() -> None:
    op.drop_index('ix_dashboard_users_email', table_name='dashboard_users')
    op.drop_index('ix_dashboard_users_tenant_id', table_name='dashboard_users')
    op.drop_table('dashboard_users')
