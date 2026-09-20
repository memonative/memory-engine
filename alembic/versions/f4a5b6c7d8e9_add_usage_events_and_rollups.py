"""Add usage_events and daily_usage_rollups tables

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-06-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'f4a5b6c7d8e9'
down_revision = 'e3f4a5b6c7d8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'usage_events',
        sa.Column('id', UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column('tenant_id', UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('api_key_id', UUID(as_uuid=True), nullable=True),
        sa.Column('endpoint', sa.String(100), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), nullable=True),
        sa.Column('memories_written', sa.Integer, nullable=False, server_default='0'),
        sa.Column('memories_retrieved', sa.Integer, nullable=False, server_default='0'),
        sa.Column('response_time_ms', sa.Integer, nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_usage_events_tenant_id', 'usage_events', ['tenant_id'])
    op.create_index('ix_usage_events_tenant_created', 'usage_events', ['tenant_id', 'created_at'])

    op.create_table(
        'daily_usage_rollups',
        sa.Column('id', UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column('tenant_id', UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('endpoint', sa.String(100), nullable=False),
        sa.Column('request_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('total_memories_written', sa.Integer, nullable=False, server_default='0'),
        sa.Column('total_memories_retrieved', sa.Integer, nullable=False, server_default='0'),
        sa.Column('avg_response_time_ms', sa.Integer, nullable=False, server_default='0'),
    )
    op.create_index('uq_daily_rollup', 'daily_usage_rollups', ['tenant_id', 'date', 'endpoint'], unique=True)

    # RLS on both tables
    for table in ('usage_events', 'daily_usage_rollups'):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id::text = current_setting('app.current_tenant_id', true))"
        )
        op.execute(
            f"CREATE POLICY worker_bypass ON {table} "
            f"USING (current_setting('app.current_tenant_id', true) = '__worker__')"
        )


def downgrade() -> None:
    for table in ('daily_usage_rollups', 'usage_events'):
        op.execute(f"DROP POLICY IF EXISTS worker_bypass ON {table}")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.drop_table('daily_usage_rollups')
    op.drop_table('usage_events')
