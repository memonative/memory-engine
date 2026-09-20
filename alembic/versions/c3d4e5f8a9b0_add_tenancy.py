"""Add multi-tenancy: tenants + api_keys tables, tenant_id everywhere

Revision ID: c3d4e5f8a9b0
Revises: b2c3d4e5f7a8
Create Date: 2026-04-19

Strategy:
  1. Create tenants + api_keys tables.
  2. Insert a deterministic 'default' tenant.
  3. Add nullable tenant_id to all six domain tables.
  4. Backfill every existing row to the default tenant.
  5. Set tenant_id NOT NULL + FK + indexes.
  6. Re-create composite indexes that previously keyed on user_id so
     they now lead with tenant_id (cheap tenant-scoped scans).

Existing single-tenant deployments keep working because their data is
silently relocated under the default tenant; the API_KEY env-var key
will be bound to that tenant on next app startup (see bootstrap step,
unrelated migration).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID


revision = 'c3d4e5f8a9b0'
down_revision = 'b2c3d4e5f7a8'
branch_labels = None
depends_on = None


# Deterministic so dev/staging/prod all share the same id for the
# implicit default tenant. Anything other than this is a real customer.
DEFAULT_TENANT_ID = '00000000-0000-0000-0000-000000000001'

DOMAIN_TABLES = (
    'memories',
    'memory_edges',
    'goals',
    'interactions',
    'reconsolidation_log',
    'memory_revisions',
)


def upgrade() -> None:
    # 1. Tenants
    op.create_table(
        'tenants',
        sa.Column('id', PGUUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('status', sa.String(20), nullable=False,
                  server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    # 2. API keys (sha256 hex of the bearer token)
    op.create_table(
        'api_keys',
        sa.Column('id', PGUUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', PGUUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_api_keys_tenant_id', 'api_keys', ['tenant_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])

    # 3. Seed the default tenant.
    op.execute(
        f"INSERT INTO tenants (id, name, status) VALUES "
        f"('{DEFAULT_TENANT_ID}', 'default', 'active') "
        f"ON CONFLICT (name) DO NOTHING"
    )

    # 4. Add tenant_id nullable, backfill, then enforce NOT NULL + FK.
    for table in DOMAIN_TABLES:
        op.add_column(
            table,
            sa.Column('tenant_id', PGUUID(as_uuid=True), nullable=True),
        )
        op.execute(
            f"UPDATE {table} SET tenant_id = '{DEFAULT_TENANT_ID}' "
            f"WHERE tenant_id IS NULL"
        )
        op.alter_column(table, 'tenant_id', nullable=False)
        op.create_foreign_key(
            f'fk_{table}_tenant_id',
            table, 'tenants',
            ['tenant_id'], ['id'],
            ondelete='CASCADE',
        )
        op.create_index(f'ix_{table}_tenant_id', table, ['tenant_id'])

    # 5. Re-create composite indexes so they lead with tenant_id.
    op.execute("DROP INDEX IF EXISTS uq_active_semantic_slot")
    op.execute(
        "CREATE UNIQUE INDEX uq_active_semantic_slot "
        "ON memories (tenant_id, user_id, attribute_slot) "
        "WHERE memory_type = 'semantic' "
        "AND decay_state IN ('active', 'fading') "
        "AND attribute_slot IS NOT NULL"
    )

    op.execute("DROP INDEX IF EXISTS ix_revisions_user_slot")
    op.execute(
        "CREATE INDEX ix_revisions_user_slot "
        "ON memory_revisions (tenant_id, user_id, attribute_slot, sequence_number)"
    )

    op.execute("DROP INDEX IF EXISTS ix_memories_user_decay")
    op.execute(
        "CREATE INDEX ix_memories_user_decay "
        "ON memories (tenant_id, user_id, decay_state)"
    )


def downgrade() -> None:
    # Reverse composite indexes.
    op.execute("DROP INDEX IF EXISTS ix_memories_user_decay")
    op.execute(
        "CREATE INDEX ix_memories_user_decay "
        "ON memories (user_id, decay_state)"
    )

    op.execute("DROP INDEX IF EXISTS ix_revisions_user_slot")
    op.execute(
        "CREATE INDEX ix_revisions_user_slot "
        "ON memory_revisions (user_id, attribute_slot, sequence_number)"
    )

    op.execute("DROP INDEX IF EXISTS uq_active_semantic_slot")
    op.execute(
        "CREATE UNIQUE INDEX uq_active_semantic_slot "
        "ON memories (user_id, attribute_slot) "
        "WHERE memory_type = 'semantic' "
        "AND decay_state IN ('active', 'fading') "
        "AND attribute_slot IS NOT NULL"
    )

    for table in DOMAIN_TABLES:
        op.drop_index(f'ix_{table}_tenant_id', table_name=table)
        op.drop_constraint(f'fk_{table}_tenant_id', table, type_='foreignkey')
        op.drop_column(table, 'tenant_id')

    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_index('ix_api_keys_tenant_id', table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_table('tenants')
