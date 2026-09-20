"""Add Row-Level Security policies on tenant-scoped tables

Revision ID: c1d2e3f4a5b6
Revises: b0c1d2e3f4a5
Create Date: 2026-05-31

Defence-in-depth: even if application code forgets a WHERE
tenant_id = ... filter, RLS prevents cross-tenant data leaks.

Policies use current_setting('app.current_tenant_id', true) which
the application sets per-connection via set_config(). FORCE ROW
LEVEL SECURITY ensures policies apply even to the table owner role,
so a misconfigured connection string cannot bypass isolation.

The worker role uses a dedicated bypass policy (see worker.py) that
sets app.current_tenant_id before querying, or operates cross-tenant
for global maintenance tasks (decay) which don't expose data to
external callers.
"""
from alembic import op


revision = 'c1d2e3f4a5b6'
down_revision = 'b0c1d2e3f4a5'
branch_labels = None
depends_on = None

TABLES = [
    'memories',
    'memory_edges',
    'goals',
    'interactions',
    'reconsolidation_log',
    'memory_revisions',
    'tenant_credentials',
    'api_keys',
]


def upgrade() -> None:
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id::text = current_setting('app.current_tenant_id', true))"
        )
    # The worker/migration role needs a bypass policy for cross-tenant
    # maintenance (decay, consolidation). Grant BYPASSRLS to the app role
    # is NOT safe — instead, the worker sets the tenant var per-tenant or
    # uses a superuser connection for global scans.
    # We create a permissive policy for the special '__worker__' sentinel
    # value so the decay job can scan all rows without per-tenant iteration.
    for table in TABLES:
        op.execute(f"DROP POLICY IF EXISTS worker_bypass ON {table}")
        op.execute(
            f"CREATE POLICY worker_bypass ON {table} "
            f"USING (current_setting('app.current_tenant_id', true) = '__worker__')"
        )


def downgrade() -> None:
    for table in reversed(TABLES):
        op.execute(f"DROP POLICY IF EXISTS worker_bypass ON {table}")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
