"""Add attribute_slot to memories

Revision ID: 7a82c4f1c911
Revises: f4f2d4e3c61b
Create Date: 2026-03-22 02:50:20.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a82c4f1c911'
down_revision = 'f4f2d4e3c61b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the column
    op.add_column(
        'memories',
        sa.Column('attribute_slot', sa.String(50), nullable=True),
    )

    # Create partial unique index: one active semantic per slot per user
    # This is the hard constraint that makes sprawl impossible
    op.execute("""
        CREATE UNIQUE INDEX uq_active_semantic_slot
        ON memories (user_id, attribute_slot)
        WHERE memory_type = 'semantic'
          AND decay_state IN ('active', 'fading')
          AND attribute_slot IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_active_semantic_slot")
    op.drop_column('memories', 'attribute_slot')
