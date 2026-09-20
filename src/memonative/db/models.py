from datetime import date, datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, Computed, Date, Integer, String, Float, DateTime, ForeignKey, Text, JSON, text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
from memonative.db.enums import MemoryType, DecayState, SourceType, EdgeType, GoalHorizon, GoalStatus, RevisionType, MemoryEventType

Base = declarative_base()


class Tenant(Base):
    __tablename__ = 'tenants'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # active | suspended. String, not enum — adding states later shouldn't
    # require an Alembic enum-alter dance.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default='active')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


class ApiKey(Base):
    __tablename__ = 'api_keys'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    # sha256 hex of the plaintext bearer token. Plaintext is returned to
    # the caller exactly once at creation and never persisted.
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
    )
    label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )


class TenantCredential(Base):
    """Per-tenant encrypted LLM credentials for BYOK.

    Engine LLM key is fully BYOK (any OpenAI-compatible endpoint).
    Embedding key is NOT BYOK — pgvector index is dimension-locked to
    text-embedding-3-small; a different provider's embeddings are
    incompatible. The platform absorbs embedding cost or requires
    specifically an OpenAI key for embeddings.
    """
    __tablename__ = 'tenant_credentials'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, unique=True, index=True,
    )
    engine_api_key_enc: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    engine_base_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )
    engine_model: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )
    embedding_api_key_enc: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )


class UsageEvent(Base):
    __tablename__ = 'usage_events'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    api_key_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True,
    )
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True,
    )
    memories_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    memories_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    __table_args__ = (
        Index('ix_usage_events_tenant_created', 'tenant_id', 'created_at'),
    )


class DailyUsageRollup(Base):
    __tablename__ = 'daily_usage_rollups'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_memories_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_memories_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index('uq_daily_rollup', 'tenant_id', 'date', 'endpoint', unique=True),
    )


class DashboardUser(Base):
    __tablename__ = 'dashboard_users'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default='owner',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


class Memory(Base):
    __tablename__ = 'memories'

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    memory_type: Mapped[MemoryType] = mapped_column(nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    compressed_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(1536))

    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    reinforcement_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # NULL means pinned: the decay job skips the row and compute_decay
    # reports full strength. Set it through MemoryEngine.set_retention
    # rather than assigning here — pinning and unpinning each carry a
    # second fixup that a bare assignment silently skips.
    half_life_hours: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=72.0
    )
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    decay_state: Mapped[DecayState] = mapped_column(nullable=False, default=DecayState.active)

    trust_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    source: Mapped[SourceType] = mapped_column(nullable=False)
    salience: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    
    context_tags = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    attribute_slot: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    event_type: Mapped[MemoryEventType] = mapped_column(
        nullable=False, default=MemoryEventType.none,
        server_default=MemoryEventType.none.value,
    )
    # When the EVENT described in `content` actually happened, as
    # resolved at extraction time from the conversation. NULL means
    # "same as the conversation date" (i.e., the fact's truth-time
    # is when it was stated). Use this for "how many days between X
    # and Y" reasoning — created_at is when we wrote the row.
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)

    # Postgres-maintained full-text vector of `content`. Read-only from the
    # ORM's perspective — inserts/updates must not try to write it.
    content_tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )

Index(
    'uq_active_semantic_slot',
    Memory.tenant_id,
    Memory.user_id,
    Memory.attribute_slot,
    unique=True,
    postgresql_where=(
        (Memory.memory_type == 'semantic')
        & (Memory.decay_state.in_(['active', 'fading']))
        & (Memory.attribute_slot.isnot(None))
    ),
)

class MemoryEdge(Base):
    __tablename__ = 'memory_edges'

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    from_memory_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey('memories.id', ondelete='CASCADE'), nullable=False)
    to_memory_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey('memories.id', ondelete='CASCADE'), nullable=False)
    edge_type: Mapped[EdgeType] = mapped_column(nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    metadata_json = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Goal(Base):
    __tablename__ = 'goals'

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    horizon: Mapped[GoalHorizon] = mapped_column(nullable=False)
    status: Mapped[GoalStatus] = mapped_column(nullable=False, default=GoalStatus.active)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class Interaction(Base):
    __tablename__ = 'interactions'

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    session_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent = mapped_column(JSONB, nullable=False)
    memories_written = mapped_column(ARRAY(PGUUID(as_uuid=True)), server_default="{}")
    memories_retrieved = mapped_column(ARRAY(PGUUID(as_uuid=True)), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

class ReconsolidationLog(Base):
    __tablename__ = 'reconsolidation_log'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    memory_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('memories.id', ondelete='CASCADE'),
        nullable=False,
    )
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False
    )
    outcome: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # reinforced, corrected, enriched
    original_content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    trust_before: Mapped[float] = mapped_column(Float, nullable=False)
    trust_after: Mapped[float] = mapped_column(Float, nullable=False)
    trigger_message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now()
    )

class MemoryRevision(Base):
    """
    Tracks the full history of a semantic memory's content changes.

    Human memory analogy:
      You don't just know you live in Mumbai. You know you USED to live
      in Delhi, and BEFORE that you incorrectly thought you'd said Paris.
      Each of those is a revision — with a reason.

    The Memory row always holds the current truth.
    This table holds the autobiography.
    """
    __tablename__ = 'memory_revisions'

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True,
        default=uuid4, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False, index=True,
    )
    memory_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('memories.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False
    )
    # The content at this point in time
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1536), nullable=True)

    # What kind of change this was
    revision_type: Mapped[RevisionType] = mapped_column(nullable=False)

    # Human-readable reason ("user corrected an error", "user relocated")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The message that triggered this revision
    trigger_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # When this value was the active truth
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True  # NULL = still current
    )

    # The attribute slot this revision belongs to (denormalised for fast queries)
    attribute_slot: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )

    # Ordering within a slot's history
    sequence_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

Index(
    'ix_revisions_user_slot',
    MemoryRevision.tenant_id,
    MemoryRevision.user_id,
    MemoryRevision.attribute_slot,
    MemoryRevision.sequence_number,
)
