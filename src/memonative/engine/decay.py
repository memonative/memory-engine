import math
from datetime import datetime, timezone
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory

@dataclass
class DecayResult:
    strength: float
    decay_state: str
    effective_half_life_hours: float

def compute_decay(
    half_life_hours: float | None,
    reinforcement_count: int,
    salience: float,
    last_accessed_at: datetime,
    now: datetime | None = None,
) -> DecayResult:
    """Compute current memory strength using Ebbinghaus decay.

    A NULL half-life means the memory is pinned. It reports full strength
    forever, which is also what keeps it above retrieval's strength floor —
    a pin that let its memory drop out of search would not be a pin.
    """
    if half_life_hours is None:
        return DecayResult(
            strength=1.0,
            decay_state="active",
            effective_half_life_hours=math.inf,
        )

    now = now or datetime.now(timezone.utc)
    
    # Handle timezone naive vs aware
    if last_accessed_at.tzinfo is None:
        last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        
    hours_elapsed = (now - last_accessed_at).total_seconds() / 3600.0

    # Half-life grows logarithmically with reinforcement
    effective_hl = half_life_hours * math.log2(reinforcement_count + 1)

    # Salience slows decay (high-emotion memories persist)
    effective_hl *= (1 + salience)

    # Exponential decay
    if effective_hl <= 0:
        strength = 0.0
    else:
        strength = math.exp(-0.693 * hours_elapsed / effective_hl)

    strength = max(0.0, min(1.0, strength))

    # Map to decay state
    if strength > 0.5:
        state = "active"
    elif strength > 0.2:
        state = "fading"
    elif strength > 0.05:
        state = "dormant"
    else:
        state = "archived"

    return DecayResult(
        strength=strength,
        decay_state=state,
        effective_half_life_hours=effective_hl,
    )

async def reinforce_memory(db: AsyncSession, memory: Memory) -> None:
    """Reinforce a memory after retrieval. Bumps count + last_accessed_at.

    Strength and decay_state are NOT reset here — let compute_decay derive
    them. The reinforcement_count bump already extends the effective
    half-life, which is the intended Ebbinghaus effect.

    We mutate the ORM instance directly rather than issuing a bulk UPDATE
    statement: a bulk UPDATE makes SQLAlchemy auto-expire every other
    Memory instance in the session, which then triggers a lazy reload on
    next attribute read — and async lazy-loads from inside an
    already-running coroutine raise MissingGreenlet.
    """
    memory.reinforcement_count = (memory.reinforcement_count or 0) + 1
    memory.last_accessed_at = datetime.now(timezone.utc)


async def reinforce_memories_bulk(
    db: AsyncSession, memory_ids: list[UUID]
) -> None:
    """Reinforce many memories in a single UPDATE.

    Called from retrieve_memories at the end of a read so the N retrieved
    memories don't produce N individual UPDATE round-trips. Raw SQL on
    purpose — SQLAlchemy's ORM bulk-update path auto-expires in-session
    Memory instances and the subsequent attribute access raises
    MissingGreenlet inside async sessions (see reinforce_memory).
    Raw ``text()`` sidesteps the expire-on-update behaviour entirely.
    """
    if not memory_ids:
        return
    await db.execute(
        text(
            "UPDATE memories "
            "SET reinforcement_count = reinforcement_count + 1, "
            "    last_accessed_at = now() "
            "WHERE id = ANY(:ids)"
        ),
        {"ids": [str(mid) for mid in memory_ids]},
    )
