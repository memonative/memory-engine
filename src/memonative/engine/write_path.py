import difflib
import json
import logging
import math
import re
import zlib
from uuid import UUID
from datetime import date, datetime, timezone
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory, MemoryEdge, MemoryRevision
from memonative.db.enums import MemoryType, RevisionType, SourceType, MemoryEventType

_VALID_MEMORY_TYPES = [t.value for t in MemoryType]
_VALID_SOURCE_TYPES = [t.value for t in SourceType]
_VALID_EVENT_TYPES = [t.value for t in MemoryEventType]

# Explicit alias map for LLM hallucinations that fuzzy-match poorly.
# Observed during eval: 'assistant'/'assistant_stated' for system-
# inferred facts (the LLM picks the literal speaker word over our
# enum); 'user_said' / 'inferred' for the same reason. Add new
# aliases here when eval logs surface unfixed drops.
_ENUM_ALIASES: dict[str, str] = {
    # source aliases
    "assistant": "system_inferred",
    "assistant_stated": "system_inferred",
    "assistant_said": "system_inferred",
    "system_stated": "system_inferred",
    "system": "system_inferred",
    "inferred": "system_inferred",
    "user": "user_stated",
    "user_said": "user_stated",
    # memory type aliases — sometimes the LLM confuses memory_type
    # with intent.primary (decide / plan / execute) or general English
    # words ("event", "experience"). Map the unambiguous ones.
    "event": "episodic",
    "experience": "episodic",
    "preference": "procedural",
    "exploratory": "episodic",
    # source values misplaced in the type field — the LLM puts
    # "system_inferred" or "user_stated" as memory_type instead of
    # source. Recover as episodic (the safe default for both).
    "system_inferred": "episodic",
    "user_stated": "episodic",
    "user_repeated": "episodic",
}


def _parse_event_date(value) -> date | None:
    """Coerce an LLM-emitted event_date into a date or None.

    Accepts ISO strings ("2023-08-14"), full ISO timestamps
    ("2023-08-14T12:00:00Z"), or already-typed date/datetime objects.
    Anything else returns None — the reader falls back to created_at,
    so a bad/missing event_date is recoverable.
    """
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() in ("null", "none", "unknown"):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    try:
        # ISO with time component
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_PAT = (
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec"
)

# YYYY-MM-DD and YYYY/MM/DD
_ISO_DATE_RE = re.compile(
    r"\b(?P<y>(?:19|20)\d{2})[-/](?P<m>0?[1-9]|1[0-2])[-/](?P<d>0?[1-9]|[12]\d|3[01])\b"
)
# "January 15, 2023" / "Jan 15 2023"
_WORD_DATE_RE = re.compile(
    rf"\b(?P<m>{_MONTH_PAT})\s+(?P<d>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:,\s*|\s+)(?P<y>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
# "15 January 2023" / "15 Jan, 2023"
_WORD_DATE_REV_RE = re.compile(
    rf"\b(?P<d>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?P<m>{_MONTH_PAT})(?:,\s*|\s+)(?P<y>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)

# Words that, when preceding a date, mean the date is an anchor / range
# boundary / future intent rather than the event date itself. We bail out
# when any of these appear within ~30 chars before the date.
_SKIP_PRECEDERS = (
    "since ", "from ", "until ", "before ", "after ", "between ",
    "will ", "plan ", "plans to ", "planning ", "scheduled ",
    "next ", "upcoming ", "by ",
)


def _backfill_event_date_from_content(content: str) -> date | None:
    """Conservatively pull event_date from content when the LLM missed it.

    Fires only when the content contains EXACTLY ONE absolute calendar
    date with a 4-digit year AND that date is not preceded by a marker
    that flips its meaning (e.g. "since 2020-01-01", "will travel on
    February 5, 2024"). Multiple dates → defer to the LLM (returns None).

    The LLM still handles relative phrases ("a month ago"), partial dates
    ("in 2023"), and holiday names. This catches the gap we measured:
    24% of memories carrying an unambiguous date in content shipped with
    event_date NULL.
    """
    if not content:
        return None

    # Collect (start, year, month, day) hits across all three formats.
    raw_hits: list[tuple[int, int, int, int]] = []

    for m in _ISO_DATE_RE.finditer(content):
        raw_hits.append((
            m.start(),
            int(m.group("y")), int(m.group("m")), int(m.group("d")),
        ))

    for m in _WORD_DATE_RE.finditer(content):
        mo = _MONTHS.get(m.group("m").lower())
        if mo is None:
            continue
        raw_hits.append((
            m.start(),
            int(m.group("y")), mo, int(m.group("d")),
        ))

    for m in _WORD_DATE_REV_RE.finditer(content):
        mo = _MONTHS.get(m.group("m").lower())
        if mo is None:
            continue
        raw_hits.append((
            m.start(),
            int(m.group("y")), mo, int(m.group("d")),
        ))

    if not raw_hits:
        return None

    # Distinct calendar dates only — the same span matched by two
    # patterns counts once.
    distinct: dict[tuple[int, int, int], int] = {}
    for start, y, mo, d in raw_hits:
        key = (y, mo, d)
        if key not in distinct or start < distinct[key]:
            distinct[key] = start
    if len(distinct) != 1:
        return None

    (y, mo, d), start = next(iter(distinct.items()))
    try:
        parsed = date(y, mo, d)
    except ValueError:
        return None

    window = content[max(0, start - 30):start].lower()
    for marker in _SKIP_PRECEDERS:
        if marker in window:
            return None

    return parsed


def _canonicalize_enum(value: str, valid: list[str], cutoff: float = 0.6) -> str | None:
    """Return the closest valid value, or None if nothing's close enough.

    Three layers of recovery:
      1. Already valid? Pass through.
      2. Known alias? Return the canonical mapping.
      3. Fuzzy-match within cutoff? Return the closest valid value.

    Layer 2 catches LLM mistakes where the literal English word
    ('assistant', 'user', 'event') is far from the canonical value's
    string form but unambiguously means it. Layer 3 catches typos and
    Unicode glitches ('epidodic', 'semantıc').
    """
    if not isinstance(value, str):
        return None
    lower = value.lower()
    if lower in valid:
        return lower
    aliased = _ENUM_ALIASES.get(lower)
    if aliased and aliased in valid:
        return aliased
    matches = difflib.get_close_matches(lower, valid, n=1, cutoff=cutoff)
    return matches[0] if matches else None
from memonative.engine.decay import reinforce_memory
from memonative.engine.scoring import compute_vector_similarity
from memonative.config import get_decay_profile
from memonative.engine.edges import create_temporal_edges

logger = logging.getLogger(__name__)


async def _lock_slot(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, attribute_slot: str
) -> None:
    """Take a Postgres advisory lock for (tenant, user, slot) for the
    rest of the current transaction.

    Two concurrent writes to the same slot would otherwise both pass the
    SELECT and then race the partial unique index — one of them would
    raise IntegrityError and abort the whole transaction. The advisory
    lock serializes them at the slot granularity without touching real
    rows. It auto-releases at COMMIT/ROLLBACK.

    pg_advisory_xact_lock takes two int4s; we derive them from a stable
    hash of the (tenant, user, slot) tuple. tenant_id is part of the key
    so two tenants writing to the same (user, slot) don't serialize.
    """
    key = f"{tenant_id}:{user_id}:{attribute_slot}".encode("utf-8")
    h = zlib.crc32(key)
    # Split into two signed int4s so the lock key is bounded
    k1 = (h >> 16) & 0xFFFF
    k2 = h & 0xFFFF
    await db.execute(
        text("SELECT pg_advisory_xact_lock(:k1, :k2)"),
        {"k1": k1, "k2": k2},
    )

# Slots that name HIGH-IDENTITY attributes — never redirect into them
# (a "User loves Mumbai food" memory must not collapse into the
# "location" slot just because Mumbai is shared). The LLM picks these
# correctly ~always; the canonicalizer's job is to catch drift on the
# long-tail custom slots, not to second-guess the protected six.
_PROTECTED_SLOTS = frozenset({
    "location",
    "name",
    "profession",
    "employer",
    "native_language",
    "relationship_status",
})


_LLM_TIEBREAKER_SYSTEM_PROMPT = """\
You are deciding if two memory facts describe the SAME attribute (one \
should supersede the other) or DIFFERENT attributes (they should \
coexist as separate facts).

The attribute does NOT have to belong to the user. It can belong to an \
event, a place, an object, or another person — anything the user could \
later report a different value for. "Where the book club meets" is one \
attribute even though it is not a property of the user.

SAME = both describe the same property/state/preference/quantity. The \
newer one would naturally REPLACE the older one as the current truth. \
Numeric updates count: "pre-approved for $350k" → "pre-approved for \
$400k" is SAME (mortgage amount updated). So do wholesale value \
replacements: the two values will often look nothing alike, because \
the whole point is that the value changed.

DIFFERENT = they're about different things, even if topically related. \
Two distinct facts that should both stay on the user's record. A \
qualifier that names a different subject ("home" wifi vs "cafe" wifi) \
makes them DIFFERENT.

Examples:
- "User pre-approved for $350,000 from Wells Fargo" vs "User pre-approved \
  for $400,000 from Wells Fargo" -> SAME (mortgage pre-approval amount, \
  updated value)
- "User has 30 dozen eggs in fridge" vs "User has 20 dozen eggs from \
  chicken coop" -> SAME (current egg inventory, updated count)
- "User wakes at 8:30 am on Saturdays" vs "User wakes at 7:30 am on \
  Saturdays" -> SAME (saturday wake time, updated)
- "Book club is at the Nag's Head on the 21st" vs "Book club venue is \
  at Rosa's" -> SAME (where the book club meets, moved)
- "Flight to Dublin on March 10 at 07:15" vs "Dublin flight is at \
  11:40" -> SAME (that flight's departure time, rescheduled)
- "User has $50k in savings" vs "User has $50k mortgage" -> DIFFERENT \
  (asset vs liability)
- "User lives in Paris" vs "User travels to Paris often" -> DIFFERENT \
  (residence vs travel pattern)
- "User loves pizza" vs "User loves pasta" -> DIFFERENT (two preferences)
- "User has 37 pre-1920 coins" vs "User has a 1915-S quarter" -> \
  DIFFERENT (the count vs an individual item)
- "The quarterly review is on March 19" vs "Priya owns the quarterly \
  review" -> DIFFERENT (the date vs the owner — same event, two \
  attributes)

CRITICAL: when uncertain, return DIFFERENT. False-positive merges \
corrupt data (two real facts collapse into one). False-negative \
non-merges are recoverable (status quo, no data loss).

Return ONLY a JSON object: {"verdict": "same"} or {"verdict": "different"}.\
"""


async def _llm_slots_match(
    llm_client,
    new_content: str,
    existing_content: str,
    new_slot: str,
    existing_slot: str,
) -> bool:
    """LLM tiebreaker for borderline slot canonicalization.

    Called only when embedding similarity is in the "maybe" zone (0.75
    to 0.90) — not confident enough to auto-merge, not weak enough to
    reject outright. The LLM judges whether the two contents describe
    the same underlying attribute. Defaults to False (no merge) on
    parse error or LLM failure — the safe direction.
    """
    prompt = (
        f"Memory A (slot={existing_slot}): {existing_content}\n"
        f"Memory B (slot={new_slot}): {new_content}"
    )
    try:
        response = await llm_client.complete(
            prompt=prompt,
            system_prompt=_LLM_TIEBREAKER_SYSTEM_PROMPT,
        )
        parsed = json.loads((response or "").strip())
        verdict = (parsed.get("verdict") or "").strip().lower()
        return verdict == "same"
    except Exception as exc:
        logger.warning(
            "slot tiebreaker LLM call failed (defaulting to no-merge): %s",
            exc,
        )
        return False


def _slot_name_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the word tokens in two snake_case slot names."""
    ta = {t for t in re.split(r"[^a-z0-9]+", a.lower()) if t}
    tb = {t for t in re.split(r"[^a-z0-9]+", b.lower()) if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def _canonicalize_slot(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    embedding: list[float],
    llm_slot: str,
    new_content: str,
    llm_client,
    auto_threshold: float = 0.90,
    maybe_floor: float = 0.75,
    name_overlap_floor: float = 0.34,
    max_name_candidates: int = 3,
) -> str:
    """Return the slot this write should land on.

    The LLM picks slot names per-conversation and sometimes drifts
    ("home_purchase" in one session, "mortgage_pre_approval" in the
    next) which silently breaks the revision chain.

    Candidates come from two independent signals, because either one
    alone has a blind spot:

    1. Content embedding similarity, in two zones:
      - sim >= auto_threshold (0.90): same attribute, auto-merge
      - maybe_floor <= sim < auto_threshold: LLM decides (qualifiers
        like "fridge"/"chicken coop" shift sim into this zone even
        when the attribute is the same)
      - sim < maybe_floor: content alone is not enough, see 2.

    2. Slot-NAME token overlap. Content similarity is systematically
       weakest in exactly the case canonicalization exists for: the
       bigger the value change, the less similar the content, so a
       real correction can land below maybe_floor and never be
       considered. "Book club is at the Nag's Head" vs "Book club
       venue is at Rosa's" is 0.51 — well under the floor — yet
       `book_club_location`/`book_club_venue` are plainly one
       attribute. The names stay similar precisely when the values do
       not, so the two signals fail in opposite situations.

    Name overlap only nominates a candidate; it never merges on its
    own. Every name-matched pair goes through the same conservative
    LLM tiebreaker, which returns "different" when uncertain.

    Guards:
      - Protected slots (location/name/...) are never overridden.
      - Never redirect TO a protected slot from a custom one.
      - LLM tiebreaker defaults to "different" on uncertainty so a
        misfire leaves the status quo rather than corrupting data.
    """
    if not embedding or not llm_slot:
        return llm_slot
    if llm_slot in _PROTECTED_SLOTS:
        return llm_slot

    result = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.attribute_slot.is_not(None),
            Memory.decay_state.in_(["active", "fading"]),
            Memory.embedding.is_not(None),
        ).order_by(
            Memory.embedding.cosine_distance(embedding)
        ).limit(3)
    )
    candidates = result.scalars().all()

    ruled_out: set[str] = set()
    for cand in candidates:
        if cand.attribute_slot == llm_slot:
            return llm_slot
        if cand.attribute_slot in _PROTECTED_SLOTS:
            continue
        if cand.embedding is None:
            continue
        sim = compute_vector_similarity(embedding, cand.embedding)
        if sim >= auto_threshold:
            logger.info(
                "slot canonicalised (auto): %r -> %r (sim=%.3f, against=%r)",
                llm_slot, cand.attribute_slot, sim,
                (cand.content or "")[:80],
            )
            return cand.attribute_slot
        if sim >= maybe_floor:
            same = await _llm_slots_match(
                llm_client,
                new_content=new_content,
                existing_content=cand.content or "",
                new_slot=llm_slot,
                existing_slot=cand.attribute_slot,
            )
            if same:
                logger.info(
                    "slot canonicalised (llm tiebreaker): %r -> %r "
                    "(sim=%.3f, against=%r)",
                    llm_slot, cand.attribute_slot, sim,
                    (cand.content or "")[:80],
                )
                return cand.attribute_slot
            # LLM said different — fall through to check the next
            # candidate (next-most-similar memory).
            ruled_out.add(cand.attribute_slot)

    # Signal 2: slots whose NAME looks like a rename of this one.
    # Only slots the tiebreaker has already rejected are excluded —
    # a candidate skipped for low content similarity is exactly the
    # case this signal exists to catch.
    name_rows = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.attribute_slot.is_not(None),
            Memory.attribute_slot.not_in(_PROTECTED_SLOTS | ruled_out | {llm_slot}),
            Memory.decay_state.in_(["active", "fading"]),
        )
    )
    scored = [
        (_slot_name_overlap(llm_slot, m.attribute_slot), m)
        for m in name_rows.scalars().all()
    ]
    scored = [(s, m) for s, m in scored if s >= name_overlap_floor]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    for overlap, cand in scored[:max_name_candidates]:
        same = await _llm_slots_match(
            llm_client,
            new_content=new_content,
            existing_content=cand.content or "",
            new_slot=llm_slot,
            existing_slot=cand.attribute_slot,
        )
        if same:
            logger.info(
                "slot canonicalised (name overlap): %r -> %r "
                "(overlap=%.2f, against=%r)",
                llm_slot, cand.attribute_slot, overlap,
                (cand.content or "")[:80],
            )
            return cand.attribute_slot

    return llm_slot


async def _matches_superseded_revision(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    embedding: list[float],
    threshold: float = 0.85,
) -> bool:
    """True if this content semantically restates a CLOSED slot revision.

    The new extraction prompt is more aggressive — long sessions that
    say "I used to wake at 8:30, now I wake at 7:30" sometimes emit
    BOTH facts. The 7:30 lands cleanly in a slot revision (current).
    The 8:30 echo is a non-slotted episodic that competes with the
    current value in retrieval and tricks the reader into picking the
    superseded fact (concrete example: dad224aa wake-up time).

    This guard catches the echo at write time. If the new memory's
    embedding is close to any superseded (valid_until IS NOT NULL)
    revision content for this user, we skip the write — the slot's
    revision chain already preserves the historical value with proper
    valid_from/valid_until bounds. Legitimate historical episodics
    that paraphrase the fact ('used to wake at 8:30 before my new job')
    stay below the threshold and write normally.
    """
    if not embedding:
        return False

    result = await db.execute(
        select(MemoryRevision).where(
            MemoryRevision.tenant_id == tenant_id,
            MemoryRevision.user_id == user_id,
            MemoryRevision.valid_until.is_not(None),
            MemoryRevision.embedding.is_not(None),
        ).order_by(
            MemoryRevision.embedding.cosine_distance(embedding)
        ).limit(3)
    )
    revisions = result.scalars().all()

    for rev in revisions:
        if rev.embedding is None:
            continue
        sim = compute_vector_similarity(embedding, rev.embedding)
        if sim >= threshold:
            return True
    return False


async def find_near_duplicate(
    db: AsyncSession, tenant_id: UUID, user_id: UUID,
    embedding: list[float], threshold: float = 0.95
) -> Memory | None:
    if not embedding:
        return None

    result = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.decay_state.in_(["active", "fading"]),
            Memory.embedding.is_not(None),
        ).order_by(
            Memory.embedding.cosine_distance(embedding)
        ).limit(1)
    )
    candidate = result.scalar_one_or_none()

    if candidate and candidate.embedding is not None:
        dot = sum(a * b for a, b in zip(embedding, candidate.embedding))
        norm_a = math.sqrt(sum(a * a for a in embedding))
        norm_b = math.sqrt(sum(b * b for b in candidate.embedding))
        if norm_a and norm_b:
            similarity = dot / (norm_a * norm_b)
            if similarity >= threshold:
                return candidate
    return None

class MemoryCandidate:
    def __init__(self, data: dict):
        self.content = data.get("content", "")
        self.type = data.get("type", "episodic")
        self.salience = data.get("salience", 0.5)
        self.context_tags = data.get("context_tags", [])
        self.source = data.get("source", "system_inferred")
        self.attribute_slot = data.get("attribute_slot", None)
        # NEW: correction_type from extraction
        self.correction_type = data.get("correction_type", None)

async def _get_next_sequence_number(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, attribute_slot: str
) -> int:
    """Get the next sequence number for a slot's revision chain."""
    result = await db.execute(
        select(func.coalesce(func.max(MemoryRevision.sequence_number), 0))
        .where(
            MemoryRevision.tenant_id == tenant_id,
            MemoryRevision.user_id == user_id,
            MemoryRevision.attribute_slot == attribute_slot,
        )
    )
    current_max = result.scalar()
    return current_max + 1

async def write_semantic_with_slot(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    content: str,
    attribute_slot: str,
    embedding: list[float],
    llm_client,
    source: str = "user_stated",
    salience: float = 0.7,
    context_tags: list[str] | None = None,
    correction_type: str | None = None,
    trigger_message: str | None = None,
    event_type: str = "none",
    event_date: date | None = None,
) -> tuple[UUID, dict | None]:
    """
    Write or update a semantic memory with an attribute slot.

    The slot determines identity — only one active memory per slot.
    When the slot already has a value, this function:
      1. Closes the previous revision (sets valid_until)
      2. Creates a new revision with the correction_type
      3. Updates the Memory row with the new content
      4. Creates a biographical episodic for associative retrieval

    The revision chain preserves WHY each change happened:
      - retraction: the old value was never true (user error)
      - temporal_update: the old value was true but has changed
    """

    # Serialize concurrent writes to this slot before reading current
    # state, so the SELECT below and the eventual INSERT/UPDATE happen
    # atomically with respect to other requests touching the same slot.
    await _lock_slot(db, tenant_id, user_id, attribute_slot)

    # Check for existing active memory in this slot
    result = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.attribute_slot == attribute_slot,
            Memory.decay_state.in_(["active", "fading"]),
        ).limit(1)
    )
    existing = result.scalar_one_or_none()

    if existing:
        # --- SAME VALUE: reinforce ---
        if existing.content.strip().lower() == content.strip().lower():
            existing.reinforcement_count += 1
            existing.strength = 1.0
            existing.decay_state = "active"
            existing.trust_score = min(1.0, existing.trust_score + 0.05)
            return existing.id, None

        # --- DIFFERENT VALUE: correction with revision chain ---

        old_content = existing.content
        now = datetime.now(timezone.utc)

        # Determine the revision type
        if correction_type == "retraction":
            rev_type = RevisionType.retraction
            reason = "User corrected an error — previous value was incorrect"
        elif correction_type == "temporal_update":
            rev_type = RevisionType.temporal_update
            reason = "User's situation changed — previous value was valid until now"
        else:
            # Default to temporal_update (more conservative — preserves old as "was true")
            rev_type = RevisionType.temporal_update
            reason = "Value updated (correction type not specified)"

        next_seq = await _get_next_sequence_number(db, tenant_id, user_id, attribute_slot)

        # 1. Close the previous revision (if any exist)
        #    Find the open revision (valid_until IS NULL) for this slot
        prev_revision = await db.execute(
            select(MemoryRevision).where(
                MemoryRevision.memory_id == existing.id,
                MemoryRevision.attribute_slot == attribute_slot,
                MemoryRevision.valid_until.is_(None),
            ).order_by(MemoryRevision.sequence_number.desc()).limit(1)
        )
        prev_rev = prev_revision.scalar_one_or_none()

        if prev_rev:
            prev_rev.valid_until = now
        else:
            # No revision chain exists yet — create the initial revision retroactively
            initial_rev = MemoryRevision(
                tenant_id=tenant_id,
                memory_id=existing.id,
                user_id=user_id,
                content=old_content,
                embedding=existing.embedding,
                revision_type=RevisionType.initial,
                reason="Initial value (retroactively recorded)",
                valid_from=existing.created_at,
                valid_until=now,
                attribute_slot=attribute_slot,
                sequence_number=next_seq,
            )
            db.add(initial_rev)
            next_seq += 1

        # 2. Create the new revision
        new_revision = MemoryRevision(
            tenant_id=tenant_id,
            memory_id=existing.id,
            user_id=user_id,
            content=content,
            embedding=embedding,
            revision_type=rev_type,
            reason=reason,
            trigger_message=trigger_message,
            valid_from=now,
            valid_until=None,  # This is now the current truth
            attribute_slot=attribute_slot,
            sequence_number=next_seq,
        )
        db.add(new_revision)

        # 3. Create biographical episodic for associative retrieval
        #    This ensures the old value surfaces in relevant conversations
        combined_tags = list(set((context_tags or []) + ["biographical"]))

        # Content varies based on correction type
        if correction_type == "retraction":
            bio_content = f"Previously incorrectly stated: {old_content}"
        else:
            bio_content = old_content  # Historical fact, no qualification needed

        biographical = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type="episodic",
            content=bio_content,
            embedding=existing.embedding,  # Reuse — no new LLM call needed
            source="consolidated",
            trust_score=0.7 if correction_type == "retraction" else 0.9,
            salience=0.5,
            half_life_hours=720.0,
            context_tags=combined_tags,
            strength=0.8,
        )
        db.add(biographical)
        await db.flush()

        # 4. Create edge linking current memory to biographical
        db.add(MemoryEdge(
            tenant_id=tenant_id,
            from_memory_id=existing.id,
            to_memory_id=biographical.id,
            edge_type="temporal",
            weight=0.8,
            metadata_json={
                "reason": "biographical_demotion",
                "revision_type": rev_type.value,
                "slot": attribute_slot,
                "old_value": old_content,
            },
        ))

        correction_info = {
            "old_content": old_content,
            "new_content": content,
            "slot": attribute_slot,
            "correction_type": rev_type.value,
            "biographical_id": str(biographical.id),
        }

        # 5. Update the Memory row (the "current truth").
        # Old content lives in the revision chain; no need to also
        # stash it in compressed_content.
        existing.content = content
        existing.embedding = embedding
        existing.source = source
        existing.trust_score = 0.7
        existing.strength = 1.0
        existing.decay_state = "active"
        existing.reinforcement_count += 1
        # The slot's value changed. Respect an explicit extraction signal
        # (DELETE = user gave up the prior value; ADD = unusual but
        # honoured) — but when extraction emitted the default `none`,
        # promote to UPDATE so the reader sees that the prior revision
        # was superseded rather than just appearing as a static fact.
        if event_type == "none":
            existing.event_type = MemoryEventType.update
        else:
            existing.event_type = MemoryEventType(event_type)
        # event_date may be None (no resolved date for the new value),
        # in which case the reader falls back to created_at.
        if event_date is not None:
            existing.event_date = event_date

        return existing.id, correction_info

    else:
        # --- NEW SLOT: create fresh ---
        profile = get_decay_profile("semantic", source)

        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type="semantic",
            content=content,
            embedding=embedding,
            source=source,
            trust_score=profile.default_trust,
            salience=salience,
            half_life_hours=profile.half_life_hours,
            context_tags=context_tags or [],
            attribute_slot=attribute_slot,
            event_type=event_type,
            event_date=event_date,
        )
        db.add(memory)
        await db.flush()

        # Create the initial revision
        initial_rev = MemoryRevision(
            tenant_id=tenant_id,
            memory_id=memory.id,
            user_id=user_id,
            content=content,
            embedding=embedding,
            revision_type=RevisionType.initial,
            reason="First value for this attribute",
            valid_from=datetime.now(timezone.utc),
            valid_until=None,
            attribute_slot=attribute_slot,
            sequence_number=1,
        )
        db.add(initial_rev)

        return memory.id, None

async def write_memories(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    candidates_raw: list[dict],
    llm_client,
    trigger_message: str | None = None,
) -> tuple[list[UUID], list[UUID]]:
    """Write extracted memories to the database.

    Embeddings are batched into a single OpenAI call.
    Contradiction detection is offloaded to a Celery task
    (``check_contradictions_task``) — the LLM call adds 500–2000ms
    which we don't want on the user-facing path. The task re-fetches
    the memory so we don't ship ORM objects across the broker.

    No commit happens here. The caller owns the transaction boundary.
    """
    # Deferred import — avoids pulling Celery into request-path test
    # runs that don't touch the writer, and keeps the write_path ↔
    # worker boundary one-directional.
    from memonative.worker import check_contradictions_task
    # Filter empties and canonicalize enum values that the extraction
    # LLM near-missed. Letting raw bad values hit Postgres aborts the
    # whole transaction with InvalidTextRepresentationError; canonicalising
    # recovers the intended memory instead of silently dropping it.
    valid: list[tuple[int, dict]] = []
    for i, raw in enumerate(candidates_raw):
        if not raw.get("content"):
            continue

        mem_type = raw.get("type", "episodic")
        if mem_type not in _VALID_MEMORY_TYPES:
            fixed = _canonicalize_enum(mem_type, _VALID_MEMORY_TYPES)
            if fixed is None:
                logger.warning(
                    "dropping memory with unrecognised type=%r content=%r",
                    mem_type, (raw.get("content") or "")[:80],
                )
                continue
            logger.info(
                "canonicalised memory type %r -> %r", mem_type, fixed,
            )
            raw["type"] = fixed

        # attribute_slot is String(50) in the schema. The extraction LLM
        # occasionally returns a long descriptive phrase as a "slot",
        # which would crash the whole transaction with
        # StringDataRightTruncationError. Truncate to the column limit
        # so the rest of the candidates still write — the truncated
        # slot still uniquely identifies the attribute well enough.
        slot = raw.get("attribute_slot")
        if isinstance(slot, str) and len(slot) > 50:
            logger.warning(
                "truncating overlong attribute_slot=%r (len=%d)",
                slot[:80], len(slot),
            )
            raw["attribute_slot"] = slot[:50]

        source = raw.get("source", "system_inferred")
        if source not in _VALID_SOURCE_TYPES:
            fixed = _canonicalize_enum(source, _VALID_SOURCE_TYPES)
            if fixed is None:
                logger.warning(
                    "dropping memory with unrecognised source=%r content=%r",
                    source, (raw.get("content") or "")[:80],
                )
                continue
            logger.info(
                "canonicalised memory source %r -> %r", source, fixed,
            )
            raw["source"] = fixed

        # event_type is new and optional. Default to "none" when missing
        # (older code paths don't set it). Bad values get coerced to
        # "none" rather than dropping the memory — event_type is purely
        # additive context for the reader, not load-bearing for storage.
        event_type = raw.get("event_type") or "none"
        if event_type not in _VALID_EVENT_TYPES:
            fixed = _canonicalize_enum(event_type, _VALID_EVENT_TYPES)
            if fixed is None:
                logger.info(
                    "unknown event_type=%r, defaulting to 'none' for content=%r",
                    event_type, (raw.get("content") or "")[:80],
                )
                fixed = "none"
            raw["event_type"] = fixed
        else:
            raw["event_type"] = event_type

        # event_date is the resolved absolute date of the described
        # event. Parse the ISO string the LLM emitted; treat anything
        # unparseable as null (the reader falls back to created_at).
        ed_raw = raw.get("event_date")
        parsed_ed = _parse_event_date(ed_raw)
        if parsed_ed is None:
            parsed_ed = _backfill_event_date_from_content(raw.get("content") or "")
        raw["event_date"] = parsed_ed

        valid.append((i, raw))
    if not valid:
        return [], []

    contents = [raw["content"] for _, raw in valid]
    embeddings = await llm_client.embed_batch(contents)

    created_ids: list[UUID] = []
    contradictions_flagged: list[UUID] = []

    # Process slotted-semantic memories BEFORE non-slotted ones in the
    # same batch. Reason: Fix B (`_matches_superseded_revision`) is a
    # reactive check — it can only suppress a non-slotted echo if the
    # supersession has already happened. When extraction emits, in one
    # batch, both the slotted $400k pre-approval (which closes the
    # $350k initial revision) and a non-slotted $350k episodic, the
    # batch order decides whether Fix B catches it. Sorting slotted
    # first guarantees every revision chain is established before any
    # non-slotted check runs. Stable sort preserves relative order
    # within each group. Concrete failure this fixes: 852ce960.
    indexed = list(zip(valid, embeddings))
    indexed.sort(
        key=lambda pe: 0 if (
            pe[0][1].get("type") == "semantic"
            and pe[0][1].get("attribute_slot")
        ) else 1
    )

    for (_, raw), embedding in indexed:
        content = raw["content"]
        mem_type = raw.get("type", "episodic")
        attribute_slot = raw.get("attribute_slot")

        if mem_type == "semantic" and attribute_slot:
            # Redirect the LLM's slot to an existing semantically-equivalent
            # slot when the user already has one. Two-zone match: auto-merge
            # at sim>=0.90; LLM tiebreaker decides at 0.75-0.90 (catches
            # cases like egg_stock/egg_inventory where qualifier shifts
            # drop sim below the strict threshold but the attribute is
            # the same). The LLM defaults to "different" on uncertainty so
            # false-positive merges stay rare.
            canonical_slot = await _canonicalize_slot(
                db, tenant_id, user_id, embedding, attribute_slot,
                new_content=content,
                llm_client=llm_client,
            )
            mem_id, correction = await write_semantic_with_slot(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                content=content,
                attribute_slot=canonical_slot,
                embedding=embedding,
                llm_client=llm_client,
                source=raw.get("source", "user_stated"),
                salience=raw.get("salience", 0.7),
                context_tags=raw.get("context_tags", []),
                correction_type=raw.get("correction_type"),
                trigger_message=trigger_message,
                event_type=raw.get("event_type", "none"),
                event_date=raw.get("event_date"),
            )
            created_ids.append(mem_id)
            if correction:
                contradictions_flagged.append(mem_id)
                # A slot correction also creates a biographical episodic
                # capturing the old value. It's a new Memory row; surface
                # its id so callers (notably the eval adapter, which
                # backdates created_at to the session date) can see it.
                bio_id = correction.get("biographical_id")
                if bio_id:
                    created_ids.append(UUID(bio_id))
            # Queue async cross-slot contradiction detection. The slot
            # mechanism handles same-slot corrections inline; this covers
            # the case where the new value conflicts with a different
            # slot's existing fact (e.g. "lives in Paris" vs an older
            # "moved to Tokyo" episodic tagged biographical).
            check_contradictions_task.delay(
                str(tenant_id), str(user_id), str(mem_id),
            )
            continue

        # --- Non-slotted memories (episodic, procedural, slotless semantic) ---
        duplicate = await find_near_duplicate(
            db, tenant_id, user_id, embedding, threshold=0.95
        )
        if duplicate:
            await reinforce_memory(db, duplicate)
            if raw.get("source") == "user_stated":
                duplicate.source = "user_repeated"
                duplicate.trust_score = min(1.0, duplicate.trust_score + 0.15)
            created_ids.append(duplicate.id)
            continue

        # Suppress non-slotted memories that restate a superseded slot
        # value. The slot's revision chain already preserves the old
        # value with valid_from/valid_until bounds — an extra episodic
        # restating the old value as if it were current creates a
        # retrieval-vs-revision conflict the reader can't resolve
        # (concrete failure mode: dad224aa Saturday wake-up time).
        if await _matches_superseded_revision(
            db, tenant_id, user_id, embedding, threshold=0.85
        ):
            logger.info(
                "suppressing non-slotted memory restating superseded fact: %r",
                (content or "")[:80],
            )
            continue

        profile = get_decay_profile(mem_type, raw.get("source", "system_inferred"))

        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=mem_type,
            content=content,
            embedding=embedding,
            source=raw.get("source", "system_inferred"),
            trust_score=profile.default_trust,
            salience=raw.get("salience", 0.5),
            half_life_hours=profile.half_life_hours,
            context_tags=raw.get("context_tags", []),
            event_type=raw.get("event_type", "none"),
            event_date=raw.get("event_date"),
        )
        db.add(memory)
        await db.flush()

        await create_temporal_edges(db, tenant_id, user_id, memory.id)

        created_ids.append(memory.id)

        # Semantic facts without a slot still benefit from contradiction
        # detection against existing facts. Episodic/procedural memories
        # don't — they're not claims about the world, they're records.
        if mem_type == "semantic":
            check_contradictions_task.delay(
                str(tenant_id), str(user_id), str(memory.id),
            )

    return created_ids, contradictions_flagged

async def archive_stale_siblings(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    corrected_memory_id: UUID,
    new_embedding: list[float],
    similarity_threshold: float = 0.75,
) -> list[UUID]:
    """Archive semantically similar memories that may contain outdated info.

    FIXED: Now excludes biographical memories (tagged 'biographical')
    from archival. These are historical records, not stale duplicates.
    """
    result = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.decay_state.in_(["active", "fading"]),
            Memory.id != corrected_memory_id,
        )
    )
    siblings = result.scalars().all()

    archived_ids = []
    for sibling in siblings:
        # Skip biographical memories — they're history, not stale
        if "biographical" in (sibling.context_tags or []):
            continue
        if sibling.embedding is None or not new_embedding:
            continue

        dot = sum(a * b for a, b in zip(new_embedding, sibling.embedding))
        norm_a = sum(a * a for a in new_embedding) ** 0.5
        norm_b = sum(b * b for b in sibling.embedding) ** 0.5
        if norm_a and norm_b:
            similarity = dot / (norm_a * norm_b)
            if similarity >= similarity_threshold:
                sibling.decay_state = "archived"
                sibling.strength = 0.0
                archived_ids.append(sibling.id)

    return archived_ids


async def retract_memories(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    retractions: list[dict],
    llm_client,
    exclude_ids: list[UUID] | None = None,
    similarity_threshold: float = 0.78,
    candidates_per_retraction: int = 3,
) -> list[UUID]:
    """Archive memories matching each retraction description.

    The extraction LLM emits retractions when the user explicitly
    disowns a prior claim ("lol I was kidding", "scratch that", ...).
    For each retraction we embed the description, find the nearest
    active memories with pgvector, and archive any whose cosine
    similarity is above `similarity_threshold`.

    `exclude_ids` lets the caller pass the ids of memories just written
    in the same request, so a fresh "User loves Python" semantic isn't
    archived by a "User loves animals, specifically python" retraction.

    Each archived memory is preserved as a biographical episodic so the
    retraction itself becomes part of history rather than a silent erase.
    """
    if not retractions:
        return []

    descriptions = [
        r.get("description", "").strip()
        for r in retractions
        if r.get("description")
    ]
    descriptions = [d for d in descriptions if d]
    if not descriptions:
        return []

    excluded = set(exclude_ids or [])
    embeddings = await llm_client.embed_batch(descriptions)

    archived_ids: list[UUID] = []
    for description, embedding in zip(descriptions, embeddings):
        if not embedding:
            continue

        result = await db.execute(
            select(Memory).where(
                Memory.tenant_id == tenant_id,
                Memory.user_id == user_id,
                Memory.decay_state.in_(["active", "fading"]),
                Memory.embedding.is_not(None),
            ).order_by(
                Memory.embedding.cosine_distance(embedding)
            ).limit(candidates_per_retraction)
        )
        candidates = result.scalars().all()

        for candidate in candidates:
            if candidate.id in excluded:
                continue
            # Don't archive history — biographicals are the receipts of
            # past corrections/retractions, not stale facts.
            if "biographical" in (candidate.context_tags or []):
                continue

            sim = compute_vector_similarity(embedding, candidate.embedding)
            if sim < similarity_threshold:
                # candidates are ordered by distance, so once one falls
                # below threshold the rest will too
                break

            old_content = candidate.content
            candidate.decay_state = "archived"
            candidate.strength = 0.0
            archived_ids.append(candidate.id)

            if candidate.attribute_slot:
                now = datetime.now(timezone.utc)
                prev_result = await db.execute(
                    select(MemoryRevision).where(
                        MemoryRevision.memory_id == candidate.id,
                        MemoryRevision.attribute_slot == candidate.attribute_slot,
                        MemoryRevision.valid_until.is_(None),
                    ).order_by(MemoryRevision.sequence_number.desc()).limit(1)
                )
                prev_rev = prev_result.scalar_one_or_none()
                if prev_rev:
                    prev_rev.valid_until = now

                next_seq = await _get_next_sequence_number(
                    db, tenant_id, user_id, candidate.attribute_slot,
                )
                db.add(MemoryRevision(
                    tenant_id=tenant_id,
                    memory_id=candidate.id,
                    user_id=user_id,
                    content=old_content,
                    embedding=candidate.embedding,
                    revision_type=RevisionType.retraction,
                    reason=f"Retracted by user: {description}",
                    trigger_message=description,
                    valid_from=now,
                    valid_until=now,
                    attribute_slot=candidate.attribute_slot,
                    sequence_number=next_seq,
                ))

            bio_tags = list(set((candidate.context_tags or []) + ["biographical"]))
            db.add(Memory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type="episodic",
                content=f"Previously stated (retracted by user): {old_content}",
                embedding=candidate.embedding,
                source="consolidated",
                trust_score=0.6,
                salience=0.4,
                half_life_hours=720.0,
                context_tags=bio_tags,
                strength=0.8,
            ))

            logger.info(
                "retracted memory id=%s sim=%.3f description=%r",
                candidate.id, sim, description,
            )

    return archived_ids
