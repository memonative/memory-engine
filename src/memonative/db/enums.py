import enum

class MemoryType(str, enum.Enum):
    episodic = 'episodic'
    semantic = 'semantic'
    procedural = 'procedural'

class DecayState(str, enum.Enum):
    active = 'active'
    fading = 'fading'
    dormant = 'dormant'
    archived = 'archived'

class SourceType(str, enum.Enum):
    user_stated = 'user_stated'
    user_repeated = 'user_repeated'
    system_inferred = 'system_inferred'
    consolidated = 'consolidated'

class EdgeType(str, enum.Enum):
    temporal = 'temporal'
    causal = 'causal'
    contradicts = 'contradicts'
    consolidates = 'consolidates'
    associated = 'associated'
    corrects = 'corrects'
    enriches = 'enriches'

class ReconsolidationOutcome(str, enum.Enum):
    reinforced = 'reinforced'
    corrected = 'corrected'
    enriched = 'enriched'
    no_change = 'no_change'

class GoalHorizon(str, enum.Enum):
    immediate = 'immediate'
    short_term = 'short_term'
    long_term = 'long_term'

class GoalStatus(str, enum.Enum):
    active = 'active'
    paused = 'paused'
    completed = 'completed'
    abandoned = 'abandoned'

class RevisionType(str, enum.Enum):
    initial = "initial"           # First value ever stored
    retraction = "retraction"     # User corrected an error ("my mistake, I meant...")
    temporal_update = "temporal_update"  # Life change ("I moved to...", "I changed jobs")
    enrichment = "enrichment"     # Detail added (content expanded, not replaced)


class MemoryEventType(str, enum.Enum):
    """Per-memory event semantics surfaced to the reader.

    Complements MemoryRevision (within-slot value history) by tagging
    inter-memory causal relationships: this memory ADDs to a collection,
    UPDATEs a prior fact, DELETEs/retracts an earlier claim, or carries
    no event semantics at all (the default).
    """
    none = "none"          # Static fact, default — no event semantics
    add = "add"            # New instance of a collection / count / list
    update = "update"      # Supersedes a prior value (paired with revision chain when slot-based)
    delete = "delete"      # User gave up / sold / lost / cancelled the fact
