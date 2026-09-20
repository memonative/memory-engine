import logging
import math
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Goal

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "i", "a", "an", "the", "to", "of", "and", "or", "in", "on",
    "is", "am", "are", "was", "were", "be", "for", "with", "my",
    "this", "that", "it", "as", "at", "by", "from",
})

# Require at least this many meaningful overlapping words for an
# update/complete/abandon to be considered a real match. Anything less
# could just as easily target an unrelated goal — better to no-op than
# silently mutate the wrong one.
_MIN_GOAL_MATCH_OVERLAP = 2


def _meaningful_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w and w not in _STOPWORDS}


def find_best_goal_match(goals: list[Goal], description: str) -> Goal | None:
    """Pick the goal that overlaps `description` most by meaningful words.

    Returns None when no goal shares enough words with the description —
    we never fall back to "the first active goal", which used to silently
    rewrite arbitrary goals when the LLM didn't supply a description.
    """
    if not goals or not description:
        return None

    desc_words = _meaningful_words(description)
    if not desc_words:
        return None

    best: Goal | None = None
    best_overlap = 0
    for g in goals:
        overlap = len(desc_words & _meaningful_words(g.description))
        if overlap > best_overlap:
            best_overlap = overlap
            best = g

    if best_overlap < _MIN_GOAL_MATCH_OVERLAP:
        return None
    return best

async def process_goal_update(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, update: dict | None
) -> UUID | None:
    if update is None:
        return None

    action = update.get("action")

    if action == "create":
        goal = Goal(
            tenant_id=tenant_id,
            user_id=user_id,
            description=update.get("description", ""),
            horizon=update.get("horizon", "immediate"),
            status="active",
            priority=5,
        )
        db.add(goal)
        await db.flush()
        return goal.id

    elif action in ("complete", "abandon", "update"):
        goals_result = await db.execute(
            select(Goal).where(
                Goal.tenant_id == tenant_id,
                Goal.user_id == user_id,
                Goal.status == "active",
            )
        )
        active_goals = goals_result.scalars().all()
        description = update.get("description", "")
        best_match = find_best_goal_match(active_goals, description)

        if best_match:
            if action == "complete":
                best_match.status = "completed"
            elif action == "abandon":
                best_match.status = "abandoned"
            else:
                best_match.description = update.get("description", best_match.description)
                best_match.updated_at = func.now()
            return best_match.id

        # No-op rather than mutate a random goal — but log it so we can
        # see when the LLM is asking for actions we can't fulfil.
        logger.info(
            "goal action=%s ignored: no match for description=%r among %d active goals",
            action, description, len(active_goals),
        )

    return None
