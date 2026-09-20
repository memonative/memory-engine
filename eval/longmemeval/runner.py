"""Runner: orchestrates ingest + query + answer generation per question
per system, streaming results to JSONL.

One question at a time, both sessions ingested fresh per question
(reset_user before each), to keep haystacks isolated. Results stream
to disk so a crash mid-run loses at most one question — restart skips
question_ids already on disk.

Cost guard: a $15 default kill-switch reads the shared CostTracker
between questions and halts cleanly when exceeded. Error guard halts
if a system's error rate exceeds 20% over at least 10 attempts
(small-sample noise is otherwise too easy to trip).

Usage:
    python -m eval.longmemeval.runner --n 100 --systems memonative,mem0
    python -m eval.longmemeval.runner --n 50 --run-id smoke
    python -m eval.longmemeval.runner --resume        # picks up last run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from memonative.config import settings as _settings
# mem0 + openai both read os.environ["OPENAI_API_KEY"] directly.
# Bridge it from .env-driven settings before any client is constructed.
if not os.environ.get("OPENAI_API_KEY") and _settings.OPENAI_API_KEY.get_secret_value():
    os.environ["OPENAI_API_KEY"] = _settings.OPENAI_API_KEY.get_secret_value()

from eval.longmemeval.cost_tracker import install_tracker, get_tracker
install_tracker()  # patch SDK before any adapter constructs its OpenAI client

from openai import AsyncOpenAI

from eval.longmemeval.adapter import MemoryAdapter
from eval.longmemeval.dataset import Example, load_longmemeval_s

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = DATA_DIR / "runs"

ANSWER_SYSTEM_PROMPT = """\
You are a personal assistant answering a question about a user using \
ONLY the retrieved memories below. Be direct and concise.

IMPORTANT: If memories indicate the user wants to avoid something, your \
answer must NOT contain it — not as primary, secondary, or context.

IMPORTANT: If memories contain the numbers/dates needed to compute the \
answer (counts to sum, ages to subtract, dates to diff), DO the \
computation. NEVER abstain when the raw data exists, even when scattered \
across different memories.

IMPORTANT: Pay close attention to the EXACT entity in the question. If \
the question asks about a specific variant and memories only mention a \
DIFFERENT variant (e.g., "electric guitar" vs "acoustic guitar", \
"Sales Manager" vs "Senior Sales Engineer"), abstain — these are \
talking about different things.

IMPORTANT: For comparison/savings questions, BOTH values must come from \
USER-stated facts (or user-relayed). If only one side has a user-stated \
value, abstain.

Before answering, reason step-by-step inside <thinking> tags:
- List every memory relevant to the question along with its dates — \
use the Event date when present, otherwise the Conversation date as a \
fallback — and any event tag (ADD/UPDATE/DELETE).
- For counting: enumerate each item with its date. Apply the question's \
EXACT verb/qualifier (e.g., "completed" = finished only, not planned; \
"LED" = leader only). Do a SECOND full scan of all memories after the \
initial count — items at low-ranked positions are commonly missed. \
Treat a base count plus an ADD-tagged memory on a later date as \
(count + 1).
- For cross-topic computation: scan ALL memories for each needed fact \
independently — they may be in unrelated conversations. List (a) what \
you need, (b) where each appears, (c) the computation.
- For temporal questions: USE THE EVENT DATE of each memory for date \
arithmetic (it's the resolved date the user is referring to). Compute \
intervals between event dates, not between conversation dates. Be \
careful: "how many days ago did X when Y happened" means \
interval(X.event_date, Y.event_date), not interval(X, today).
- For "where is X" / "what is X's current value": pick the most recent \
memory and the slot's latest revision.
- State your conclusion. Only the text OUTSIDE the <thinking> tags is \
shown to the user.

Memory header tags you will see:
- [semantic — slot: <name>] — stable fact stored under an attribute \
slot. Multiple revisions of the same slot form a chain; the row shown \
is the current truth.
- [episodic] — events, experiences, transient states. No revision chain.
- [procedural] — workflow / interaction preferences.
- event=ADD — user acquired/added a new instance of an enumerable \
thing. Increment any count or list this applies to.
- event=UPDATE — this row superseded a prior value. Older revisions \
(in the revision history block) are NO LONGER current.
- event=DELETE — user gave up / sold / cancelled / lost the thing. The \
fact is no longer active.
- No event tag — static fact, no state change.

Two distinct dates appear on memories — DO NOT CONFUSE THEM:
- "Conversation date: YYYY-MM-DD" — when the user TOLD the assistant \
this fact. Always present.
- "Event date: YYYY-MM-DD" — when the described event ACTUALLY \
happened (the extraction resolved relative phrases like "a month ago" \
to absolute dates at write time). Shown ONLY when it differs from \
the conversation date.
- When only Conversation date appears, the event happened ~on that \
date (the user stated a present fact, or a same-day event).
- For any "how many days between X and Y" / "which happened first" / \
"how long ago did X" question, USE THE EVENT DATE when available — \
never use the Conversation date for the actual event arithmetic.

Rules:

1. USE ONLY the memories provided. Do not draw on outside knowledge or \
training data.

2. MOST RECENT WINS for conflicting values of the same fact. Exceptions: \
(a) memories about different people/contexts aren't conflicting; (b) for \
historical event dates, use the memory closest to the event; (c) for \
current counts, the latest value REPLACES earlier ones — don't sum unless \
the more recent memory carries event=ADD on a later date than the count.

3. REVISION HISTORY: when a semantic memory has a revision history \
block, all previous values WERE TRUE during their (valid_from → \
valid_until) windows. "What was X before Y?" requires looking at the \
history, not just the current value. A revision marked as a retraction \
means the prior value was never true — ignore it for "what was X" \
questions. A temporal_update means the prior value was true until the \
new one took over.

4. EVENT TAGS as ground truth: ADD increments a collection, UPDATE \
supersedes the prior value, DELETE retracts. Use these directly; do \
NOT re-derive state changes from prose when an event tag tells you.

5. COUNTING: scan ALL memories chronologically. Build a numbered list \
in <thinking> with date and one-line description. Deduplicate by \
matching dates and content. Items in a single memory count separately. \
A base count memory plus subsequent ADD-tagged memories on later dates \
= count + (number of ADDs). On the SAME date, a count already includes \
any "added X" mentioned in the same conversation.

6. ABSTAIN as "I don't know." ONLY when: (a) the topic is genuinely \
unmentioned in any memory; (b) the question asks about a specific \
event/entity and only a DIFFERENT one exists; (c) comparison/ordering \
needs two values and only one is present. Before abstaining, do a \
keyword scan of ALL memories — they're ranked by relevance, not \
chronologically, so the answer may sit lower in the list. If ANY \
memory mentions the topic, attempt an answer using what you have.

7. YES/NO and comparisons: "Did the user ever X?" with no matching \
memory → "No." For two-value comparisons, find both across all \
memories and compare directly.

8. ACTIONS vs INTENTIONS: a memory written as past-tense or explicitly \
completed = the action happened. Future-tense ("plans to", "will") = \
intention only. A plan with a specified date and no contradiction by a \
later memory → assume completed on that date.

9. USER FACTS vs ASSISTANT ADVICE: a memory starting "Assistant told \
user..." or "Assistant recommended..." is advice, not the user's \
experience. For personal questions ("what did I"), prefer user-stated \
memories. For factual lookups ("what is X"), assistant-stated facts are \
fine.

10. CONNECT MEMORIES across topics: facts needed for one computation \
are often spread across unrelated conversations. Search ALL memories \
for each fact independently.

11. BE CONCISE: after <thinking>, give one short phrase or sentence. No \
preamble, no hedging, no restating the question. If abstaining, output \
exactly "I don't know."\
"""


def stratified_sample(
    examples: list[Example], n: int, seed: int = 42,
) -> list[Example]:
    """Sample `n` examples while preserving question_type proportions.

    LongMemEval-S is unbalanced (multi-session: 133, single-session-
    preference: 30). Random sampling at small n drops categories
    entirely. Stratified sampling rounds per-type counts to keep each
    type represented.
    """
    if n >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    by_type: dict[str, list[Example]] = {}
    for ex in examples:
        by_type.setdefault(ex.question_type, []).append(ex)

    total = len(examples)
    out: list[Example] = []
    for qtype, exs in by_type.items():
        k = max(1, round(n * len(exs) / total))
        rng.shuffle(exs)
        out.extend(exs[:k])

    rng.shuffle(out)
    return out[:n]


def _format_memory_block(mem) -> str:
    """Format a single RetrievedMemory into a rich text block."""
    parts = []

    event_tag = ""
    event = getattr(mem, "event_type", "none") or "none"
    if event != "none":
        event_tag = f" event={event.upper()}"

    if mem.attribute_slot:
        header = f"[{mem.memory_type} — slot: {mem.attribute_slot}{event_tag}]"
    else:
        header = f"[{mem.memory_type}{event_tag}]"
    parts.append(header)

    parts.append(f"  Content: {mem.content}")

    # Conversation date: when the user told the assistant this. Always
    # shown when available.
    if mem.created_at:
        parts.append(f"  Conversation date: {mem.created_at}")

    # Event date: when the described event actually happened. Shown
    # only when the extractor resolved a distinct event date — for most
    # memories the event happened when the user said it (i.e., event
    # date = conversation date) so we omit the line to keep the block
    # short. When present and DIFFERENT from conversation date, the
    # reader should use this for date-arithmetic questions.
    ev_date = getattr(mem, "event_date", "") or ""
    if ev_date and ev_date != mem.created_at:
        parts.append(f"  Event date: {ev_date}")

    parts.append(f"  Trust: {mem.trust_score:.1f}")

    if mem.revision_history and len(mem.revision_history) > 1:
        parts.append("  Revision history (oldest to newest):")
        for rev in mem.revision_history:
            valid_range = ""
            if rev.get("valid_from"):
                end = rev.get("valid_until") or "present"
                valid_range = f" (valid {rev['valid_from']} to {end})"
            rev_type = rev.get("revision_type", "")
            type_label = f" [{rev_type}]" if rev_type != "initial" else ""
            parts.append(f"    - \"{rev['content']}\"{type_label}{valid_range}")

    return "\n".join(parts)


import tiktoken
_ENCODER = tiktoken.encoding_for_model("gpt-4o")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


async def _generate_answer(
    llm: AsyncOpenAI,
    question: str,
    memories: list,
    question_date: str = "",
    timeline_block: str = "",
) -> tuple[str, int]:
    if not memories:
        memory_block = "(no memories available)"
    else:
        blocks = [_format_memory_block(m) for m in memories[:20]]
        memory_block = "\n\n".join(blocks)

    memory_tokens = _count_tokens(memory_block)

    date_hint = ""
    if question_date:
        from memonative.engine.query_expansion import detect_temporal_context
        ctx = detect_temporal_context(question)
        if ctx.has_temporal_marker:
            date_hint = (
                f"\nTemporal context: Today's date is {question_date}. "
                f"Use this to resolve any relative time references "
                f"(e.g. 'last week', 'X days ago').\n"
            )

    timeline_section = ""
    if timeline_block:
        timeline_section = f"\n{timeline_block}\n"
        memory_tokens += _count_tokens(timeline_block)

    user = (
        f"Retrieved memories:\n\n{memory_block}\n"
        f"{timeline_section}\n"
        f"{date_hint}"
        f"Question: {question}\nAnswer:"
    )
    # No temperature / top_p / penalty params here — DeepSeek defaults
    # thinking ON for v4-pro and silently ignores those sampling controls
    # in thinking mode. Thinking is exactly what we want for the reader
    # (counting distinct items, computing date gaps, ordering events),
    # so leave the default in place.
    response = await llm.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    # The reader prompt asks the model to reason inside <thinking> tags
    # and put the final answer after. Strip the reasoning before the
    # judge sees it — long chain-of-thought traces confuse the judge.
    return _extract_final_answer(raw), memory_tokens


import re as _re

_THINKING_RE = _re.compile(r"<thinking>.*?</thinking>", _re.DOTALL | _re.IGNORECASE)


def _extract_final_answer(raw: str) -> str:
    """Strip <thinking>...</thinking> blocks and return the final answer.

    The reader prompt instructs the model to reason inside <thinking> tags
    and put the user-facing answer after. Three failure modes the model
    may hit:
      1. Well-formed: <thinking>...</thinking>ANSWER — regex strips it.
      2. Unclosed open: <thinking>...ANSWER — take everything after the
         last open marker (the model started reasoning but never said
         "done thinking").
      3. Unopened close: ...thinking text...</thinking>ANSWER — take
         everything after the close marker.
    Tolerates tag variants (<think>, <mem_thinking>) in case the model
    drifts from the schema.
    """
    if not raw:
        return raw
    cleaned = _THINKING_RE.sub("", raw)
    for close in ("</thinking>", "</think>", "</mem_thinking>"):
        if close in cleaned:
            cleaned = cleaned.rsplit(close, 1)[-1]
    for opener in ("<thinking>", "<think>", "<mem_thinking>"):
        if opener in cleaned:
            cleaned = cleaned.rsplit(opener, 1)[-1]
    return cleaned.strip() or raw.strip()


def _load_done_ids(out_path: Path) -> set[str]:
    """Question_ids that finished successfully — those should be skipped
    on resume. Error rows are intentionally NOT in this set so the
    runner retries them after a code fix; if the bug is deterministic
    the error guard will halt the run, capping the blast radius."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                # Retried on next run.
                continue
            qid = row.get("question_id")
            if qid:
                done.add(qid)
    return done


async def run_adapter(
    adapter: MemoryAdapter,
    examples: list[Example],
    out_path: Path,
    *,
    cost_kill: float,
    error_kill_rate: float = 0.20,
    error_kill_min: int = 10,
    log_every: int = 5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids(out_path)
    if done:
        print(f"[{adapter.name}] resuming — {len(done)} already done")

    # Reader uses DeepSeek (OpenAI-compatible SDK with a base_url
    # override). The judge in scorer.py stays on OpenAI so accuracy
    # numbers remain comparable to prior runs and to mem0's published
    # results.
    llm = AsyncOpenAI(
        api_key=_settings.DEEPSEEK_API_KEY.get_secret_value(),
        base_url=_settings.DEEPSEEK_BASE_URL,
    )
    tracker = get_tracker()

    n_attempted = 0
    n_errors = 0
    started_at = time.time()

    with out_path.open("a", encoding="utf-8") as out_f:
        for i, ex in enumerate(examples):
            if ex.question_id in done:
                continue

            if tracker.total_cost >= cost_kill:
                print(
                    f"[{adapter.name}] COST KILL: "
                    f"${tracker.total_cost:.2f} >= ${cost_kill:.2f}"
                )
                break

            if n_attempted >= error_kill_min:
                rate = n_errors / n_attempted
                if rate >= error_kill_rate:
                    print(
                        f"[{adapter.name}] ERROR KILL: "
                        f"{rate:.1%} >= {error_kill_rate:.1%} "
                        f"({n_errors}/{n_attempted})"
                    )
                    break

            n_attempted += 1
            user_id = f"lme_{ex.question_id}"
            t0 = time.time()
            row: dict

            try:
                # Always reset first — ensures rerun on resume starts
                # from a clean haystack (previous partial run may have
                # left memories behind).
                await adapter.reset_user(user_id)

                for s in ex.sessions:
                    await adapter.ingest(
                        user_id, s.session_id, s.date, s.turns
                    )

                retrieved = await adapter.query(
                    user_id, ex.question, top_k=20,
                )

                # Optional second pass: for inventory / sequence questions,
                # the Memonative adapter returns a chronological context
                # block keyed on an LLM-extracted entity. mem0 doesn't
                # implement this — the runner gracefully degrades.
                timeline_block = ""
                qtb = getattr(adapter, "query_timeline_block", None)
                if qtb is not None:
                    timeline_block = await qtb(user_id, ex.question)

                generated, memory_tokens = await _generate_answer(
                    llm, ex.question, retrieved,
                    question_date=ex.question_date,
                    timeline_block=timeline_block,
                )

                row = {
                    "question_id": ex.question_id,
                    "question_type": ex.question_type,
                    "question": ex.question,
                    "gold_answer": ex.answer,
                    "answer_session_ids": sorted(ex.answer_session_ids),
                    "retrieved": [
                        {
                            "content": r.content,
                            "session_id": r.source_session_id,
                            "score": r.score,
                        }
                        for r in retrieved
                    ],
                    "generated_answer": generated,
                    "memory_tokens": memory_tokens,
                    "elapsed_s": round(time.time() - t0, 2),
                    "cost_so_far_usd": round(tracker.total_cost, 4),
                }
            except Exception as e:
                n_errors += 1
                row = {
                    "question_id": ex.question_id,
                    "question_type": ex.question_type,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 2),
                    "cost_so_far_usd": round(tracker.total_cost, 4),
                }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % log_every == 0 or n_attempted == 1:
                elapsed = time.time() - started_at
                rate_s = n_attempted / elapsed if elapsed else 0
                print(
                    f"[{adapter.name}] "
                    f"{n_attempted} attempted, {n_errors} errors, "
                    f"${tracker.total_cost:.2f} spent, "
                    f"{rate_s:.2f} q/s",
                    flush=True,
                )


async def main_async(args: argparse.Namespace) -> None:
    examples = load_longmemeval_s()
    if args.question_types:
        wanted = set(args.question_types.split(","))
        examples = [e for e in examples if e.question_type in wanted]
    if args.question_ids:
        # Explicit-id mode: bypass stratified sampling entirely. Used for
        # targeted re-runs where we want to confirm specific fixes (e.g.
        # the previously-failing question set) without re-ingesting the
        # full sample. Order in the JSONL matches the order on the CLI.
        wanted_ids = [qid.strip() for qid in args.question_ids.split(",") if qid.strip()]
        by_id = {e.question_id: e for e in examples}
        missing = [qid for qid in wanted_ids if qid not in by_id]
        if missing:
            print(f"WARN: unknown question_ids skipped: {missing}")
        sampled = [by_id[qid] for qid in wanted_ids if qid in by_id]
    else:
        sampled = stratified_sample(examples, args.n, seed=args.seed)

    print(f"Loaded {len(examples)} eligible examples; sampled {len(sampled)}")
    type_counts: dict[str, int] = {}
    for ex in sampled:
        type_counts[ex.question_type] = type_counts.get(ex.question_type, 0) + 1
    print(f"Stratified breakdown: {type_counts}")

    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the sample so the scorer reads exactly the same questions
    # the runner saw. JSONL of just question_ids — small, source-of-truth
    # for which examples were targeted.
    sample_path = run_dir / "sample.jsonl"
    if not sample_path.exists():
        with sample_path.open("w", encoding="utf-8") as f:
            for ex in sampled:
                f.write(json.dumps({
                    "question_id": ex.question_id,
                    "question_type": ex.question_type,
                }) + "\n")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems:
        print("ERROR: no systems specified")
        sys.exit(1)

    for system in systems:
        out_path = run_dir / f"{system}.jsonl"
        print(f"\n=== Running {system} -> {out_path} ===")

        adapter: MemoryAdapter
        if system == "memonative":
            from eval.longmemeval.adapters.memonative_adapter import (
                MemonativeAdapter,
            )
            adapter = MemonativeAdapter()
        elif system == "mem0":
            from eval.longmemeval.adapters.mem0_adapter import Mem0Adapter
            adapter = Mem0Adapter()
        else:
            print(f"unknown system: {system}")
            continue

        try:
            await run_adapter(
                adapter,
                sampled,
                out_path,
                cost_kill=args.cost_kill,
            )
        finally:
            await adapter.aclose()

    tracker = get_tracker()
    print(f"\n=== Run complete ===")
    print(f"Total cost: ${tracker.total_cost:.2f}")
    for model, s in sorted(tracker.by_model.items()):
        print(f"  {model}: {s.calls} calls, ${s.cost:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100,
                   help="number of questions (stratified sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", type=str, default="v1")
    p.add_argument("--systems", type=str, default="memonative,mem0")
    p.add_argument("--question-types", type=str, default="",
                   help="comma-separated filter (e.g. multi-session,knowledge-update)")
    p.add_argument("--question-ids", type=str, default="",
                   help="comma-separated question_ids to run (bypasses sampling). "
                        "Used for targeted re-runs of specific failures.")
    p.add_argument("--cost-kill", type=float, default=15.0,
                   help="halt when total OpenAI cost exceeds this (USD)")
    p.add_argument("--unbuffered", action="store_true",
                   help="flush prints immediately (default true; off for tests)")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
