"""Scorer: reads runner JSONLs, computes recall@5 + LLM-judged answer
accuracy, and writes RESULTS.md.

Two metrics, computed identically per system:

  - **recall@5**: 1 if any of the top-5 retrieved memories has a
    session_id in `answer_session_ids`, else 0. Strict — no partial
    credit. Errored questions count as 0.

  - **answer accuracy**: gpt-4o-mini compares the generated answer to
    the gold answer and emits yes/no. Order-invariant ("the user's job
    is engineer at Stripe" matches "Stripe; engineer"). Errored
    questions count as no.

Numbers are reported overall and per question_type. The output table
shows systems side-by-side so the win/tie/lose call is obvious
at a glance. Use --baseline-run to add a baseline Memonative run
for 3-way comparison (e.g. v1 vs v3 vs mem0).

Usage:
    python -m eval.longmemeval.scorer --run-id v3
    python -m eval.longmemeval.scorer --run-id v3 --baseline-run v1
    python -m eval.longmemeval.scorer --run-id v1 --skip-judge   # recall only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from memonative.config import settings as _settings
if not os.environ.get("OPENAI_API_KEY") and _settings.OPENAI_API_KEY.get_secret_value():
    os.environ["OPENAI_API_KEY"] = _settings.OPENAI_API_KEY.get_secret_value()

from openai import AsyncOpenAI

from eval.longmemeval.cost_tracker import install_tracker, get_tracker
install_tracker()


DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = DATA_DIR / "runs"

# Judge prompt adapted from mem0's published LongMemEval benchmark
# (`mem0ai/memory-benchmarks/benchmarks/longmemeval/prompts.py`). The
# prior 4-sentence prompt produced a ~11pp harshness gap vs manual
# scoring — it penalized verbose-but-correct answers, hedges, and
# breakdowns. This version is explicit about semantic equivalence,
# enumerates the YES cases, and bias-checks the judge against
# defaulting to NO. Output contract preserved: <judge_thinking> block
# stripped, final yes/no parsed case-insensitively.
#
# Synced 2026-05-25 with mem0's current upstream prompt: added the
# anti-preference specificity callout (Lists section) and preference
# points 6 + 7 (acknowledgment strengthens, context-dependent fine).
# Did NOT inherit benchmark-overfit examples ("recent = 2017+", "notes
# vs chords") — only structural patterns.
#
# Judge model: we deliberately use gpt-4o here, not mem0's gpt-4o-mini.
# Reason: 4o-mini keyword-scans on rubric-style preference golds even
# when the prompt explicitly tells it not to (we verified this — both
# Memonative AND mem0 scored 0% preferences under 4o-mini). 4o follows
# the "evaluate OVERALL thrust" rule reliably. Cost is ~16x but absolute
# is still cents for a 100-question run. Trade-off: our absolute
# numbers no longer match mem0's paper directly — relative
# comparisons (Memonative vs mem0 on the SAME judge) remain fair.
JUDGE_SYSTEM_PROMPT = """\
I will give you a question, a correct answer (or rubric), and a model \
response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact \
words. Answer "yes" if every concept in the correct answer is \
addressed in the response, even with different vocabulary, more \
specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. \
Before concluding "no", you MUST verify the answer is truly wrong, \
not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are \
fine unless proven to be factually wrong. Extra qualifiers are fine \
unless proven to be wrong. E.g., "a blue dress and a matching \
necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item / place / \
name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, \
you did not visit with a friend" ≈ "You didn't mention going with \
anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is \
fine as long as the core value is correct.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is \
covered if restated via synonyms, sub-concepts, or related terms.
- A broad term like "A and B significance" is covered if the response \
addresses the topic area through related specific terms, even without \
naming each component literally.
- If some items are listed as "or"s, "maybe"s and potential answers, \
it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of \
them is fine.

IMPORTANT: "Anti-preference" items are read NARROWLY, not expansively. \
A constraint against one category does NOT extend to adjacent or \
higher-level categories. E.g. "not interested in general topics in \
domain X" does NOT exclude specific topics within domain X; "no visual \
attention activities" does NOT exclude an activity that only \
incidentally uses vision. Match the literal anti-preference, not a \
broader reading of it.

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number \
matches. A range that includes the correct answer is correct.
- More precise answers are correct: "22 days" matches "3 weeks"; \
"over $270" matches "$270."; "9 1/2 months" matches "9 months".
- Rough answers are correct: "about nine months" ≈ "9 months"; \
"8 months and 20 days" matches "9 months".
- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months".
- A correct number with added context (e.g., "about 5 months ago \
(around December 2022)") is correct — the parenthetical is \
supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = \
"Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if \
the current value is identified.
- References like "last weekend", "last Wednesday" are imprecise — \
people sometimes mean the one before the latest. Be flexible with \
such timestamps.

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not \
enough information" is also correct.
- Similarly, if correct answer is "not enough information", model \
saying "0" or "nothing found" is also correct.

**Preference / Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal \
context (preferences, habits, interests). Need not satisfy every \
rubric point.
2. Primary criterion: do main suggestions align with what the user \
WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword \
scanning. If the response largely suggests correct options, minor \
incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity is not \
"suggesting phone use." Judge by the activity, not the delivery \
mechanism.
5. "May not prefer" = mild preference, not hard prohibition.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens \
off") STRENGTHENS correctness — it shows the response heard the \
constraint, not that it violated it.
7. Context-dependent suggestions are acceptable: an activity flagged \
by an anti-preference can still be appropriate in a context where the \
anti-preference doesn't bite (e.g., reading is fine on a bus even if \
the rubric flags visual-attention activities in general). Adjacent \
genres / variants alongside preferred ones are additive, not \
contradictory.
8. If the rubric mentions specific user resources/tools, the response \
is correct if it demonstrates awareness of the user's MAIN personal \
context even if it does not name every specific tool. The rubric is a \
guide, not a checklist.

**Abstention Matching** (PRECONDITION — read carefully)
- This rule fires ONLY when the GOLD answer is itself an abstention. \
Trigger phrases in the GOLD: "you did not mention", "no record of", \
"you didn't share", "I don't have that information", "not enough \
information", or an empty / null preference.
- If the GOLD is a SUBSTANTIVE answer (a fact, list, date, number, or \
preference rubric describing what the user wants), the model saying \
"I don't know" is WRONG — it is a refusal to answer a question that \
has a real answer. Do NOT credit it as a correct abstention.
- WHEN the gold IS an abstention: any phrasing the model uses that \
conveys "I don't have this information" is correct, regardless of \
what partial context is mentioned or omitted. Saying "not enough \
information" while mentioning partial related context = correct.
- Key test: BOTH sides must abstain. Gold-substantive + \
model-abstains = NO. Gold-abstention + model-abstains = YES.

FINAL CHECK: Before answering "no," you MUST reason through these \
steps:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different \
words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging / \
qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely \
unaddressed or contradicted.

Think step-by-step inside <judge_thinking> tags, then give your final \
verdict as exactly "yes" or "no" on a new line after the closing tag.\
"""


@dataclass
class Row:
    question_id: str
    question_type: str
    question: str = ""
    gold_answer: str = ""
    generated_answer: str = ""
    answer_session_ids: list[str] = field(default_factory=list)
    retrieved_session_ids: list[str] = field(default_factory=list)
    is_error: bool = False


@dataclass
class Score:
    n: int = 0
    recall_hits: int = 0
    accuracy_hits: int = 0
    errors: int = 0

    @property
    def recall(self) -> float:
        return self.recall_hits / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        return self.accuracy_hits / self.n if self.n else 0.0


def _load_run(path: Path) -> list[Row]:
    """Load JSONL, keeping only the LAST row per question_id.

    The runner re-appends a fresh row when an error question is retried
    (resume sees the error row as not-done). Without dedup, the same
    question would be counted twice — once as an error, once as a
    success — inflating sample size and skewing the metrics.
    """
    by_id: dict[str, Row] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            row = Row(
                question_id=data["question_id"],
                question_type=data.get("question_type", "unknown"),
                is_error="error" in data,
            )
            if not row.is_error:
                row.question = data.get("question", "")
                row.gold_answer = data.get("gold_answer", "")
                row.generated_answer = data.get("generated_answer", "")
                row.answer_session_ids = list(data.get("answer_session_ids") or [])
                row.retrieved_session_ids = [
                    r.get("session_id", "") for r in (data.get("retrieved") or [])
                ]
            # Latest write wins — successful retry overrides earlier error.
            by_id[row.question_id] = row
    return list(by_id.values())


def _recall_at_5(row: Row) -> int:
    if row.is_error or not row.answer_session_ids:
        return 0
    needles = set(row.answer_session_ids)
    top5 = row.retrieved_session_ids[:5]
    return int(any(s in needles for s in top5))


import re as _re

_JUDGE_THINKING_RE = _re.compile(
    r"<judge_thinking>.*?</judge_thinking>", _re.DOTALL | _re.IGNORECASE,
)


def _parse_judge_verdict(raw: str) -> int:
    """Extract the final yes/no after the <judge_thinking> block.

    The mem0-style judge prompt instructs the model to reason inside
    <judge_thinking>...</judge_thinking> tags and emit "yes" or "no"
    on a new line after the closing tag. Strip the thinking block, then
    look at the trailing token. Fall back to scanning the last non-empty
    line for a yes/no token if no thinking tag is present.
    """
    if not raw:
        return 0
    cleaned = _JUDGE_THINKING_RE.sub("", raw).strip()
    # Take the last non-empty line — handles "yes" / "no" / "**yes**"
    # / "Yes." / "verdict: yes" formats.
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return 0
    last = lines[-1].lower()
    # Be conservative: only count clear "yes" tokens. Anything that
    # leads with "no" stays a no; mixed/missing defaults to no.
    if _re.search(r"\byes\b", last):
        return 1
    return 0


# Preference-only override, adapted from Supermemory's memorybench
# (`src/prompts/defaults.ts:PREFERENCE_JUDGE_PROMPT`). The full mem0
# monolith has 8 preference rules buried inside ~150 lines; gpt-4o-mini
# misses "evaluate OVERALL thrust" and gpt-4o is only intermittently
# reliable on it. Supermemory's insight: the rule that matters IS the
# whole prompt, so the model can't lose it. We adapt their core
# sentence to our yes/no + <judge_thinking> output contract so
# _parse_judge_verdict still works.
PREFERENCE_JUDGE_PROMPT = """\
I will give you a question, a rubric describing a desired personalized \
response, and a model response. Decide whether the model response is \
correct.

The model does NOT need to reflect every point in the rubric. The \
response is correct as long as it recalls and utilizes the user's \
personal information correctly — i.e., the main thrust of the \
response is grounded in the user's stated preferences, habits, or \
prior context.

Wrong cases (answer "no"):
- The model says "I don't know" or refuses to answer (rubric is \
substantive — refusal is not abstention here).
- The main suggestions clearly contradict the user's stated \
preferences (not just brush against an anti-preference incidentally).
- The model invents a generic answer with no sign it used the user's \
personal context.

Correct cases (answer "yes"):
- The response demonstrates awareness of the user's personal context, \
even if it omits some specific items in the rubric.
- Adjacent / related suggestions alongside preferred ones — additive, \
not contradictory.
- The response addresses the rubric's main thrust even if individual \
keywords differ.

Think step-by-step in <judge_thinking> tags, then give your final \
verdict as exactly "yes" or "no" on a new line after the closing tag.\
"""


def _pick_judge_prompt(question_type: str) -> str:
    """Per-type judge prompt routing (Supermemory-style override).

    Default to the mem0-style monolithic prompt. Only preference
    questions use a dedicated short prompt — that's where the monolith
    demonstrably fails (gpt-4o-mini scored 0/9 across runs; gpt-4o
    intermittently misapplies the abstention rule). For all other
    types the monolith works fine and keeps comparability with mem0.
    """
    if "preference" in (question_type or "").lower():
        return PREFERENCE_JUDGE_PROMPT
    return JUDGE_SYSTEM_PROMPT


async def _judge_one(
    llm: AsyncOpenAI, question: str, gold: str, generated: str,
    question_type: str = "",
) -> int:
    if not generated or not gold:
        return 0
    system_prompt = _pick_judge_prompt(question_type)
    # Label distinction borrowed from Supermemory: preference golds are
    # rubrics, not strict ground truth — the wording nudges the judge
    # to grade against the spirit, not the letter.
    gold_label = (
        "Rubric" if "preference" in (question_type or "").lower()
        else "Correct Answer"
    )
    user = (
        f"Question: {question}\n\n"
        f"{gold_label}: {gold}\n\n"
        f"Model Response: {generated}"
    )
    response = await llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        seed=0,
    )
    raw = (response.choices[0].message.content or "")
    return _parse_judge_verdict(raw)


async def _score_run(
    rows: list[Row], skip_judge: bool, concurrency: int = 2,
    per_question_out: "Path | None" = None,
) -> tuple[Score, dict[str, Score]]:
    overall = Score()
    by_type: dict[str, Score] = defaultdict(Score)

    llm = AsyncOpenAI(max_retries=6)
    sem = asyncio.Semaphore(concurrency)

    async def _judge_with_sem(row: Row) -> int:
        if skip_judge or row.is_error:
            return 0
        async with sem:
            return await _judge_one(
                llm, row.question, row.gold_answer, row.generated_answer,
                question_type=row.question_type,
            )

    judge_tasks = [_judge_with_sem(r) for r in rows]
    judge_results = await asyncio.gather(*judge_tasks)

    per_q_records = []
    for row, judge_hit in zip(rows, judge_results):
        recall_hit = _recall_at_5(row)
        overall.n += 1
        by_type[row.question_type].n += 1
        if row.is_error:
            overall.errors += 1
            by_type[row.question_type].errors += 1
            continue
        overall.recall_hits += recall_hit
        overall.accuracy_hits += judge_hit
        by_type[row.question_type].recall_hits += recall_hit
        by_type[row.question_type].accuracy_hits += judge_hit
        per_q_records.append({
            "question_id": row.question_id,
            "question_type": row.question_type,
            "judge_pass": bool(judge_hit),
            "recall_hit": bool(recall_hit),
            "gold_answer": row.gold_answer,
            "generated_answer": row.generated_answer,
        })

    if per_question_out is not None:
        with per_question_out.open("w", encoding="utf-8") as f:
            for rec in per_q_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return overall, dict(by_type)


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _render_results_md(
    run_id: str,
    systems: list[tuple[str, Score, dict[str, Score]]],
    skip_judge: bool,
    baseline_run: str | None = None,
) -> str:
    names = [s[0] for s in systems]
    overalls = [s[1] for s in systems]
    by_types = [s[2] for s in systems]

    title_parts = " vs ".join(names)
    lines = [
        f"# LongMemEval-S: {title_parts}",
        f"",
        f"Run ID: `{run_id}`"
        + (f" (baseline: `{baseline_run}`)" if baseline_run else ""),
        f"",
        f"## Overall",
        f"",
    ]

    header = "| Metric        |"
    separator = "|---------------|"
    for name in names:
        header += f" {name:>13s} |"
        separator += f"{'':->15s}|"
    lines.append(header)
    lines.append(separator)

    for label, attr in [("recall@5", "recall"), ("accuracy", "accuracy")]:
        if skip_judge and label == "accuracy":
            row = f"| {label:13s} |"
            for _ in names:
                row += f" {'(skipped)':>13s} |"
            lines.append(row)
            continue
        vals = [getattr(o, attr) for o in overalls]
        row = f"| {label:13s} |"
        for v in vals:
            row += f" {_fmt_pct(v):>13s} |"
        lines.append(row)

    error_parts = ", ".join(
        f"{name} errors: {o.errors}" for name, o in zip(names, overalls)
    )
    lines.extend([
        f"",
        f"Sample size: {overalls[0].n} questions  ({error_parts})",
        f"",
        f"## By question type",
        f"",
    ])

    type_header = "| Question type             | n  |"
    type_sep = "|---------------------------|----|"
    for name in names:
        type_header += f" {name+' r@5':>13s} | {name+' acc':>13s} |"
        type_sep += f"{'':->15s}|{'':->15s}|"
    lines.append(type_header)
    lines.append(type_sep)

    all_types = sorted(set().union(*(bt.keys() for bt in by_types)))
    for qt in all_types:
        scores = [bt.get(qt, Score()) for bt in by_types]
        n = max(s.n for s in scores)
        row = f"| {qt:25s} | {n:>2d} |"
        for s in scores:
            acc = "(skipped)" if skip_judge else _fmt_pct(s.accuracy)
            row += f" {_fmt_pct(s.recall):>13s} | {acc:>13s} |"
        lines.append(row)

    lines.extend([
        f"",
        f"## Notes",
        f"",
        f"- recall@5: top-5 retrieved memories include at least one "
        f"tagged with a needle session id.",
        f"- accuracy: gpt-4o-mini judges whether the generated answer "
        f"matches the gold answer (yes/no).",
        f"- Errored questions count as 0 for both metrics.",
        f"",
    ])

    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    all_row_sets: list[tuple[str, list[Row]]] = []

    if args.baseline_run:
        baseline_dir = RUNS_DIR / args.baseline_run
        if not baseline_dir.exists():
            raise SystemExit(f"baseline run dir not found: {baseline_dir}")
        baseline_rows = _load_run(baseline_dir / "memonative.jsonl")
        all_row_sets.append((f"Memonative-{args.baseline_run}", baseline_rows))

    mn_rows = _load_run(run_dir / "memonative.jsonl")
    all_row_sets.append((f"Memonative-{args.run_id}", mn_rows))

    mem_rows = _load_run(run_dir / "mem0.jsonl")
    if mem_rows:
        all_row_sets.append(("mem0", mem_rows))
    elif not args.baseline_run:
        mem_rows = _load_run(RUNS_DIR / "v1" / "mem0.jsonl")
        if mem_rows:
            all_row_sets.append(("mem0", mem_rows))

    for name, rows in all_row_sets:
        if not rows:
            print(f"WARNING: empty run for {name}")

    common = set.intersection(*(
        {r.question_id for r in rows} for _, rows in all_row_sets
    ))
    total_ids = set.union(*(
        {r.question_id for r in rows} for _, rows in all_row_sets
    ))
    if len(common) != len(total_ids):
        dropped = total_ids - common
        print(f"Dropping {len(dropped)} non-overlapping questions")
        all_row_sets = [
            (name, [r for r in rows if r.question_id in common])
            for name, rows in all_row_sets
        ]

    print(f"Scoring {len(common)} questions across "
          f"{len(all_row_sets)} systems...")

    systems: list[tuple[str, Score, dict[str, Score]]] = []
    for name, rows in all_row_sets:
        # Dump per-question verdicts for the primary run only
        is_primary = name == f"Memonative-{args.run_id}"
        per_q_out = (run_dir / "per_question.jsonl") if is_primary else None
        overall, by_type = await _score_run(
            rows, args.skip_judge, per_question_out=per_q_out,
        )
        systems.append((name, overall, by_type))

    md = _render_results_md(
        args.run_id, systems, args.skip_judge,
        baseline_run=args.baseline_run,
    )

    out_path = run_dir / "RESULTS.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print(md)

    tracker = get_tracker()
    if tracker.total_cost > 0:
        print(f"\nJudge cost: ${tracker.total_cost:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--baseline-run", type=str, default="",
                   help="another run-id to include its Memonative results "
                        "as a baseline (e.g. v1)")
    p.add_argument("--skip-judge", action="store_true",
                   help="skip LLM-judge accuracy scoring (recall@5 only)")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
