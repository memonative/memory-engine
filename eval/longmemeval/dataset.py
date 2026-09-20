"""LongMemEval-S loader.

The dataset is hosted on HuggingFace (xiaowu0162/longmemeval). We grab
the raw `longmemeval_s.json` once and cache it under
`eval/longmemeval/data/`. ~50MB, never re-downloaded.

Format (per example):
    {
      "question_id": "...",
      "question_type": "single-session-user" | "multi-session" | ...,
      "question": "...",
      "answer": "...",
      "question_date": "2023-05-01 (Mon)",
      "haystack_session_ids": ["sess_1", "sess_2", ...],
      "haystack_dates":       ["2023-04-30", ...],
      "haystack_sessions":    [
          [{"role": "user"/"assistant", "content": "...",
            "has_answer": true?}, ...],
          ...
      ],
      "answer_session_ids":   ["sess_42"]   # the needle(s)
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).parent / "data"
# HF mirror stores the file extension-less; we cache locally with .json
# so editors and `cat` don't have to guess at the format.
LME_S_REMOTE = "longmemeval_s"
LME_S_FILENAME = "longmemeval_s.json"
HF_REPO = "xiaowu0162/longmemeval"


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[dict]


@dataclass
class Example:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[Session]
    answer_session_ids: set[str]

    @property
    def haystack_size(self) -> int:
        return len(self.sessions)


def _local_path() -> Path:
    return DATA_DIR / LME_S_FILENAME


def download_longmemeval_s(force: bool = False) -> Path:
    """Download LongMemEval-S to the local cache. Idempotent.

    Uses huggingface_hub if available; falls back to a direct
    `https://huggingface.co/.../resolve/main/...` URL via urllib so
    callers don't strictly need the hf client installed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = _local_path()
    if target.exists() and not force:
        return target

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=LME_S_REMOTE,
            local_dir=str(DATA_DIR),
        )
        # Remote file is extension-less; copy to the .json cache path so
        # subsequent loads find it and editors can recognise the format.
        src = Path(path)
        if src != target:
            target.write_bytes(src.read_bytes())
    except ImportError:
        import urllib.request
        url = (
            f"https://huggingface.co/datasets/{HF_REPO}/"
            f"resolve/main/{LME_S_REMOTE}"
        )
        with urllib.request.urlopen(url) as resp:
            target.write_bytes(resp.read())

    return target


def load_longmemeval_s(
    limit: int | None = None,
    question_types: set[str] | None = None,
) -> list[Example]:
    """Return parsed examples. Downloads on first call if missing.

    `limit` caps the number of examples (handy for smoke tests).
    `question_types` filters to a subset of LongMemEval's 5 categories.
    """
    path = download_longmemeval_s()
    raw = json.loads(path.read_text(encoding="utf-8"))

    out: list[Example] = []
    for row in raw:
        if question_types and row["question_type"] not in question_types:
            continue

        sessions = [
            Session(session_id=sid, date=date, turns=turns)
            for sid, date, turns in zip(
                row["haystack_session_ids"],
                row["haystack_dates"],
                row["haystack_sessions"],
            )
        ]
        out.append(Example(
            question_id=row["question_id"],
            question_type=row["question_type"],
            question=row["question"],
            answer=row["answer"],
            question_date=row.get("question_date", ""),
            sessions=sessions,
            answer_session_ids=set(row["answer_session_ids"]),
        ))

        if limit and len(out) >= limit:
            break

    return out


def question_type_counts(examples: Iterable[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.question_type] = counts.get(ex.question_type, 0) + 1
    return counts


if __name__ == "__main__":
    # Smoke test: download (if needed), parse, print summary.
    print(f"Downloading/locating dataset...")
    path = download_longmemeval_s()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  cached at: {path}  ({size_mb:.1f} MB)")

    examples = load_longmemeval_s()
    print(f"\nLoaded {len(examples)} examples")
    print(f"\nQuestion type breakdown:")
    for qtype, n in sorted(question_type_counts(examples).items()):
        print(f"  {qtype:32s} {n:4d}")

    print(f"\nFirst example:")
    ex = examples[0]
    print(f"  id:        {ex.question_id}")
    print(f"  type:      {ex.question_type}")
    print(f"  question:  {ex.question[:100]}...")
    print(f"  answer:    {ex.answer[:100]}...")
    print(f"  haystack:  {ex.haystack_size} sessions")
    print(f"  needle(s): {ex.answer_session_ids}")
