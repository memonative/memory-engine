# Launching the v1 run

This is a long-running benchmark (~12 hrs wall-clock with parallel
terminals, ~21 hrs sequential). Launch in YOUR terminal, not Claude's,
so the processes survive even if the Claude session ends.

## Fastest: two parallel terminals (~12 hrs)

Open two terminal windows in this repo. Run these side by side:

**Terminal A — Memonative**
```bash
PYTHONPATH=src venv/Scripts/python.exe -m eval.longmemeval.runner \
    --n 100 --run-id v1 --systems memonative --cost-kill 15
```

**Terminal B — mem0**
```bash
PYTHONPATH=src venv/Scripts/python.exe -m eval.longmemeval.runner \
    --n 100 --run-id v1 --systems mem0 --cost-kill 15
```

They share the same `--run-id v1` and `--seed 42`, so both adapters
attack the exact same 100 questions. They write to different JSONL
files (`runs/v1/memonative.jsonl` and `runs/v1/mem0.jsonl`) so there's
no contention. Postgres and Qdrant are independent.

## Slower: one terminal, sequential (~21 hrs)

```bash
PYTHONPATH=src venv/Scripts/python.exe -m eval.longmemeval.runner \
    --n 100 --run-id v1 --cost-kill 15
```

## Resume if anything dies

Same command — runner skips question_ids already in the JSONL. Crashes,
ctrl-Cs, machine sleeps, network blips: all recoverable by re-running.

Each question's first step is `reset_user`, so even if a question's
state was partially written before a crash, the rerun starts from a
clean haystack for that user.

## When both are done

```bash
PYTHONPATH=src venv/Scripts/python.exe -m eval.longmemeval.scorer --run-id v1
```

Outputs `eval/longmemeval/data/runs/v1/RESULTS.md` with the head-to-head
table.

## Watching progress

Both runners print every 5 questions. To watch live:
```bash
tail -f eval/longmemeval/data/runs/v1/memonative.jsonl | wc -l
tail -f eval/longmemeval/data/runs/v1/mem0.jsonl | wc -l
```

Per-question avg from the smoke test:
- Memonative: ~5 min, $0.03/question
- mem0: ~7 min, $0.05/question

100 questions × both systems ≈ $8 total OpenAI spend.
