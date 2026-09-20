# Contributing

Thanks for looking. This is a working engine rather than a demo, so the bar is
"would we want to debug this at 3am", not "does it pass".

## What this project is trying to be

Memonative is a general memory layer meant to sit behind many different
applications. That shapes what gets merged:

- **Changes must not be tuned to one application or one benchmark.** The
  LongMemEval numbers in the README are a measurement, not a target. A change
  that lifts the score by special-casing that dataset's phrasing makes the engine
  worse at its actual job.
- **Retrieval and scoring changes need evidence.** "This felt better" is not
  evidence. Say what you ran it against and what moved.
- **Multi-tenancy is a core feature, not deployment plumbing.** Anything that
  touches a tenant-scoped table has to go through the RLS path.

If you are unsure whether something fits, open an issue before writing it. A
paragraph is cheaper than a rejected pull request.

## Getting set up

```bash
pip install -r requirements-dev.txt
pre-commit install          # gitleaks, runs on every commit
```

The package lives in `src/`, so `PYTHONPATH=src` is required for every local
command — pytest, uvicorn and alembic all need it.

Postgres and Redis are easiest via Docker:

```bash
docker-compose up -d db redis
PYTHONPATH=src alembic upgrade head
```

## Tests

```bash
PYTHONPATH=src pytest                      # everything
PYTHONPATH=src pytest -m "not perf"        # what CI runs
PYTHONPATH=src pytest tests/test_decay.py -k reinforce
```

Two markers matter:

- `dbintegration` — needs a live Postgres with pgvector. These **skip silently**
  when the database is unreachable. A green run without a database is therefore
  not a passing run, which is why CI always starts one.
- `perf` — asserts on query latency. Excluded from CI because shared runners
  cannot hold timing steady, and a check that fails at random is a check people
  learn to ignore. Run it locally before touching retrieval.

New behaviour needs a test. Bug fixes need a test that fails before the fix.

## What CI checks

Three jobs, all of which must pass:

1. **test** — the suite against a real Postgres and Redis, on Python 3.11 and 3.13
2. **secrets** — gitleaks over the full commit history
3. **build** — both distributions build, pass `twine check`, and do not overlap

That third one exists because packaging breaks invisibly. A misplaced TOML table
once swallowed the entire dependency list here while all tests kept passing.

## Style

There is no formatter to appease. Match the surrounding code.

Comments are the one thing we are opinionated about: write them when the *why*
is not obvious — a constraint, an ordering requirement, a workaround for
something surprising. Do not write them to restate what the line does. Most of
the comments in this codebase exist because someone lost an afternoon to that
exact line.

## Pull requests

- One concern per PR. A refactor bundled with a fix is hard to review and harder
  to revert.
- Explain what you changed and why in the description. If it changes behaviour,
  say what a caller would notice.
- Say how you tested it, including what you could not test.

## Licensing of contributions

The engine is under [BSL 1.1](LICENSE) and the SDK in
[`client/`](client/LICENSE) is MIT. By opening a pull request you agree your
contribution is licensed under whichever of the two applies to the files you
touched.
