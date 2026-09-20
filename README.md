<p align="center">
  <h1 align="center">Memonative</h1>
  <p align="center">
    <strong>The memory engine for AI agents</strong>
  </p>
  <p align="center">
    A neuroscience-inspired memory layer that decays, consolidates, and tracks revision history — not another vector store with summarisation on top.
  </p>
  <p align="center">
    Built by <a href="https://github.com/Amaymani"><strong>@Amaymani</strong></a> and <a href="https://github.com/Freak3123"><strong>@Freak3123</strong></a>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &middot;
    <a href="#api-reference">API Reference</a> &middot;
    <a href="#python-sdk">SDK</a> &middot;
    <a href="#auth-model">Auth</a> &middot;
    <a href="https://memonative.com">Website</a>
  </p>
  <p align="center">
    <a href="https://github.com/memonative/memory-engine/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/memonative/memory-engine/actions/workflows/ci.yml/badge.svg"></a>
    <img alt="License" src="https://img.shields.io/badge/license-BSL_1.1-blue">
    <img alt="Python" src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white">
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
    <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL_16-pgvector-4169E1?logo=postgresql&logoColor=white">
  </p>
</p>

---

## Why Memonative?

Most memory layers for AI agents are thin wrappers around a vector database: embed, store, retrieve, summarise. That works for search — it doesn't work for *memory*.

Human memory doesn't just store and retrieve. It **decays** without reinforcement. It **consolidates** repeated experiences into stable knowledge. It **reconsolidates** on recall — subtly updating itself every time it's accessed. And it maintains a **revision history**: you don't just know where you live, you know where you *used to* live.

Memonative brings all of this to AI agents through small, JSON-only HTTP primitives that work with any framework — Anthropic tool use, OpenAI function calling, LangGraph, CrewAI, or your own.

### Key capabilities

- **Three memory types** — semantic (stable facts), episodic (experiences), procedural (preferences) — each with its own decay profile
- **Attribute slots with revision chains** — facts compete on identity (`location`, `employer`, etc.) with full temporal history. Distinguish a retraction ("that was wrong") from a temporal update ("that changed")
- **Ebbinghaus decay + reinforcement** — strength decays with configurable half-life; retrieval reinforces; reinforcement count extends half-life
- **Consensus engine** — repeated low-confidence episodics are promoted to high-confidence semantic facts
- **Reconsolidation on recall** — retrieved memories can be enriched or reinforced as a side effect, not as an explicit edit
- **Contradiction detection** — temporal and contradiction edges between conflicting facts, detected asynchronously to keep the hot path fast
- **Full audit trail** — every fact change is recorded with revision type, reason, trigger message, and temporal bounds. Enterprise-ready compliance out of the box
- **Goal tracking** — track user goals with horizon levels (immediate / short-term / long-term)
- **Background lifecycle** — Celery workers run Ebbinghaus decay hourly and episodic consolidation every 4 hours
- **Multi-tenant isolation** — PostgreSQL Row-Level Security enforces tenant boundaries at the database level, so one deployment can back many applications
- **Per-tenant LLM keys** — optional; each tenant can override the shared engine key with its own, encrypted at rest with Fernet

---

## Benchmark

**81.0% accuracy on [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval)** (100 stratified questions), with zero reader-prompt overfitting. Every point of lift is architectural.

| Question type | n | Accuracy | vs baseline |
|---|---|---|---|
| knowledge-update | 15 | **93.3%** | +40.0 |
| temporal-reasoning | 27 | **88.9%** | +29.6 |
| single-session-user | 14 | **85.7%** | +7.1 |
| multi-session | 27 | **74.1%** | +7.4 |
| single-session-assistant | 11 | **72.7%** | +45.4 |
| single-session-preference | 6 | **50.0%** | +33.3 |

> Published 90%+ numbers on this benchmark typically come from heavily eval-overfit reader prompts. Memonative's reader prompt is unchanged between the baseline (57%) and this run (81%) — all improvement is architectural.

<details>
<summary>What changed architecturally</summary>

- **Atomic extraction** — each list item, table cell, and duration becomes its own memory with a resolved `event_date`, plus truncated-JSON repair for long sessions
- **Two-zone slot canonicalizer** — cosine >= 0.90 auto-merges, 0.75-0.90 falls through to an LLM tiebreaker. Equivalent slot names converge without false merges
- **Superseded-revision suppression** — stale slot values stop being echoed as new memories
- **Per-row event semantics** — `event_type` (add/update/delete) + resolved `event_date` let the reader reason about *when* something happened
- **Timeline retrieval + LLM dedup rerank** — composable retrieval modules alongside vector + FTS fusion
- **DeepSeek engine LLM** — `temperature=0` deterministic extraction with 16K token output window

</details>

<details>
<summary>Reproducing</summary>

```bash
PYTHONPATH=src python -m eval.longmemeval.runner \
  --n 100 --seed 42 --run-id v3-extraction-fix-100 --systems memonative

PYTHONPATH=src python -m eval.longmemeval.scorer \
  --run-id v3-extraction-fix-100 --baseline-run v3
```

See `eval/longmemeval/LAUNCH.md` for parallel execution and resumability.

</details>

---

## Quickstart

### Prerequisites

- Docker Desktop
- An [OpenAI API key](https://platform.openai.com/api-keys) (embeddings only)
- A [DeepSeek API key](https://platform.deepseek.com/) (engine LLM)

### 1. Clone and configure

```bash
git clone https://github.com/memonative/memory-engine.git
cd memory-engine
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

### 2. Start the stack

```bash
docker-compose up -d --build
```

This brings up PostgreSQL (pgvector), Redis, the FastAPI API server, a Celery worker, and the Celery beat scheduler.

### 3. Run migrations

```bash
docker-compose exec api alembic upgrade head
```

### 4. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Interactive API docs are at `http://localhost:8000/docs`.

### 5. Send your first message

```bash
curl -sX POST http://localhost:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "message": "I just moved to Berlin from London. Started a new job at Stripe."
  }' | python -m json.tool
```

The engine extracts structured facts (location=Berlin, employer=Stripe), writes them with attribute slots, creates revision entries for the temporal update, and returns an `injection_text` ready for your system prompt.

---

## Architecture

```
                        ┌─→  DeepSeek  (engine LLM: extraction, consensus,
                        │              edges, rerank, timeline, consolidation,
                        │              reconsolidation, slot canonicalization)
PostgreSQL 16 + pgvector ←┤
          ↑             FastAPI
       Celery worker     │
          ↑              └─→  OpenAI   (text-embedding-3-small, 1536 dims)
        Redis
```

### Request flow (`POST /v1/process`)

```
message  →  extraction  →  write + embed (parallel)  →  retraction
         →  retrieval   →  consensus + reconsolidation (parallel)
         →  apply corrections  →  goals  →  format injection text
```

Each stage is a composable module in `engine/`. Contradiction detection is deferred to Celery to keep the hot path fast.

### LLM split

| Role | Provider | Model | Purpose |
|------|----------|-------|---------|
| Engine | DeepSeek | deepseek-v4-flash | All structured-JSON tasks: extraction, consensus, edges, consolidation, reconsolidation, timeline classification, slot canonicalization, dedup rerank |
| Embedding | OpenAI | text-embedding-3-small | 1536-dim vectors, dimension-locked to pgvector HNSW index |

Both keys come from the environment by default. A tenant can override the engine key with its own via the admin API — see [Auth model](#auth-model).

---

## API Reference

### Easy mode — one endpoint

`POST /v1/process` runs the full pipeline and returns structured data plus a pre-formatted `injection_text` for your system prompt. Best for chatbots that want memory without complexity.

### Agent mode — composable primitives

For tool-calling agents. JSON-only. Read endpoints have no side effects.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/v1/process` | Full pipeline: extract, write, retrieve, consensus, reconsolidate, format |
| `POST` | `/v1/memory/search` | Read-only relevance search with token budget and provenance |
| `GET` | `/v1/memory/{memory_id}` | Single memory with full provenance metadata |
| `POST` | `/v1/memory/write` | Direct structured write (bypasses extraction LLM) |
| `GET` | `/v1/memory/slots/{user_id}` | List all known attribute slots with current values |
| `GET` | `/v1/memory/audit/{user_id}` | Full audit trail across all slots (paginated) |
| `GET` | `/v1/memory/facts/{user_id}/{slot}` | Current value + revision chain for one slot |
| `GET` | `/v1/tools/{provider}` | Ready-to-register tool manifest (`anthropic`, `openai`, `deepseek`) |
| `POST` | `/v1/consolidate/{user_id}` | Manually trigger episodic consolidation |

### Admin API (master-key gated)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/admin/tenants` | Create tenant + initial API key |
| `GET` | `/admin/tenants` | List all tenants |
| `POST` | `/admin/tenants/{id}/keys` | Mint additional API key |
| `DELETE` | `/admin/keys/{id}` | Revoke a key |
| `PUT` | `/admin/tenants/{id}/credentials` | Set per-tenant LLM credentials (encrypted at rest) |
| `GET` | `/admin/tenants/{id}/credentials` | View credential status (keys are never exposed) |
| `DELETE` | `/admin/tenants/{id}/credentials` | Remove per-tenant credentials |

### Memory provenance

Every memory returned by agent endpoints carries full provenance:

```
id, content, type, attribute_slot, score, strength, trust_score,
salience, source, decay_state, reinforcement_count, context_tags,
created_at, last_accessed_at, estimated_tokens
```

The agent uses this metadata to decide what to trust, how much to inject, and when to question stale information.

---

## Using with AI frameworks

### Anthropic Claude

```python
import anthropic, requests

manifest = requests.get("http://localhost:8000/v1/tools/anthropic").json()

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=manifest["tools"],
    messages=[{"role": "user", "content": "Where do I live again?"}],
)

# Dispatch tool_use blocks to /v1/memory/* endpoints.
# Bind user_id server-side — it is intentionally NOT a tool parameter.
```

### OpenAI

```python
import openai, requests

manifest = requests.get("http://localhost:8000/v1/tools/openai").json()

client = openai.OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o",
    tools=manifest["tools"],
    messages=[{"role": "user", "content": "Where do I live again?"}],
)
```

### DeepSeek

```python
import openai, requests

manifest = requests.get("http://localhost:8000/v1/tools/deepseek").json()

client = openai.OpenAI(
    api_key="sk-your-deepseek-key",
    base_url="https://api.deepseek.com/v1",
)
resp = client.chat.completions.create(
    model="deepseek-chat",
    tools=manifest["tools"],
    messages=[{"role": "user", "content": "Where do I live again?"}],
)
```

> **Why `user_id` is not a tool parameter:** If `user_id` were in the tool schema, an agent could (or could be prompt-injected to) read another user's memory. The host application must bind the session's `user_id` server-side before forwarding calls.

---

## Python SDK

A separate package from the engine, so an app that only *calls* Memonative
doesn't install Postgres drivers and a task queue to do it:

```bash
pip install memonative-client
```

Source at [`client/`](client/). MIT-licensed, unlike the engine — see
[Licence](#licence). Sync and async clients:

```python
from memonative_client import Client

with Client("http://localhost:8000", api_key="mn_...") as mn:
    # Full pipeline
    mn.process(user_id=uid, message="I moved to Berlin")

    # Read operations
    facts = mn.recall(uid, attribute_slot="location")
    history = mn.history(uid, attribute_slot="location")
    results = mn.search(uid, query="where do I live?", top_k=5)

    # Direct write (bypasses extraction LLM)
    mn.write(uid, memories=[{
        "content": "user prefers Python",
        "type": "procedural",
        "salience": 0.6,
    }])
```

### Async

```python
from memonative_client import AsyncClient

async with AsyncClient("http://localhost:8000", api_key="mn_...") as mn:
    resp = await mn.search(uid, "where do I live?")
```

### Agent tool dispatch

`dispatch_tool_call` handles the `user_id`-binding and endpoint routing in one line:

```python
from memonative_client import Client, dispatch_tool_call

mn = Client(base_url, api_key=key)
manifest = mn.tool_manifest("anthropic")

# In your agent loop:
for block in response.content:
    if block.type == "tool_use":
        result = dispatch_tool_call(
            mn,
            user_id=session_user_id,
            name=block.name,
            tool_input=block.input,
        )
```

### Tenant provisioning

```python
with Client(base_url, api_key=MASTER_KEY) as mn:
    bundle = mn.admin.create_tenant(name="acme")
    print(bundle.api_key)  # plaintext, shown once — capture immediately
```

---

## Deployment

### Local / single application

```bash
cp .env.example .env
# Set OPENAI_API_KEY and DEEPSEEK_API_KEY
docker-compose up -d --build
```

Leaving `API_KEY` empty puts the engine in dev mode: no auth, and every request lands in the implicit `default` tenant. Fine on a laptop, never on a reachable host.

### Production

`docker-compose.prod.yml` runs the same images against managed Postgres and Redis, with a one-shot `migrate` service, two API replicas, two workers, and one beat. It reads `.env.prod` — set `API_KEY` there to turn auth on.

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Many applications on one deployment

Row-Level Security isolates tenants at the database level, so one engine can back several applications without their memories ever meeting. Mint a tenant and a key per application via the admin API, and optionally give each its own LLM key.

Per-tenant keys need `MASTER_ENCRYPTION_KEY` set — without it there is nothing to decrypt stored credentials with, and every tenant runs on the shared `DEEPSEEK_API_KEY`.

---

## Auth model

Two trust levels, by design.

### `/v1/*` — tenant-scoped

Every request resolves to exactly one tenant via the bearer token. All reads and writes are filtered by `tenant_id`. Cross-tenant lookups return 404 — never revealing existence.

### `/admin/*` — master-key only

Tenant-scoped keys are rejected even if valid. Used for tenant provisioning, key management, and per-tenant credential setup.

| `API_KEY` env var | `/v1/*` routes | `/admin/*` routes |
|---|---|---|
| Empty | Unauthenticated, all requests use `default` tenant | Disabled (503) |
| Set | Bearer required: tenant key (DB lookup) or master key | Master key required |

Bearer tokens are SHA-256 hashed at rest. Lost keys cannot be recovered — revoke and mint a new one.

### Provisioning a tenant

```bash
# Create tenant + initial key (plaintext shown ONCE)
curl -sX POST http://localhost:8000/admin/tenants \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"acme"}'

# Give the tenant its own engine LLM key (needs MASTER_ENCRYPTION_KEY)
curl -sX PUT http://localhost:8000/admin/tenants/$TENANT_ID/credentials \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"engine_api_key":"sk-tenant-deepseek-key"}'
```

---

## Audit trail

Every change to a slotted fact is recorded as a revision entry with:

- **Revision type** — `initial`, `temporal_update`, `retraction`, `enrichment`
- **Temporal bounds** — `valid_from` / `valid_until` for bitemporal queries
- **Reason** — why the change happened
- **Trigger message** — the user message that caused the change

Query via the API:

```bash
# List all known facts for a user
curl http://localhost:8000/v1/memory/slots/$USER_ID \
  -H "Authorization: Bearer $TOKEN"

# Full audit trail across all slots
curl http://localhost:8000/v1/memory/audit/$USER_ID \
  -H "Authorization: Bearer $TOKEN"

# Revision history for a specific fact
curl http://localhost:8000/v1/memory/facts/$USER_ID/location \
  -H "Authorization: Bearer $TOKEN"
```

---

## Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Secret scanning on commit (gitleaks)
pip install pre-commit && pre-commit install

# Run tests (DB-less by default, DB tests auto-skip without Postgres)
PYTHONPATH=src pytest

# Run a specific test
PYTHONPATH=src pytest tests/test_specific.py -k "test_name"

# DB integration tests (requires live Postgres)
PYTHONPATH=src pytest -m dbintegration

# Run API outside Docker (against dockerised db + redis)
PYTHONPATH=src uvicorn memonative.main:app --host 0.0.0.0 --port 8000 --reload

# Schema migrations
PYTHONPATH=src alembic revision --autogenerate -m "describe change"
PYTHONPATH=src alembic upgrade head
```

---

## Project structure

```
src/memonative/
  main.py              — FastAPI entrypoint
  config.py            — settings, scoring weights, decay profiles
  worker.py            — Celery app + beat schedule
  api/
    routes.py          — /v1 endpoints
    admin.py           — /admin tenant + key management
    schemas.py         — Pydantic request/response models
    llm.py             — LLM client wrapper (engine + embedding)
    usage.py           — usage accounting seam (no-op; hosted builds override)
  engine/
    extraction.py      — message → structured memories + intent
    write_path.py      — slot-aware writes, revision chains, canonicalization
    retrieval.py       — vector + FTS fusion, scoring, association walk
    decay.py           — Ebbinghaus decay + reinforcement
    consolidation.py   — episodic clustering → semantic knowledge
    consensus.py       — multi-evidence voting
    reconsolidation.py — enrich/reinforce on recall
    edges.py           — temporal + contradiction edges
    goals.py           — goal CRUD
    timeline.py        — temporal-intent classifier
    rerank.py          — LLM dedup of near-duplicate candidates
  db/
    models.py          — SQLAlchemy 2.0 models
    enums.py           — domain enums
    database.py        — async session factory
  auth/
    deps.py            — FastAPI auth dependencies
    keys.py            — token generation + hashing
    bootstrap.py       — startup tenant/key provisioning
    credentials.py     — per-tenant credential resolution
    encryption.py      — Fernet envelope encryption
  agent_tools.py       — tool manifests for Anthropic/OpenAI/DeepSeek
client/                — `memonative-client` SDK, published separately (MIT)
  src/memonative_client/
alembic/               — database migrations
eval/                  — LongMemEval benchmark harness
tests/                 — pytest suite
```

---

## Authors

Memonative is built and maintained by:

- **[@Amaymani](https://github.com/Amaymani)**
- **[@Freak3123](https://github.com/Freak3123)**

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[security policy](SECURITY.md).

---

## Licence

[Business Source License 1.1](LICENSE)

Licensor: Memonative. You may use Memonative for any purpose, including production self-hosted deployments, except offering it as a commercial managed memory service. On **2029-06-07**, the licence converts to **Apache 2.0**.

The SDK in [`client/`](client/) is **[MIT](client/LICENSE)**, not BSL. It gets vendored into other people's applications, where a source-available licence would be a reason to reach for something else — and it carries none of the engine, only request building and response parsing.
