# Security policy

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private reporting instead:

[**Report a vulnerability**](https://github.com/memonative/memory-engine/security/advisories/new)

That opens a private advisory visible only to you and the maintainers. You will
get an acknowledgement within 3 working days and an assessment within 10.

Useful things to include, roughly in order of how much they help:

- The version or commit you tested, and whether you ran it self-hosted or from PyPI
- What an attacker gains — reading another tenant's memories is a different
  severity from crashing a worker
- A request sequence that reproduces it, or a failing test
- Whether authentication was required, and at which level (tenant key or master key)

Report it even if you are not sure it is exploitable. A wrong report costs us
ten minutes; an unreported tenant-isolation bug costs somebody their data.

## Disclosure

We will agree a disclosure date with you, defaulting to 90 days from the report
or the day a fix ships, whichever comes first. Fixes are released as a patch
version and described in a published advisory. If you want credit, say so and
name how you would like to be credited; if you would rather stay anonymous, that
is fine too. We will not involve lawyers over good-faith research.

## What is in scope

Anything in this repository: the engine, the `memonative-client` SDK, the
migrations, and the Docker setup. In particular we want to hear about

- **Cross-tenant data access** — the strongest claim this project makes is that
  one deployment can back many applications without them seeing each other.
  Anything that breaks that is the highest severity we issue.
- **Authentication bypass** on `/v1/*` or, worse, `/admin/*`
- **Key material leaking** — API keys, per-tenant LLM keys, or the Fernet master
  key appearing in logs, error responses, or API output
- **Injection** reaching the database or the LLM prompt in a way that crosses a
  tenant boundary

The hosted service at memonative.com is a separate codebase. Report issues with
it the same way and we will route them.

## What is not a vulnerability

- **Running with an empty `API_KEY`.** That is documented dev mode and disables
  authentication deliberately. Deploying it that way is a misconfiguration.
- **Prompt injection changing what gets remembered.** The engine stores what the
  conversation says. A user who lies to it will have their lie remembered. That
  is the design, and it stops at their own tenant.
- **LLM spend from a caller you authorised.** Rate limiting and quota are the
  deployer's job.
- Missing hardening headers, or scanner output with no demonstrated impact.

## Notes for people deploying this

Three properties do real work here, and all three can be switched off by accident:

- Tenant isolation is Postgres Row-Level Security keyed on
  `app.current_tenant_id`. It is transaction-scoped, so it has to be re-set after
  every commit. A code path that opens a session without setting it is a bug
  worth reporting.
- API keys are stored only as SHA-256 digests. The plaintext is shown once at
  creation and cannot be recovered — which is the point.
- Per-tenant LLM keys are Fernet-encrypted with `MASTER_ENCRYPTION_KEY`. Lose
  that key and the credentials are unrecoverable; leak it and they are all
  readable. Keep it out of the repo and rotate it separately from the database.

Every commit here is scanned by gitleaks in CI, over full history rather than
just the diff, because a secret that was committed and later deleted is still
published.
