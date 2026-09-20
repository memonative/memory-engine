"""Session-wide test defaults.

`.env` holds real cloud credentials so the app can run locally against the
deployed stack. That is the wrong target for a test suite: the write path
fires `check_contradictions_task.delay(...)` for every memory it writes, so
running the golden corpus with the deployed broker configured would push
twenty runs' worth of tasks onto the production queue.

Pinned here rather than per-test because the settings object is built on
first import of `memonative.config`, so whichever test module imported it
first would otherwise decide for the whole session. Set `REDIS_URL_TEST` to
point somewhere other than the docker-compose Redis.
"""

from __future__ import annotations

import os

os.environ["REDIS_URL"] = os.environ.get("REDIS_URL_TEST", "redis://localhost:6379/0")

from memonative.config import settings  # noqa: E402

# Hosts that are somebody's real data. The suite writes and deletes memories
# (`purge_user` truncates four tables for the golden user), so pointing it at
# one of these is destructive, not merely untidy. Forgetting to pass
# DATABASE_URL is all it takes: settings then falls back to `.env`, which is
# configured for the deployed stack.
_MANAGED_DB_HOSTS = ("neon.tech", "upstash.io", "rds.amazonaws.com")

if os.environ.get("MEMONATIVE_ALLOW_MANAGED_DB") != "1":
    for _host in _MANAGED_DB_HOSTS:
        if _host in settings.DATABASE_URL:
            raise RuntimeError(
                f"DATABASE_URL points at a managed host ({_host}). The test "
                f"suite writes and deletes memories. Start the local stack "
                f"(`docker compose up -d db redis`) and pass "
                f"DATABASE_URL=postgresql+asyncpg://memonative:memonative@"
                f"localhost:5432/memonative, or set "
                f"MEMONATIVE_ALLOW_MANAGED_DB=1 if you really mean it."
            )

from tests.golden.runner import install_uuid_defaults  # noqa: E402

# Must happen before the first INSERT compiles anywhere in the session — see
# the function's docstring. Harmless for tests that don't care: ids stay
# random until `deterministic_uuids` seeds the generator.
install_uuid_defaults()
