"""Engine exceptions.

The engine used to raise `fastapi.HTTPException` directly from places like
the LLM client, which made a web framework a hard dependency of code that
does no HTTP. These replace it.

Each error carries the `status_code` and `detail` its `HTTPException`
carried, so the API layer re-raises mechanically instead of maintaining a
translation table that drifts from the raise sites. An in-process caller can
ignore `status_code` entirely and match on the type.

Stdlib only. Anything importable from here must stay importable with no
third-party packages installed.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base for every error the engine raises on purpose."""

    status_code = 500

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class NotFoundError(EngineError):
    """A requested memory, slot or revision chain does not exist."""

    status_code = 404


class ConfigurationError(EngineError):
    """The engine is missing configuration it needs to serve the request.

    503 rather than 500: the deployment is incomplete, not broken, and the
    same request will succeed once the operator fixes it.
    """

    status_code = 503


class EngineLLMError(EngineError):
    """An LLM or embedding call failed.

    `status_code` varies by cause (429 for quota, 502 for upstream, 500 for
    anything unrecognised), so it is per-instance rather than per-class.
    """

    def __init__(self, detail: str, *, status_code: int = 502):
        self.status_code = status_code
        super().__init__(detail)
