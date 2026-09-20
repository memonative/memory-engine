"""SDK exception hierarchy.

Callers should catch MemonativeError for anything this SDK raises, or
one of the specific subclasses when they want to branch.

The hierarchy mirrors the HTTP status families, not the full grid —
we're not exhaustively shadowing every 4xx. 422 is treated as
ValidationError (client sent bad data); 5xx is ServerError.
"""

from __future__ import annotations


class MemonativeError(Exception):
    """Base for every error the SDK raises."""

    def __init__(self, message: str, *, status_code: int | None = None, body: object | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthError(MemonativeError):
    """401 or 403 — bearer token missing, invalid, or lacks permission."""


class NotFoundError(MemonativeError):
    """404 — resource doesn't exist (or is masked by tenant scoping)."""


class ValidationError(MemonativeError):
    """400 or 422 — request payload rejected by the server."""


class ServerError(MemonativeError):
    """5xx — server-side failure. Retrying may help; tuning is the
    caller's responsibility (SDK has no built-in retry)."""
