"""Shared HTTP helpers for Client and AsyncClient.

Two jobs:

  build_headers(...)      — bearer injection, content-type, optional request id
  raise_for_status(resp)  — map HTTP status families onto the SDK error types

Having these in one place keeps sync and async clients symmetric, and
makes it easy to add a new behaviour (e.g. retry) in a single spot.
"""

from __future__ import annotations

from typing import Any

import httpx

from memonative_client.errors import (
    AuthError,
    NotFoundError,
    ServerError,
    MemonativeError,
    ValidationError,
)


def build_headers(api_key: str | None, request_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-ID"] = request_id
    return headers


def raise_for_status(resp: httpx.Response) -> None:
    """Translate a non-2xx HTTP response into a typed SDK exception."""
    if resp.is_success:
        return

    # Try to parse a JSON body once; fall back to text.
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text

    message = _extract_message(body) or f"HTTP {resp.status_code}"
    status = resp.status_code

    if status in (401, 403):
        raise AuthError(message, status_code=status, body=body)
    if status == 404:
        raise NotFoundError(message, status_code=status, body=body)
    if status in (400, 422):
        raise ValidationError(message, status_code=status, body=body)
    if 500 <= status < 600:
        raise ServerError(message, status_code=status, body=body)
    # Any other 4xx falls through as a generic MemonativeError rather
    # than being silently treated as success.
    raise MemonativeError(message, status_code=status, body=body)


def _extract_message(body: Any) -> str | None:
    # FastAPI: {"detail": "..."} or {"detail": [{"msg": "..."}, ...]}
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and "msg" in first:
                return str(first["msg"])
    if isinstance(body, str) and body:
        return body[:300]
    return None
