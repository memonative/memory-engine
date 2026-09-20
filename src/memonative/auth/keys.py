"""API key generation + hashing.

Plaintext keys are returned to the caller exactly once at creation and
never persisted. Only their SHA-256 hex digest lives in the database.

SHA-256 (not bcrypt/argon2) is correct here: API keys are high-entropy
random strings, not user-chosen passwords. The threat model is "leak of
the api_keys table must not yield usable bearers", which a fast hash
satisfies. Slow hashes only matter when input entropy is low.

Format:
  mn_<43 urlsafe chars>           plaintext bearer (46 chars total)
  <64 hex chars>                  stored digest

The `mn_` prefix is a recognizability marker — it shows up in logs,
secret-scanners, and grep, making leaked keys easier to detect.

Changing this prefix does not invalidate keys already issued: `hash_key`
digests the whole plaintext and nothing validates the prefix, so existing
bearers keep working and only newly minted ones carry the new marker.
"""

from __future__ import annotations

import hashlib
import secrets

KEY_PREFIX = "mn_"
_TOKEN_BYTES = 32  # → 43 urlsafe-base64 chars after prefix


def generate_key() -> str:
    """Mint a fresh plaintext bearer token. Show this to the user once."""
    return KEY_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)


def hash_key(plaintext: str) -> str:
    """SHA-256 hex digest of a plaintext bearer. Matches the api_keys.key_hash column."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
