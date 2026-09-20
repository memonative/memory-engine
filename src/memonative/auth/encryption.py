"""Envelope encryption for per-tenant secrets (LLM keys).

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography library.
The master key lives in MASTER_ENCRYPTION_KEY (env / secrets manager).

Unlike bearer tokens (hashed, compare-only), LLM keys must be
decryptable because we call the upstream API with them.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from memonative.config import settings


def _get_multi_fernet() -> "MultiFernet":
    """Build a MultiFernet from comma-separated keys for rotation support.

    First key is used for encryption; all keys are tried for decryption.
    """
    from cryptography.fernet import MultiFernet

    raw = settings.MASTER_ENCRYPTION_KEY.get_secret_value()
    if not raw:
        raise RuntimeError(
            "MASTER_ENCRYPTION_KEY is not set. "
            "Per-tenant BYOK requires an encryption key. "
            "Generate one with: python -c "
            "\"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("MASTER_ENCRYPTION_KEY is empty after parsing.")
    return MultiFernet([Fernet(k.encode()) for k in keys])


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a plaintext secret. Returns a URL-safe base64 string."""
    if not plaintext:
        raise ValueError("Cannot encrypt an empty string.")
    return _get_multi_fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a ciphertext back to the original plaintext.

    Tries all keys in MASTER_ENCRYPTION_KEY (comma-separated) so that
    key rotation doesn't orphan existing ciphertext.
    """
    try:
        return _get_multi_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        raise ValueError(
            "Failed to decrypt tenant credential. "
            "None of the configured MASTER_ENCRYPTION_KEY values could decrypt this value."
        )
