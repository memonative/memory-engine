from typing import Literal
from uuid import UUID

from pydantic import SecretStr
from pydantic_settings import BaseSettings


# Deterministic id for the implicit default tenant. Mirrors the constant
# used in the tenancy migration so dev/staging/prod align. Lives here
# rather than in `auth/deps.py` so an in-process caller can reach it
# without importing FastAPI; `auth.deps` re-exports it.
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-00000000eeee")


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://memonative:memonative@localhost:5432/memonative"
    OPENAI_API_KEY: SecretStr = SecretStr("")
    DEEPSEEK_API_KEY: SecretStr = SecretStr("")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    REDIS_URL: str = "redis://localhost:6379/0"
    API_KEY: SecretStr = SecretStr("")
    MASTER_ENCRYPTION_KEY: SecretStr = SecretStr("")

    # Cross-encoder reranking
    RERANK_ENABLED: bool = True
    RERANK_BACKEND: Literal["local", "api"] = "local"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_API_KEY: SecretStr = SecretStr("")
    RERANK_API_PROVIDER: str = ""
    RERANK_API_MODEL: str = ""
    RERANK_BLEND_WEIGHT: float = 0.7
    RERANK_MAX_CANDIDATES: int = 30

    # Timeline retrieval on hot path (gated on intent, no extra LLM call)
    TIMELINE_RETRIEVAL_ENABLED: bool = True

    class Config:
        env_file = ".env"


settings = Settings()

INTENT_TYPE_WEIGHTS = {
    "learn": {"episodic": 0.3, "semantic": 0.8, "procedural": 0.6},
    "recall": {"episodic": 0.9, "semantic": 0.5, "procedural": 0.2},
    "decide": {"episodic": 0.5, "semantic": 0.8, "procedural": 0.3},
    "plan": {"episodic": 0.4, "semantic": 0.7, "procedural": 0.5},
    "execute": {"episodic": 0.2, "semantic": 0.6, "procedural": 0.9},
    "debug": {"episodic": 0.6, "semantic": 0.7, "procedural": 0.8},
    "create": {"episodic": 0.3, "semantic": 0.5, "procedural": 0.7},
    "reflect": {"episodic": 0.8, "semantic": 0.6, "procedural": 0.4},
    "explore": {"episodic": 0.4, "semantic": 0.6, "procedural": 0.3},
    "social": {"episodic": 0.7, "semantic": 0.4, "procedural": 0.5},
}

SCORING_WEIGHTS = {
    "vector":     0.50,
    "strength":   0.20,
    "trust":      0.15,
    "salience":   0.10,
    "type_match": 0.05,
}

def get_decay_profile(memory_type: str, source: str):
    from dataclasses import dataclass
    
    @dataclass
    class DecayProfile:
        half_life_hours: float
        default_trust: float
        default_salience: float
        
    if memory_type == "episodic":
        if source == "user_stated":
            return DecayProfile(72.0, 0.7, 0.5)
        else:
            return DecayProfile(48.0, 0.4, 0.3)
    elif memory_type == "semantic":
        if source == "user_stated":
            return DecayProfile(720.0, 0.8, 0.6)
        else:
            return DecayProfile(720.0, 0.8, 0.5) # Consolidated trust inherited dynamically
    elif memory_type == "procedural":
        if source == "user_stated":
            return DecayProfile(2160.0, 0.8, 0.5)
        else:
            return DecayProfile(1440.0, 0.5, 0.4)
    return DecayProfile(72.0, 0.5, 0.5)
