"""Backwards-compatible re-export of `memonative.llm`.

The client moved out of the API layer so importing it no longer requires
FastAPI. `routes.py`, `worker.py` and `scripts/record_golden.py` still import
`LLMClient` from here.
"""

from memonative.llm import LLMClient, LLMProtocol

__all__ = ["LLMClient", "LLMProtocol"]
