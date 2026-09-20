"""Agent SDK tool manifests.

Exposes Memonative's agent-native endpoints as ready-to-register tool
schemas for Anthropic Claude tool use and OpenAI function calling.

Design choices:

- user_id is intentionally OMITTED from tool input schemas. The host
  application must bind the current session's user_id server-side
  before forwarding the call to /v1/memory/*. If user_id were a tool
  parameter, an agent could read another user's memory by guessing a
  UUID — a memory layer must never let that happen.

- Tools are deliberately small and orthogonal:
    search_memory  → read-only relevance search with token budget
    recall_fact    → fetch the current value (and history) of one slot
    save_memory    → write a structured memory directly (no extraction)
    list_history   → revision chain for one slot

  The fat /v1/process endpoint is NOT exposed as a tool — it's the
  "easy mode" chat-app entry point, not an agent primitive.
"""

from typing import Any


# ---------------------------------------------------------------
# Tool definitions. JSON-schema fragments we'll wrap per provider.
# ---------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_memory",
        "description": (
            "Search the user's long-term memory for items relevant to a "
            "natural-language query. Returns scored matches with full "
            "provenance (trust_score, strength, source, age) so you can "
            "decide what to trust. Read-only — does not write or "
            "reinforce. Use this before answering questions that depend "
            "on prior conversation or personal context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what you're looking for.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 8,
                    "description": "Maximum number of memories to return.",
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural"],
                    },
                    "description": (
                        "Restrict to specific types: 'semantic' for "
                        "stable facts, 'episodic' for events and "
                        "experiences, 'procedural' for interaction "
                        "preferences. Omit to search all types."
                    ),
                },
                "token_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200000,
                    "description": (
                        "Cap the total estimated tokens returned. "
                        "Memories are returned in score order until the "
                        "budget is exhausted; truncated=true if more "
                        "matches existed."
                    ),
                },
                "include_associations": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Also walk graph edges to surface associated "
                        "memories that didn't score on their own."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_fact",
        "description": (
            "Fetch the current value of one specific attribute slot "
            "(e.g. 'location', 'employer', 'name'). Use this when you "
            "need an exact known fact, not similarity search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attribute_slot": {
                    "type": "string",
                    "maxLength": 50,
                    "description": (
                        "The slot name. Common slots: location, name, "
                        "profession, employer, native_language, "
                        "relationship_status. Custom slots are "
                        "lowercase_underscore."
                    ),
                },
            },
            "required": ["attribute_slot"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Write a memory directly. Use this when YOU (the agent) "
            "have learned something new about the user that should "
            "persist — e.g., after a successful tool call confirms a "
            "fact, or when the user makes a clear declarative "
            "statement. Slot-based semantic memories with the same "
            "attribute_slot replace prior values automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "What to remember. Concise, present-tense, "
                        "atomic (one fact per call)."
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "default": "episodic",
                    "description": (
                        "semantic = stable fact (requires "
                        "attribute_slot); episodic = event or "
                        "experience; procedural = interaction "
                        "preference or workflow."
                    ),
                },
                "attribute_slot": {
                    "type": "string",
                    "maxLength": 50,
                    "description": (
                        "REQUIRED when type='semantic'. The slot name "
                        "the new value occupies; an existing value in "
                        "the same slot will be moved to history."
                    ),
                },
                "salience": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": (
                        "0.2 mundane, 0.5 useful, 0.7 important, "
                        "0.9 critical. Higher salience decays slower."
                    ),
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 lowercase single-word topic tags.",
                },
                "correction_type": {
                    "type": "string",
                    "enum": ["retraction", "temporal_update"],
                    "description": (
                        "When updating an existing slot value, classify "
                        "the change. 'retraction' = the old value was "
                        "wrong; 'temporal_update' = the old value was "
                        "true but the situation changed. Omit for new "
                        "facts."
                    ),
                },
                "trigger_message": {
                    "type": "string",
                    "description": (
                        "The user message (or your reasoning) that "
                        "prompted this write. Stored in the audit trail."
                    ),
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "list_history",
        "description": (
            "List the full revision history of one attribute slot — "
            "every value it has held, why it changed (retraction vs "
            "temporal_update), and when. Use this when you need to "
            "explain or reason about how a fact evolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attribute_slot": {
                    "type": "string",
                    "maxLength": 50,
                    "description": "The slot to inspect.",
                },
            },
            "required": ["attribute_slot"],
        },
    },
]


# ---------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------

def as_anthropic_tools() -> list[dict[str, Any]]:
    """Anthropic Messages API tool format.

    https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    """
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in _TOOLS
    ]


def as_openai_tools() -> list[dict[str, Any]]:
    """OpenAI Chat Completions / Responses API function-calling format.

    https://platform.openai.com/docs/guides/function-calling
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in _TOOLS
    ]


def as_deepseek_tools() -> list[dict[str, Any]]:
    """DeepSeek Chat Completions tool-calling format.

    DeepSeek's API is OpenAI-compatible — same `{"type": "function",
    "function": {...}}` schema consumed by the same SDK with a
    `base_url` override. We expose it as its own provider name purely
    for discoverability, so callers asking for `/v1/tools/deepseek`
    get a valid manifest instead of a 400.

    https://api-docs.deepseek.com/guides/function_calling
    """
    return as_openai_tools()


def build_tool_manifest(provider: str) -> dict[str, Any]:
    provider_norm = provider.strip().lower()
    if provider_norm == "anthropic":
        return {"provider": "anthropic", "tools": as_anthropic_tools()}
    if provider_norm == "openai":
        return {"provider": "openai", "tools": as_openai_tools()}
    if provider_norm == "deepseek":
        return {"provider": "deepseek", "tools": as_deepseek_tools()}
    raise ValueError(
        f"unknown provider {provider!r}. "
        f"supported: 'anthropic', 'openai', 'deepseek'"
    )


# ---------------------------------------------------------------
# Tool name → /v1/memory/* endpoint mapping.
# Host apps use this to dispatch a tool call to the right HTTP path.
# ---------------------------------------------------------------

TOOL_ROUTES: dict[str, dict[str, str]] = {
    "search_memory": {"method": "POST", "path": "/v1/memory/search"},
    "recall_fact": {"method": "GET", "path": "/v1/memory/facts/{user_id}/{attribute_slot}"},
    "save_memory": {"method": "POST", "path": "/v1/memory/write"},
    "list_history": {"method": "GET", "path": "/v1/memory/facts/{user_id}/{attribute_slot}"},
}
