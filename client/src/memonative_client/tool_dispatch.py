"""Tool-call dispatcher for Anthropic / OpenAI agent loops.

The problem this solves:

  The tool manifest at /v1/tools/{provider} deliberately omits
  user_id from every tool's input schema — if an agent could pass a
  user_id, it could be prompt-injected into reading another user's
  memory. So the host app has to bind the current session's user_id
  server-side before forwarding the call.

  That "forward a tool_use block to the right /v1/memory/* endpoint,
  inject user_id, return a JSON-serialisable tool_result" dance is
  boilerplate in every agent loop. This module does it in one call.

Usage (sync):

    from memonative_client import Client
    from memonative_client.tool_dispatch import dispatch_tool_call

    client = Client(base_url, api_key=key)

    # In your Anthropic tool-use loop, for each tool_use block:
    result = dispatch_tool_call(
        client,
        user_id=session_user_id,
        name=block.name,
        tool_input=block.input,
    )
    # Feed `result` back as the tool_result content.

Async variant is `async_dispatch_tool_call`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from memonative_client.async_client import AsyncClient
from memonative_client.client import Client


# Names the server's tool manifest exposes. Kept in sync manually with
# src/memonative/agent_tools.py — the server is the source of truth.
_KNOWN_TOOLS = {"search_memory", "recall_fact", "save_memory", "list_history"}


class UnknownToolError(ValueError):
    """Raised when a tool name doesn't match any Memonative primitive.

    Distinct from SDK errors (network / HTTP) so callers can tell
    "the agent asked for something we don't expose" apart from
    "the server rejected a request".
    """


def _build_save_memory_payload(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Translate a save_memory tool schema into a MemoryWriteItem body.

    The tool schema exposes a *single* memory write at a time (cleaner
    for agents), while /v1/memory/write accepts a list. We wrap the
    agent's single item into a list of length 1.
    """
    item: dict[str, Any] = {
        "content": tool_input["content"],
        "type": tool_input.get("type", "episodic"),
        "salience": tool_input.get("salience", 0.5),
    }
    if "attribute_slot" in tool_input and tool_input["attribute_slot"]:
        item["attribute_slot"] = tool_input["attribute_slot"]
    if "context_tags" in tool_input and tool_input["context_tags"]:
        item["context_tags"] = tool_input["context_tags"]
    if "correction_type" in tool_input and tool_input["correction_type"]:
        item["correction_type"] = tool_input["correction_type"]
    return item


def dispatch_tool_call(
    client: Client,
    *,
    user_id: UUID | str,
    name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Route one agent tool call to the matching Memonative primitive.

    Returns a JSON-serialisable dict — drop it straight into a
    tool_result content block. Raises UnknownToolError when the tool
    name isn't one Memonative provides; SDK errors (auth, not-found,
    validation, server) propagate unchanged so the caller can decide
    whether to retry, surface to the user, or fall through.
    """
    if name == "search_memory":
        resp = client.search(
            user_id,
            query=tool_input["query"],
            top_k=tool_input.get("top_k", 8),
            memory_types=tool_input.get("memory_types"),
            token_budget=tool_input.get("token_budget"),
            include_associations=tool_input.get("include_associations", True),
        )
        return resp.model_dump(mode="json")

    if name == "recall_fact":
        resp = client.recall(user_id, tool_input["attribute_slot"])
        return resp.model_dump(mode="json")

    if name == "save_memory":
        item = _build_save_memory_payload(tool_input)
        resp = client.write(
            user_id,
            memories=[item],
            trigger_message=tool_input.get("trigger_message"),
        )
        return resp.model_dump(mode="json")

    if name == "list_history":
        resp = client.history(user_id, tool_input["attribute_slot"])
        return resp.model_dump(mode="json")

    raise UnknownToolError(
        f"{name!r} is not a Memonative tool. Known: {sorted(_KNOWN_TOOLS)}"
    )


async def async_dispatch_tool_call(
    client: AsyncClient,
    *,
    user_id: UUID | str,
    name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Async twin of `dispatch_tool_call`. Same contract."""
    if name == "search_memory":
        resp = await client.search(
            user_id,
            query=tool_input["query"],
            top_k=tool_input.get("top_k", 8),
            memory_types=tool_input.get("memory_types"),
            token_budget=tool_input.get("token_budget"),
            include_associations=tool_input.get("include_associations", True),
        )
        return resp.model_dump(mode="json")

    if name == "recall_fact":
        resp = await client.recall(user_id, tool_input["attribute_slot"])
        return resp.model_dump(mode="json")

    if name == "save_memory":
        item = _build_save_memory_payload(tool_input)
        resp = await client.write(
            user_id,
            memories=[item],
            trigger_message=tool_input.get("trigger_message"),
        )
        return resp.model_dump(mode="json")

    if name == "list_history":
        resp = await client.history(user_id, tool_input["attribute_slot"])
        return resp.model_dump(mode="json")

    raise UnknownToolError(
        f"{name!r} is not a Memonative tool. Known: {sorted(_KNOWN_TOOLS)}"
    )
