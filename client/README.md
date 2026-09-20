# memonative-client

Python client for the [Memonative](https://github.com/memonative/memory-engine) memory engine.

Talks to a Memonative server over HTTP. It does not contain the engine — point
it at a deployment you run yourself, or at a hosted one. The only thing that
changes between the two is `base_url`.

```bash
pip install memonative-client
```

Two dependencies: `httpx` and `pydantic`. If you want to *run* the engine
rather than call it, install `memonative` instead.

## Use

```python
from memonative_client import Client

with Client("http://localhost:8000", api_key="mn_...") as mn:
    # Full pipeline: extract, write, retrieve, reinforce.
    resp = mn.process(user_id=uid, message="I moved to Berlin")

    # Read-only.
    results = mn.search(uid, query="where do I live?", top_k=5)
    facts = mn.recall(uid, attribute_slot="location")

    # Write a known fact directly, skipping the extraction LLM.
    mn.write(uid, memories=[{
        "content": "user prefers Python",
        "type": "procedural",
        "salience": 0.6,
    }])
```

`AsyncClient` mirrors every method:

```python
from memonative_client import AsyncClient

async with AsyncClient("http://localhost:8000", api_key="mn_...") as mn:
    resp = await mn.search(uid, "where do I live?")
```

## Agent tool calls

The server publishes tool schemas for Anthropic, OpenAI and DeepSeek.
`dispatch_tool_call` routes a model's tool call to the right endpoint:

```python
from memonative_client import Client, dispatch_tool_call

mn = Client(base_url, api_key=key)
manifest = mn.tool_manifest("anthropic")

for block in response.content:
    if block.type == "tool_use":
        result = dispatch_tool_call(
            mn,
            user_id=session_user_id,   # bound here, never by the model
            name=block.name,
            tool_input=block.input,
        )
```

`user_id` is not part of any published tool schema, and `dispatch_tool_call`
takes it as a separate argument for that reason: a model that could name the
user whose memory it reads is one prompt injection away from reading someone
else's. Bind it from your own session.

## Errors

Every non-2xx response raises a subclass of `MemonativeError`:
`AuthError` (401/403), `NotFoundError` (404), `ValidationError` (422),
`ServerError` (5xx).

No retries, no caching, no background batching — one method call is one HTTP
request, so timeouts and backoff stay under your control.

## Licence

MIT. The engine itself is [BSL 1.1](../LICENSE); this client is deliberately
permissive so it can be vendored into closed-source applications.
