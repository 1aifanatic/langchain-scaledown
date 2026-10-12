# LangChain Middleware by ScaleDown

`langchain-scaledown` adds [ScaleDown](https://scaledown.ai) small language models (SLMs)
to any LangChain agent as middleware. Add one line to `middleware=[...]` and your
agent summarizes long chats, compresses retrieved documents, or turns conversations into
structured records with purpose-built SLMs, instead of spending frontier-model tokens.

**Privacy:** this package runs inside your application. It has no telemetry and sends
nothing anywhere except ScaleDown's API endpoints, and only when your agent runs, using
the `SCALEDOWN_API_KEY` you provide. Each middleware sends only the text it needs (see
[What gets sent](#what-gets-sent)).

## What it does

- **Summarizes**: `ScaledownSummarizationMiddleware` replaces older chat history with a
  ScaleDown summary once a conversation gets long. It's a drop-in for LangChain's built-in
  `SummarizationMiddleware`, with no second LLM call to pay for.
  - `sd_summarize`: replaces frontier LLM summarization calls (~90% cheaper)
- **Compresses**: `ScaledownCompressionMiddleware` shrinks large retrieved context before
  each model call, relative to the user's question. LangChain has no built-in equivalent.
  - `sd_compress`: reduces context tokens 50–70% before your LLM call (for RAG, long docs)
- **Compresses tool results on request**: `ScaledownToolCompressionMiddleware` gives
  your tools an optional `compression_query` parameter. When the model says what it's looking for,
  the tool's output is compressed down to the relevant parts before the model sees it.
- **Extracts**: `ScaledownExtractionMiddleware` turns a conversation into a structured
  record as it happens, such as a support ticket with the order number, error message,
  sentiment and issue type.
  - `sd_extract`: replaces frontier LLM entity extraction calls (~95% cheaper)
  - `sd_classify`: used automatically for label fields like sentiment and issue type

All four **fail open**. If ScaleDown is unreachable, out of credits, or rejects a
request, the agent keeps running as if the middleware weren't there, and a warning is
logged.

## Installation

```bash
pip install -U langchain-scaledown
```

Until the first PyPI release, install from GitHub:

```bash
pip install "git+https://github.com/scaledown-team/langchain-integration#subdirectory=libs/scaledown"
```

Set your API key:

```bash
export SCALEDOWN_API_KEY="your-api-key"
```

Requires Python 3.10+ and `langchain` 1.3+.

## Quickstart

```python
from langchain.agents import create_agent
from langchain_scaledown import (
    ScaledownExtractionMiddleware,
    ScaledownSummarizationMiddleware,
)

agent = create_agent(
    "openai:gpt-6-sol",
    tools=[...],
    middleware=[
        ScaledownSummarizationMiddleware(
            "openai:gpt-6-sol", trigger=("tokens", 4000), keep=("messages", 20)
        ),
        ScaledownExtractionMiddleware({
            "order_id": "Order number the customer mentions",
            "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
        }),
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Order A-1042 failed again!"}]})
result["scaledown_extraction"]["fields"]
# {"order_id": "A-1042", "sentiment": "frustrated"}
```

## Works with

| Product | How |
|---|---|
| LangChain agents | `create_agent(..., middleware=[...])` |
| Deep Agents | `create_deep_agent(..., middleware=[...])` |
| Sync and async | `agent.invoke()` and `agent.ainvoke()` |
| Checkpointers | With a checkpointer, summaries persist and extraction records fill in turn by turn |

## Middleware

| Middleware | ScaleDown SLM | Runs | What it changes |
|---|---|---|---|
| `ScaledownSummarizationMiddleware` | `sd_summarize` | Before a model call, once the conversation passes `trigger` | Replaces older messages in agent state with one summary message |
| `ScaledownCompressionMiddleware` | `sd_compress` | Before each model call, when context is over `min_context_chars` | Replaces the system message sent to the model with a compressed version |
| `ScaledownToolCompressionMiddleware` | `sd_compress` | After a tool call where the model set `compression_query`, when the output is over `min_result_chars` | Adds `compression_query` to tool schemas; replaces the tool result with a compressed version |
| `ScaledownExtractionMiddleware` | `sd_extract` + `sd_classify` | Once per agent run, before the first model call | Adds a `scaledown_extraction` record to agent state; optionally shows it to the model |

### Summarization (`sd_summarize`)

A drop-in alternative to LangChain's built-in
[`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization),
with the same `trigger` / `keep` options. When the conversation crosses `trigger`,
every message except the most recent ones selected by `keep` is replaced in state by a
single ScaleDown summary. Each stretch of history is summarized once. Tool results are
never separated from the call that requested them.

```python
ScaledownSummarizationMiddleware(
    "openai:gpt-6-sol",         # your agent's model: used for its context window and token counting
    trigger=("tokens", 4000),   # or ("messages", 50), or a list: summarize when any is hit
    keep=("messages", 20),      # or ("tokens", 2000)
    instructions="Keep order numbers and dates.",  # optional, passed to ScaleDown
)
```

> **Note:** `sd_summarize` is in private preview. Contact ScaleDown to enable it on your key.

### Compression (`sd_compress`)

Before each model call, the system message is compressed relative to the user's latest
question, and the model sees the compressed version.

```python
agent = create_agent(
    "openai:gpt-4.1",
    system_prompt=f"Answer using these documents:\n\n{retrieved_docs}",
    middleware=[ScaledownCompressionMiddleware(min_context_chars=2000)],
)
```

> **Use it for needle-in-a-haystack, RAG-style workloads**, where a big block of
> retrieved documents holds a few relevant facts. **Don't use it for short prompts or
> creative generation**, where compression can drop instructions, tone, or detail.
>
> **With Deep Agents**, the system message also holds the harness's own instructions,
> and compression would condense those too.

Pass `context_extractor=` to choose what gets compressed. It's a function
`(request) -> (context, prompt)` that returns `None` to skip. The compressed context
always replaces the system message.

### Tool result compression (`sd_compress`)

Adds an optional `compression_query` parameter to your tools' schemas. The model fills it in when it
only needs part of a tool's output, for example `fetch_url(url=..., compression_query="refund
policy for EU orders")`. The middleware removes `compression_query` before the tool runs, so your tools
don't change. It then compresses the output relative to `compression_query`. This works for any
tool, including MCP and third-party tools.

```python
from langchain_scaledown import ScaledownToolCompressionMiddleware

agent = create_agent(
    "openai:gpt-6-luna",
    tools=[search_docs, fetch_url, *mcp_tools],
    middleware=[
        ScaledownToolCompressionMiddleware(
            tools=["fetch_url"],     # default: every tool
            min_result_chars=4000,   # leave shorter outputs alone
        )
    ],
)
```

The compressed result starts with a note like
`[Compressed by ScaleDown for "refund policy for EU orders": 18,402 -> 3,110 chars. ...]`,
so the model can call the tool again without `compression_query` if it needs the full output.
Error results, `Command` results, and results with non-text content (such as images)
are never compressed.

> **Watch for name clashes.** Search tools often already have a `query` parameter, which
> is why the added one is called `compression_query`. Rename it with `param_name=`. The middleware
> raises `ValueError` if a targeted tool already has a parameter with that name.

### Extraction (`sd_extract` + `sd_classify`)

Turns a conversation into a structured record. One schema holds two kinds of fields,
and ScaleDown handles both, extraction through the OpenAI-compatible API and classification through the Decisions API:

| Field type | Schema | Value | Good for |
|---|---|---|---|
| Extractive | A description string | Copied verbatim from what the user said | Order numbers, error messages, emails, names |
| Classification | An object with `labels` | One of your labels, via `sd_classify` | Sentiment, issue type, priority |

Only the **user's** messages are extracted from, never the agent's own replies, so every
extractive value is something the customer actually said.

```python
from langchain.tools import ToolRuntime, tool

ticket = ScaledownExtractionMiddleware(
    entities={
        "order_id": "Order number the customer mentions",
        "error_message": "Exact error message or code the customer saw",
        "steps_tried": "What the customer says they already tried",
        "customer_email": "Customer's email address",
        "issue_type": {
            "labels": [
                {"name": "billing", "rubric": "Is this about a charge, refund, or payment?"},
                {"name": "technical", "rubric": "Is this about a bug or something not working?"},
                {"name": "account", "rubric": "Is this about login, password, or account settings?"},
            ]
        },
        "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
    },
    context_chars=0,  # skip evidence snippets to keep state small
)


@tool
def file_ticket(runtime: ToolRuntime) -> str:
    """File a support ticket with the details gathered so far."""
    fields = runtime.state.get("scaledown_extraction", {}).get("fields", {})
    return crm.create_ticket(**fields)  # your ticketing system


agent = create_agent("openai:gpt-4.1", tools=[file_ticket], middleware=[ticket])
```

With a checkpointer, extraction re-runs every turn over the whole conversation, so the
record fills in as the customer adds detail.

> **`inject_into_prompt=True`** (off by default) also shows the fields to the model, for
> example so it can escalate a frustrated customer. Extractive values are customer text,
> so this puts customer text into the system prompt. Have the agent branch on
> classification fields, whose values can only be one of your labels.

## Example output

After `agent.invoke(...)`, the extraction record is in the result:

```python
result["scaledown_extraction"]
```

```json
{
  "fields": {
    "order_id": "A-1042",
    "error_message": "ERR_PAYMENT_DECLINED",
    "steps_tried": null,
    "customer_email": "jane@example.com",
    "issue_type": "billing",
    "sentiment": "frustrated"
  },
  "entities": [
    {"text": "A-1042", "type": "order_id", "confidence": 0.97, "start": 6, "end": 12},
    {"text": "billing", "type": "issue_type", "confidence": 0.93, "start": 0, "end": 0}
  ]
}
```

`fields` has one entry per schema key: the highest-confidence value, or `null` if nothing
was found. `entities` holds ScaleDown's raw matches with confidence scores.

## What gets sent

| Middleware | Sent to ScaleDown | Endpoint |
|---|---|---|
| Summarization | The older messages being summarized | `/v1/chat/completions` (`model: summarize`) |
| Compression | The system message and the latest user message | `/compress/raw/` |
| Tool result compression | The tool's output and the model's `compression_query`, only when the model sets it | `/compress/raw/` |
| Extraction | The user's messages (not the agent's replies) and your schema | `/v1/chat/completions` (`model: extract`) for extractive fields, `/v1/scaledown` (Decisions API) for classification keys |

## Configuration

| Setting | Where | Default |
|---|---|---|
| API key | `SCALEDOWN_API_KEY` env var, or `api_key=` | Required |
| API base URL | `SCALEDOWN_BASE_URL` env var, or `base_url=` | `https://api.scaledown.xyz` |
| Shared client | `client=ScaledownClient(...)` on any middleware | One client per middleware |

## Development

```bash
git clone https://github.com/scaledown-team/langchain-integration
cd langchain-integration/libs/scaledown
uv sync --all-groups
make test                                        # unit tests (no network)
make lint                                        # ruff + mypy
SCALEDOWN_API_KEY=... make integration_tests     # tests against the live ScaleDown API
uv run python scripts/smoke_test.py              # every middleware in a real agent (mocked API)
```

## Get a ScaleDown API key

Sign up at [scaledown.ai/dashboard](https://scaledown.ai/dashboard) — 50 million free tokens included.
