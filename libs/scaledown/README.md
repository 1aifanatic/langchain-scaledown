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
- **Extracts**: `ScaledownExtractionMiddleware` turns a conversation into a structured
  record as it happens, such as a support ticket with the order number, error message,
  sentiment and issue type.
  - `sd_extract`: replaces frontier LLM entity extraction calls (~95% cheaper)
  - `sd_classify`: used automatically for label fields like sentiment and issue type

All three **fail open**. If ScaleDown is unreachable, out of credits, or rejects a
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
| `ScaledownExtractionMiddleware` | `sd_extract` + `sd_classify` | Once per agent run, before the first model call | Adds a `scaledown_extraction` record to agent state; optionally shows it to the model |

### Summarization (`sd_summarize`)

A drop-in for LangChain's built-in
[`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization),
with the same `trigger` / `keep` options. When the conversation crosses `trigger`,
every message except the most recent ones selected by `keep` is replaced in state by a
single ScaleDown summary. Each stretch of history is summarized once. Tool results are
never separated from the call that requested them.

Pass your agent's model. It's used for its context window (for `"fraction"` limits) and
for token counting, never to write summaries.

```python
ScaledownSummarizationMiddleware(
    "openai:gpt-6-sol",
    trigger=("fraction", 0.8),  # or ("tokens", 4000), ("messages", 50), or a list of these
    keep=("fraction", 0.1),     # or ("messages", 20), ("tokens", 2000)
    instructions="Keep order numbers and dates.",  # optional, passed to ScaleDown
)
```

By default ScaleDown is asked for the same sections as LangChain's summary prompt
(session intent, summary, artifacts, next steps), and all of the older history is sent
(`trim_tokens_to_summarize=None`).

#### With Deep Agents

Deep Agents has its own summarization middleware, which also saves the evicted history
to a file the agent can read back. Keep it, and have ScaleDown write its summaries by
passing a `ScaledownSummarizer`:

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_scaledown import ScaledownSummarizer

backend = StateBackend()
summarization = SummarizationMiddleware(
    model="openai:gpt-6-sol",
    backend=backend,
    trigger=("fraction", 0.85),
    keep=("fraction", 0.10),
    summarizer=ScaledownSummarizer(),
)
agent = create_deep_agent(
    model="openai:gpt-6-sol",
    backend=backend,
    middleware=[summarization],  # replaces the default summarization middleware
)
```

`ScaledownSummarizer` is a `Runnable` that takes a string or a list of messages and
returns the summary, so it works anywhere a `summarizer=` is accepted, or on its own
(`ScaledownSummarizer().invoke(text)`).

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

### Extraction (`sd_extract` + `sd_classify`)

Turns a conversation into a structured record. One schema holds two kinds of fields,
and ScaleDown handles both in a single call:

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

## Example: support triage

[`examples/support_triage.py`](examples/support_triage.py) runs a support agent end to
end. ScaleDown fills in each ticket from the customer's messages, and a `file_ticket`
tool routes it by issue type and escalates frustrated customers. The LLM never retypes
the order number or email.

```bash
pip install langchain-scaledown langchain-anthropic
export SCALEDOWN_API_KEY=...  ANTHROPIC_API_KEY=...
python examples/support_triage.py                         # or --model openai:gpt-4.1
```

It triages a batch of tickets into a table, then shows one conversation where the
record fills in turn by turn (`+` marks a newly extracted field). Output from a real run:

```text
#    order_id   error_message       issue_type  sentiment   team            priority
---  ---------  ------------------  ----------  ----------  --------------  --------
1    A-1042     -                   billing     neutral     Billing         normal
2    B-2207     ERR_SYNC_TIMEOUT    technical   frustrated  Tech Support    high
3    C-3310     INVALID_2FA_CODE    account     frustrated  Accounts        high
...

--- Turn 2 ---
Customer: It's jane@example.com
ScaleDown extracted:
    order_id: A-1042
    error_message: ERR_PAYMENT_DECLINED
    steps_tried: tried two different cards
  + customer_email: jane@example.com
    issue_type: billing
    sentiment: frustrated
Ticket filed: T-1006 -> Billing (high priority): Customer Jane (jane@example.com)
experiencing repeated ERR_PAYMENT_DECLINED on order A-1042 despite trying two
different cards; frustrated after third contact.
```

### From a voicemail

`--audio` transcribes a recording locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(no API key; the model downloads on first run), then files the transcript as a ticket
with the same extraction. On a Mac you can make a sample voicemail with the built-in
`say` voice:

```bash
say -o voicemail.aiff "Hi, this is Marco calling about order B 2207. The app crashes \
every time I open my cart, it says sync timeout. I already reinstalled it twice. \
Honestly this is really frustrating. My email is marco at example dot com. Thanks."

pip install faster-whisper "av<19"  # PyAV 19 dropped an option faster-whisper uses
python examples/support_triage.py --audio voicemail.aiff
```

Output from a real run:

```text
Transcript: Hi, this is Marco calling about Order B2207. The app crashes every time I
open my cart that says sync timeout. I already reinstalled it twice. Honestly this is
really frustrating. My email is Marco at example.com. Thanks.

ScaleDown extracted:
    order_id: B2207
    error_message: sync timeout
    steps_tried: reinstalled it twice
    customer_email: Marco at example.com
    issue_type: technical
    sentiment: frustrated

Ticket filed: T-1001 -> Tech Support (high priority): ...
```

Whisper writes the email as it was spoken, and extraction copies it as is rather than
guessing an address. The agent's callback note asks the team to confirm it.

## What gets sent

| Middleware | Sent to ScaleDown | Endpoint |
|---|---|---|
| Summarization | The older messages being summarized | `/summarization/abstractive` |
| Compression | The system message and the latest user message | `/compress/raw/` |
| Extraction | The user's messages (not the agent's replies) and your schema | `/extract` |

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
