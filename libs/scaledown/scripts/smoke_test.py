# mypy: ignore-errors
# (Throwaway dev script, not shipped in the package; typed loosely on purpose.)
"""Smoke test: run ScaleDown middleware inside a real `create_agent()` agent.

Not part of the test suite. Uses FakeListChatModel and mocks the ScaleDown
client methods, so no API key or network is needed:

    poetry run python scripts/smoke_test.py
"""

import asyncio
import os
from unittest.mock import patch

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_scaledown import (
    ScaledownClient,
    ScaledownCompressionMiddleware,
    ScaledownExtractionMiddleware,
    ScaledownSummarizationMiddleware,
)

os.environ.setdefault("SCALEDOWN_API_KEY", "dummy-key-for-smoke-test")


class Spy(AgentMiddleware):
    """Innermost middleware: records the request the model actually receives."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    def wrap_model_call(self, request, handler):
        self.seen.append(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self.seen.append(request)
        return await handler(request)


def run_summarization(use_async: bool) -> None:
    spy = Spy()
    model = FakeListChatModel(responses=["final answer", "follow-up answer"])
    agent = create_agent(
        model=model,
        middleware=[
            ScaledownSummarizationMiddleware(
                model, trigger=("messages", 6), keep=("messages", 2)
            ),
            spy,
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "s1"}}
    history = []
    for i in range(3):
        history += [HumanMessage(f"question {i}"), AIMessage(f"answer {i}")]
    history.append(HumanMessage("latest question"))

    def run(inputs):
        if use_async:
            return asyncio.run(agent.ainvoke(inputs, config))
        return agent.invoke(inputs, config)

    with patch.object(
        ScaledownClient, "summarize", return_value="MOCK SUMMARY"
    ) as summarize:
        first = run({"messages": history})
        sent = spy.seen[-1].messages
        second = run({"messages": [HumanMessage("one more question")]})

    print(f"[summarization {'async' if use_async else 'sync'}]")
    print(f"  sd_summarize called: {summarize.call_count}x across 2 turns")
    print(f"  model saw {len(sent)} messages (input had {len(history)}):")
    for m in sent:
        print(f"    {type(m).__name__}: {m.text[:60]!r}")
    print(
        f"  state after turn 1: {len(first['messages'])} messages; "
        f"after turn 2: {len(second['messages'])}"
    )
    print(f"  final reply: {second['messages'][-1].text!r}")
    # Summarized once and persisted: turn 2 builds on the summary, no new call.
    assert summarize.call_count == 1
    assert len(sent) == 3 and "MOCK SUMMARY" in sent[0].text
    assert len(first["messages"]) == 4
    assert "MOCK SUMMARY" in second["messages"][0].text
    assert second["messages"][-1].text == "follow-up answer"


def run_compression(use_async: bool) -> None:
    spy = Spy()
    big_context = "The launch code is 7-4-1. " + "Irrelevant filler. " * 300
    agent = create_agent(
        model=FakeListChatModel(responses=["the code is 7-4-1"]),
        system_prompt=big_context,
        middleware=[ScaledownCompressionMiddleware(min_context_chars=2000), spy],
    )
    with patch.object(
        ScaledownClient,
        "compress",
        return_value={"successful": True, "compressed_prompt": "launch code: 7-4-1"},
    ) as compress:
        inputs = {"messages": [HumanMessage("What is the launch code?")]}
        result = (
            asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
        )

    system = spy.seen[-1].system_message
    print(f"[compression {'async' if use_async else 'sync'}]")
    print(f"  sd_compress called: {compress.call_count}x")
    print(f"  compress prompt arg: {compress.call_args.args[1]!r}")
    print(f"  system prompt: {len(big_context)} chars -> {len(system.text)} chars")
    print(f"  model saw system message: {system.text!r}")
    print(f"  final reply: {result['messages'][-1].text!r}")
    assert compress.call_count == 1
    assert compress.call_args.args == (big_context, "What is the launch code?")
    assert system.text == "launch code: 7-4-1"


class ToolCallingFake(GenericFakeChatModel):
    """GenericFakeChatModel that accepts bind_tools, so it can emit tool calls."""

    def bind_tools(self, tools, **kwargs):
        return self


TICKET_ENTITIES = {
    "order_id": "Order number the customer mentions",
    "error_message": "Exact error message or code the customer saw",
    "customer_email": "Customer's email address",
    "issue_type": {"labels": ["billing", "technical", "account"]},
    "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
}


def extract_response(text, entities, **kwargs):
    """Fake /extract: returns only spans that are actually present in `text`."""
    found = [
        ("A-1042", "order_id"),
        ("ERR_PAYMENT_DECLINED", "error_message"),
        ("jane@example.com", "customer_email"),
    ]
    spans = [
        {"text": value, "type": key, "confidence": 0.95}
        for value, key in found
        if value in text
    ]
    spans += [
        {"text": "billing", "type": "issue_type", "confidence": 0.93},
        {"text": "frustrated", "type": "sentiment", "confidence": 0.9},
    ]
    return {"entities": spans, "structured_result": None, "ocr_text": None}


def run_extraction(use_async: bool) -> None:
    filed = []

    @tool
    def file_ticket(runtime: ToolRuntime) -> str:
        """File a support ticket from the fields extracted so far."""
        fields = runtime.state["scaledown_extraction"]["fields"]
        filed.append(dict(fields))
        return f"Filed ticket #{len(filed)}"

    spy = Spy()
    model = ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "file_ticket", "args": {}, "id": "c1"}],
                ),
                AIMessage(content="Filed. What's your email?"),
                AIMessage(content="Thanks, updated."),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=[file_ticket],
        system_prompt="You are a support agent.",
        middleware=[
            ScaledownExtractionMiddleware(TICKET_ENTITIES, inject_into_prompt=True),
            spy,
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}}
    turns = [
        "Order A-1042 failed with ERR_PAYMENT_DECLINED. Third time, I'm fed up.",
        "It's jane@example.com",
    ]
    with patch.object(
        ScaledownClient, "extract", side_effect=extract_response
    ) as extract:
        results = []
        for turn in turns:
            inputs = {"messages": [HumanMessage(turn)]}
            results.append(
                asyncio.run(agent.ainvoke(inputs, config))
                if use_async
                else agent.invoke(inputs, config)
            )

    first, second = (r["scaledown_extraction"]["fields"] for r in results)
    print(f"[extraction {'async' if use_async else 'sync'}]")
    print(f"  sd_extract called: {extract.call_count}x (once per turn)")
    print(f"  turn 1 fields: {first}")
    print(f"  turn 2 fields: {second}")
    print(f"  file_ticket tool read from state: {filed[0]}")
    injected = spy.seen[0].system_message.text
    print("  model saw system prompt:")
    for line in injected.splitlines():
        print(f"    {line}")
    assert extract.call_count == 2
    # Turn 2 re-extracts over the whole conversation, human messages only.
    assert turns[0] in extract.call_args.args[0]
    assert "What's your email" not in extract.call_args.args[0]
    assert first["customer_email"] is None
    assert second["customer_email"] == "jane@example.com"
    assert second["order_id"] == "A-1042"
    assert filed[0]["issue_type"] == "billing"
    assert injected.startswith("You are a support agent.")
    assert '- order_id: "A-1042"' in injected


if __name__ == "__main__":
    for use_async in (False, True):
        run_summarization(use_async)
        run_compression(use_async)
        run_extraction(use_async)
    print("\nSMOKE TEST PASSED")
