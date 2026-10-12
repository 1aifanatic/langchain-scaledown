"""Integration tests against the live ScaleDown API.

Require `SCALEDOWN_API_KEY`. Run with `make integration_tests`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from langchain_scaledown import (
    ScaledownClient,
    ScaledownCompressionMiddleware,
    ScaledownExtractionMiddleware,
    ScaledownSummarizationMiddleware,
)

TICKET_TEXT = (
    "Hi, order A-1042 failed at checkout with ERR_PAYMENT_DECLINED. "
    "This is the third time and I'm really fed up. "
    "You can reach me at jane@example.com."
)

TICKET_ENTITIES: dict[str, Any] = {
    "order_id": "Order number the customer mentions",
    "error_message": "Exact error message or code the customer saw",
    "customer_email": "Customer's email address",
    "issue_type": {
        "labels": [
            {"name": "billing", "rubric": "Is this about a charge or payment?"},
            {"name": "technical", "rubric": "Is this about a bug or outage?"},
            {"name": "account", "rubric": "Is this about login or settings?"},
        ]
    },
    "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
}

DOCS = (
    "Refund policy: purchases can be refunded within 30 days of delivery. "
    + "Our company was founded in a small garage and has grown steadily. " * 60
)


@pytest.fixture(scope="module")
def client() -> ScaledownClient:
    return ScaledownClient()


def test_extract(client: ScaledownClient) -> None:
    response = client.extract(TICKET_TEXT, TICKET_ENTITIES, context_chars=0)
    found = {e["type"]: e["text"] for e in response["entities"]}
    # Extractive fields are verbatim spans from the input.
    assert found["order_id"] in TICKET_TEXT
    assert "A-1042" in found["order_id"]
    assert found["customer_email"] == "jane@example.com"
    # Classification keys come back as one of the labels.
    assert found["issue_type"] in {"billing", "technical", "account"}
    assert found["sentiment"] in {"frustrated", "neutral", "satisfied"}


def test_compress(client: ScaledownClient) -> None:
    response = client.compress(DOCS, "How long is the refund window?")
    assert response["successful"] is True, response
    compressed = response.get("compressed_prompt")
    assert isinstance(compressed, str), f"no compressed_prompt in {response!r}"
    assert len(compressed) < len(DOCS)


def test_summarize(client: ScaledownClient) -> None:
    """`sd_summarize` is in private preview; the key needs access to it."""
    summary = client.summarize("The customer asked about a refund. " * 30)
    assert isinstance(summary, str) and summary.strip()


def test_extraction_middleware_in_agent() -> None:
    agent = create_agent(
        model=FakeListChatModel(responses=["Thanks, looking into it."]),
        middleware=[ScaledownExtractionMiddleware(TICKET_ENTITIES, context_chars=0)],
    )
    result = agent.invoke({"messages": [HumanMessage(TICKET_TEXT)]})
    fields = result["scaledown_extraction"]["fields"]
    assert "A-1042" in fields["order_id"]
    assert fields["customer_email"] == "jane@example.com"
    assert fields["sentiment"] in {"frustrated", "neutral", "satisfied"}


def test_compression_middleware_in_agent() -> None:
    seen: list[ModelRequest] = []

    @wrap_model_call
    def spy(
        request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        seen.append(request)
        return handler(request)

    agent = create_agent(
        model=FakeListChatModel(responses=["30 days."]),
        system_prompt=DOCS,
        middleware=[ScaledownCompressionMiddleware(min_context_chars=500), spy],
    )
    result = agent.invoke(
        {"messages": [HumanMessage("How long is the refund window?")]}
    )
    assert result["messages"][-1].text == "30 days."
    # The model must have received the compressed context, not the original.
    system = seen[0].system_message
    assert system is not None
    assert len(system.text) < len(DOCS)


def test_summarization_middleware_in_agent() -> None:
    history: list[Any] = []
    for i in range(6):
        history.append(HumanMessage(f"Question {i} about my refund for order A-1042."))
        history.append(FakeListChatModel(responses=[f"Answer {i}."]).invoke("x"))
    history.append(HumanMessage("So how long do I have?"))
    model = FakeListChatModel(responses=["30 days."])
    agent = create_agent(
        model=model,
        middleware=[
            ScaledownSummarizationMiddleware(
                model, trigger=("messages", 10), keep=("messages", 2)
            )
        ],
    )
    result = agent.invoke({"messages": history})
    # The older history was replaced by one summary message in state.
    assert len(result["messages"]) < len(history)
    assert result["messages"][-1].text == "30 days."
