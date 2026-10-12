"""Unit tests for `ScaledownSummarizer`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_scaledown import ScaledownAPIError, ScaledownClient, ScaledownSummarizer
from langchain_scaledown.summarizer import DEFAULT_INSTRUCTIONS


@pytest.fixture
def client() -> ScaledownClient:
    return ScaledownClient(api_key="test-key")


def test_string_input_is_sent_as_is(client: ScaledownClient) -> None:
    summarizer = ScaledownSummarizer(client=client)
    with patch.object(ScaledownClient, "summarize", return_value="sum") as summarize:
        assert summarizer.invoke("raw transcript") == "sum"
    summarize.assert_called_once_with(
        "raw transcript", instructions=DEFAULT_INSTRUCTIONS, max_tokens=2048
    )


def test_messages_are_formatted_as_xml(client: ScaledownClient) -> None:
    summarizer = ScaledownSummarizer(instructions=None, max_tokens=100, client=client)
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    with patch.object(ScaledownClient, "summarize", return_value="sum") as summarize:
        summarizer.invoke(messages)
    text = summarize.call_args.args[0]
    assert "hi" in text and "hello" in text
    assert text.startswith("<")
    assert summarize.call_args.kwargs == {"instructions": None, "max_tokens": 100}


async def test_ainvoke(client: ScaledownClient) -> None:
    summarizer = ScaledownSummarizer(client=client)
    with patch.object(ScaledownClient, "summarize", return_value="sum"):
        assert await summarizer.ainvoke("raw transcript") == "sum"
        assert await summarizer.ainvoke([HumanMessage(content="hi")]) == "sum"


def test_errors_propagate(client: ScaledownClient) -> None:
    summarizer = ScaledownSummarizer(client=client)
    with (
        patch.object(
            ScaledownClient, "summarize", side_effect=ScaledownAPIError("down")
        ),
        pytest.raises(ScaledownAPIError, match="down"),
    ):
        summarizer.invoke("raw transcript")


def test_with_fallbacks(client: ScaledownClient) -> None:
    summarizer = ScaledownSummarizer(client=client).with_fallbacks(
        [ScaledownSummarizer(client=client)], exceptions_to_handle=(ScaledownAPIError,)
    )
    with patch.object(
        ScaledownClient,
        "summarize",
        side_effect=[ScaledownAPIError("down"), "fallback summary"],
    ):
        assert summarizer.invoke("raw transcript") == "fallback summary"
