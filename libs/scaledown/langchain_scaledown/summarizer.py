"""ScaleDown summarizer for LangChain and Deep Agents summarization middleware."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_scaledown._client import ScaledownClient, _resolve_client

DEFAULT_INSTRUCTIONS = (
    "This summary replaces an AI agent's conversation history, and the agent "
    "will continue working from it. Use these sections, writing 'None' for any "
    "section with nothing to report:\n"
    "## SESSION INTENT: the user's overall goal.\n"
    "## SUMMARY: key context, decisions and the reasons for them, and options "
    "that were rejected.\n"
    "## ARTIFACTS: files or resources created, modified, or accessed, with "
    "file paths and the changes made.\n"
    "## NEXT STEPS: what remains to be done."
)
"""Default `instructions`, adapted from LangChain's default summary prompt."""


class ScaledownSummarizer(Runnable[str | Sequence[BaseMessage], str]):
    """Summarize text or messages with ScaleDown's `sd_summarize`.

    Pass it as the `summarizer` of LangChain's or Deep Agents'
    `SummarizationMiddleware`. The middleware keeps
    deciding when to summarize and which messages to keep; ScaleDown writes the
    summary instead of a second LLM call.

    The input is either a string, sent to ScaleDown as is, or a sequence of
    messages, formatted with `get_buffer_string(..., format="xml")`.

    Errors raise `ScaledownAPIError`. To fall back to another summarizer, use
    `with_fallbacks`.

    Note:
        `sd_summarize` is in private preview. Contact ScaleDown to enable it on
        your account.

    Args:
        instructions: Rules appended to ScaleDown's base summarization
            instructions. Defaults to `DEFAULT_INSTRUCTIONS`, which asks for the
            same sections as LangChain's default summary prompt. Pass `None` to
            use ScaleDown's default behavior only.
        max_tokens: Maximum length of the generated summary.
        client: Preconfigured `ScaledownClient`. If omitted, one is created from
            `api_key` / `base_url` (or the `SCALEDOWN_API_KEY` /
            `SCALEDOWN_BASE_URL` environment variables).

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_scaledown import ScaledownSummarizer

        agent = create_agent(
            "openai:gpt-6-sol",
            tools=[...],
            middleware=[
                SummarizationMiddleware(
                    "openai:gpt-6-sol",
                    trigger=("fraction", 0.8),
                    keep=("fraction", 0.1),
                    trim_tokens_to_summarize=None,
                    summarizer=ScaledownSummarizer(),
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        instructions: str | None = DEFAULT_INSTRUCTIONS,
        max_tokens: int = 2048,
        client: ScaledownClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.instructions = instructions
        self.max_tokens = max_tokens
        self.client = _resolve_client(client, api_key, base_url)

    def _summarize(self, input: str | Sequence[BaseMessage]) -> str:  # noqa: A002
        text = (
            input if isinstance(input, str) else get_buffer_string(input, format="xml")
        )
        return self.client.summarize(
            text, instructions=self.instructions, max_tokens=self.max_tokens
        )

    def invoke(
        self,
        input: str | Sequence[BaseMessage],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        """Summarize `input` with ScaleDown."""
        return self._call_with_config(self._summarize, input, config)
