"""Thin synchronous REST client for the ScaleDown API.

Endpoint shapes follow https://docs.scaledown.ai/api-reference/.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.scaledown.xyz"
DEFAULT_TIMEOUT = 60.0


class ScaledownAPIError(Exception):
    """Raised when a ScaleDown API request fails."""


class ScaledownClient:
    """Minimal client for the ScaleDown summarization and compression endpoints.

    Args:
        api_key: ScaleDown API key. Falls back to the `SCALEDOWN_API_KEY`
            environment variable.
        base_url: API base URL. Falls back to the `SCALEDOWN_BASE_URL`
            environment variable, then `https://api.scaledown.xyz`.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        api_key = api_key or os.environ.get("SCALEDOWN_API_KEY")
        if not api_key:
            raise ValueError(
                "No ScaleDown API key found. Pass `api_key=...` or set the "
                "SCALEDOWN_API_KEY environment variable. Get a key at "
                "https://scaledown.ai/dashboard"
            )
        self.api_key = api_key
        self.base_url = (
            base_url or os.environ.get("SCALEDOWN_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            response = requests.post(
                url, json=body, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ScaledownAPIError(f"ScaleDown request to {path} failed: {e}") from e
        except ValueError as e:
            raise ScaledownAPIError(
                f"ScaleDown returned a non-JSON response from {path}"
            ) from e
        if not isinstance(data, dict):
            raise ScaledownAPIError(
                f"ScaleDown returned an unexpected response from {path}: {data!r}"
            )
        return data

    def summarize(
        self,
        text: str,
        instructions: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Abstractively summarize `text`. Returns the summary string."""
        body: dict[str, Any] = {"text": text, "max_tokens": max_tokens}
        if instructions is not None:
            body["instructions"] = instructions
        data = self._post("/summarization/abstractive", body)
        try:
            return str(data["summary"])
        except KeyError as e:
            raise ScaledownAPIError(
                "ScaleDown summarization response missing 'summary'"
            ) from e

    def extract(
        self,
        text: str,
        entities: Mapping[str, Any],
        instruction: str | None = None,
        threshold: float | None = None,
        top_n: int | None = None,
        context_chars: int | None = None,
    ) -> dict[str, Any]:
        """Extract entities from `text` using a custom schema.

        `entities` maps each field name to a description string, an object
        (`description` / `threshold` / `top_n`), a nested schema, or a
        classification key (an object with a `labels` list, which ScaleDown
        routes to its classification model).

        Returns the full response dict (`entities`, `structured_result`, ...).
        Optional parameters left as `None` are omitted, so the API defaults apply.
        """
        body: dict[str, Any] = {"text": text, "entities": dict(entities)}
        optional = {
            "instruction": instruction,
            "threshold": threshold,
            "top_n": top_n,
            "context_chars": context_chars,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return self._post("/extract", body)

    def compress(self, context: str, prompt: str, rate: Any = "auto") -> dict[str, Any]:
        """Compress `context` relative to `prompt`.

        Returns the full response dict. Callers must check
        `response["successful"]` before trusting `response["compressed_prompt"]`.

        The live API returns the compressed text and token counts nested under
        `results` (the published docs show them at the top level). Nested
        fields are copied to the top level, so both shapes read the same way.
        """
        body = {"context": context, "prompt": prompt, "scaledown": {"rate": rate}}
        data = self._post("/compress/raw/", body)
        results = data.get("results")
        if isinstance(results, dict):
            for key, value in results.items():
                data.setdefault(key, value)
        return data


def _resolve_client(
    client: ScaledownClient | None, api_key: str | None, base_url: str | None
) -> ScaledownClient:
    if client is not None:
        return client
    return ScaledownClient(api_key=api_key, base_url=base_url)
