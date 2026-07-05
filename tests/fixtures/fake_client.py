"""Minimal OpenAI-compatible fake client, call-index-driven canned responses. Mirrors
ctx-capture's own tests/fixtures/fake_client.py — same pattern, local copy since ctx-capture's
test fixtures aren't part of its importable package surface.
"""

from __future__ import annotations

import copy
from typing import Any


class FakeResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(dict(self))


class _FakeCompletions:
    def __init__(self, canned_responses: list[FakeResponse]) -> None:
        self._canned = canned_responses
        self._call_index = 0
        self.call_log: list[dict[str, Any]] = []

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> FakeResponse:
        self.call_log.append({"model": model, "messages": copy.deepcopy(messages), "params": copy.deepcopy(params)})
        response = self._canned[self._call_index]
        self._call_index += 1
        return response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, canned_responses: list[FakeResponse]) -> None:
        self.chat = _FakeChat(_FakeCompletions(canned_responses))

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self.chat.completions.call_log
