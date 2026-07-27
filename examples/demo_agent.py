"""A runnable demo agent for manual verification (see docs/DESIGN.md "Manual verification" /
session task list). Same shape as tests/fixtures/toy_agent.py, kept separate because test
fixtures aren't meant to be imported from outside the test suite — this module is what a real
`--entrypoints examples.demo_agent:ENTRYPOINTS` points at.

Run `python examples/seed_demo_trace.py` first to record a real source trace, then point the MCP
server at it (see .mcp.json / README) and call `replay_from_step` from a real MCP client.
"""

from __future__ import annotations

import json
from typing import Any

AGENT_NAME = "demo-agent"


class FakeResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        import copy

        return copy.deepcopy(dict(self))


class DemoClient:
    """Deterministic stand-in for a real model client: a fixed 4-step script (research a topic,
    then answer), so a manual replay session is reproducible without needing a live API key."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self._call_index = 0

    class _Completions:
        def __init__(self, outer: "DemoClient") -> None:
            self._outer = outer

        def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> FakeResponse:
            response = self._outer._responses[self._outer._call_index]
            self._outer._call_index += 1
            return response

    @property
    def chat(self):
        class _Chat:
            def __init__(self, outer):
                self.completions = DemoClient._Completions(outer)

        return _Chat(self)


def tool_impl(query: str) -> str:
    return f"search results for: {query}"


def _tool_call_response(step: int, query: str) -> FakeResponse:
    return FakeResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 20 + step, "completion_tokens": 6, "total_tokens": 26 + step},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{step}",
                                "type": "function",
                                "function": {"name": "search", "arguments": json.dumps({"query": query})},
                            }
                        ],
                    }
                }
            ],
        }
    )


def _final_response(step: int, text: str) -> FakeResponse:
    return FakeResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 20 + step, "completion_tokens": 10, "total_tokens": 30 + step},
            "choices": [{"message": {"role": "assistant", "content": text, "tool_calls": None}}],
        }
    )


def build_baseline_responses() -> list[FakeResponse]:
    return [
        _tool_call_response(0, "best hiking trails colorado"),
        _tool_call_response(1, "hiking trail difficulty ratings"),
        _tool_call_response(2, "hiking trail permit requirements"),
        _final_response(3, "Recommend Mount Elbert for an experienced hiker; permit not required."),
    ]


def _budget_final_response() -> FakeResponse:
    """The canned reply this fixed script gives when its system prompt is overridden toward
    "budget-friendly" — so the README quickstart's documented override actually demonstrates a
    changed outcome, not just a mechanically-completed replay. Still a deterministic stand-in
    for a model, not a real one: it keys off a literal substring, nothing more."""
    return _final_response(3, "Recommend Bear Lake Trailhead (free, no permit) for a budget-conscious hiker.")


def record_demo_agent(recorder: Any, capturing_client: Any) -> None:
    """Runs the demo agent directly against the ctx-capture SDK to build a source trace. See
    examples/seed_demo_trace.py."""
    messages: list[dict] = [{"role": "system", "content": "You are a helpful trip-planning agent."}]

    for _ in range(4):
        recorder.begin_step()
        response = capturing_client.chat.completions.create(model="demo-model", messages=messages, temperature=0)
        choice_message = response["choices"][0]["message"]
        messages.append(
            {
                "role": "assistant",
                "content": choice_message.get("content"),
                "tool_calls": choice_message.get("tool_calls"),
            }
        )
        tool_calls = choice_message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc["function"]["arguments"])
                wrapped_tool = recorder.wrap_tool(tool_impl, tool_name=tc["function"]["name"])
                result, tool_call_id = wrapped_tool(**args)
                recorder.record_insertion(tool_call_id, result)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        recorder.end_step()


def build_replay_entrypoint(canned_responses: list[FakeResponse]):
    """The integration-contract entrypoint (docs/DESIGN.md "Integration contract"): builds its
    own client and routes tool calls through ctx.tool_router instead of calling them directly."""
    from trace_replay.harness import ReplayContext

    def entrypoint(ctx: ReplayContext) -> None:
        remaining = list(canned_responses[ctx.resume_step_index :])
        if ctx.overrides.system_prompt and "budget" in ctx.overrides.system_prompt.lower():
            remaining[-1] = _budget_final_response()
        client = DemoClient(remaining)
        capturing_client = ctx.recorder.wrap_client(client, provider="demo-provider")
        messages = ctx.messages
        model = ctx.overrides.model or "demo-model"
        params = {"temperature": 0, **ctx.overrides.params}

        for _ in range(ctx.resume_step_index, ctx.resume_step_index + len(remaining)):
            ctx.recorder.begin_step()
            response = capturing_client.chat.completions.create(model=model, messages=messages, **params)
            choice_message = response["choices"][0]["message"]
            messages.append(
                {
                    "role": "assistant",
                    "content": choice_message.get("content"),
                    "tool_calls": choice_message.get("tool_calls"),
                }
            )
            tool_calls = choice_message.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    args = json.loads(tc["function"]["arguments"])
                    result = ctx.tool_router.call(tc["function"]["name"], args)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            ctx.recorder.end_step()

    return entrypoint


# What --entrypoints examples.demo_agent:ENTRYPOINTS points the MCP server at.
ENTRYPOINTS = {AGENT_NAME: build_replay_entrypoint(build_baseline_responses())}
