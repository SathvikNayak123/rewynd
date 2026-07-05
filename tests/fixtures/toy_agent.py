"""A toy 6-step tool-using agent: two versions of the same logic.

`record_toy_agent` runs it directly against the ctx-capture SDK (no trace-replay involved) to
produce a source trace, exactly like a real instrumented agent would.

`build_replay_entrypoint` wraps the same logic behind the integration contract (docs/DESIGN.md
"Integration contract"): it builds its own model client and routes tool calls through
`ctx.tool_router` instead of calling them directly. The `canned_responses` it's given simulate
"what the live model would say" — steps 0-3 are identical between the baseline and override
response sets on purpose (an overridden system prompt shouldn't retroactively change history),
step 4 diverges (the override changes what the agent asks for), simulating a real model reacting
differently to a different system prompt.
"""

from __future__ import annotations

import json
from typing import Any

from tests.fixtures.fake_client import FakeOpenAIClient, FakeResponse
from trace_replay.harness import ReplayContext

TOTAL_STEPS = 6


def tool_impl(query: str) -> str:
    return f"result for {query}"


def _tool_call_response(step: int, query: str) -> FakeResponse:
    return FakeResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 10 + step, "completion_tokens": 5, "total_tokens": 15 + step},
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
            "usage": {"prompt_tokens": 10 + step, "completion_tokens": 8, "total_tokens": 18 + step},
            "choices": [{"message": {"role": "assistant", "content": text, "tool_calls": None}}],
        }
    )


def build_baseline_responses() -> list[FakeResponse]:
    return [
        _tool_call_response(0, "q0"),
        _tool_call_response(1, "q1"),
        _tool_call_response(2, "q2"),
        _tool_call_response(3, "q3"),
        _tool_call_response(4, "q4"),
        _final_response(5, "final answer"),
    ]


def build_override_responses() -> list[FakeResponse]:
    """Identical to baseline through step 3; diverges from step 4 onward, simulating the live
    model reacting to a system-prompt override applied at the step-3 resume point."""
    return [
        _tool_call_response(0, "q0"),
        _tool_call_response(1, "q1"),
        _tool_call_response(2, "q2"),
        _tool_call_response(3, "q3"),
        _tool_call_response(4, "q4-override"),
        _final_response(5, "final answer (override)"),
    ]


def record_toy_agent(recorder: Any, capturing_client: Any) -> list[dict]:
    """Runs the toy agent directly against the ctx-capture SDK to build a source trace."""
    messages: list[dict] = [{"role": "system", "content": "You are a helpful test agent."}]

    for step in range(TOTAL_STEPS):
        recorder.begin_step()
        response = capturing_client.chat.completions.create(model="toy-model", messages=messages, temperature=0)
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

    return messages


def build_replay_entrypoint(canned_responses: list[FakeResponse]):
    """Returns an Entrypoint (docs/DESIGN.md "Integration contract") bound to a specific
    canned-response script. The entrypoint owns building its own client — trace-replay only
    injects `ctx.messages` and `ctx.tool_router`."""

    def entrypoint(ctx: ReplayContext) -> None:
        remaining_responses = canned_responses[ctx.resume_step_index :]
        fake_client = FakeOpenAIClient(remaining_responses)
        capturing_client = ctx.recorder.wrap_client(fake_client, provider="toy-provider")
        messages = ctx.messages

        model = ctx.overrides.model or "toy-model"
        params = {"temperature": 0, **ctx.overrides.params}

        for step in range(ctx.resume_step_index, ctx.resume_step_index + len(remaining_responses)):
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
