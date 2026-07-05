"""THE replay fidelity test. See docs/DESIGN.md "Evaluating the server itself".

Replay step 0 of a recorded run in full-MOCK mode: the resulting trajectory must reproduce
exactly — same tool calls with the same args, same message content at every step, same final
answer. This is trace-replay's equivalent of ctx-capture's capture-fidelity acceptance test and
is CI-blocking forever: it's the proof that "the past is frozen" isn't just a design claim.
"""

from __future__ import annotations

import json
from typing import Any

from ctx_capture.capture import TraceRecorder

from tests.fixtures.toy_agent import build_baseline_responses, build_replay_entrypoint, record_toy_agent
from tests.fixtures.fake_client import FakeOpenAIClient
from trace_replay.engine import resume_from
from trace_replay.modes import ToolModeConfig


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _record_source_trace():
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    return recorder.trace


def test_full_mock_replay_from_step_zero_reproduces_trajectory_exactly():
    source_trace = _record_source_trace()
    assert len(source_trace.steps) == 6

    entrypoint = build_replay_entrypoint(build_baseline_responses())
    replayed_trace = resume_from(
        source_trace,
        resume_step_index=0,
        entrypoint=entrypoint,
        tool_modes=ToolModeConfig(default="MOCK"),
    )

    assert len(replayed_trace.steps) == len(source_trace.steps)

    for step_index in range(len(source_trace.steps)):
        source_step = source_trace.steps[step_index]
        replayed_step = replayed_trace.steps[step_index]

        assert _canonical(source_step.model_call.messages) == _canonical(replayed_step.model_call.messages), (
            f"step {step_index}: replayed messages diverged from the recording"
        )
        assert len(source_step.tool_calls) == len(replayed_step.tool_calls)
        for source_tc, replayed_tc in zip(source_step.tool_calls, replayed_step.tool_calls):
            assert source_tc.tool_name == replayed_tc.tool_name
            assert _canonical(source_tc.args_raw) == _canonical(replayed_tc.args_raw)
            assert source_tc.result_as_inserted == replayed_tc.result_as_inserted

    source_final = source_trace.steps[-1].model_call.response["choices"][0]["message"]["content"]
    replayed_final = replayed_trace.steps[-1].model_call.response["choices"][0]["message"]["content"]
    assert source_final == replayed_final == "final answer"

    # No divergences: every live tool call matched a recorded one.
    assert replayed_trace.metadata["trace_replay"]["divergences"] == []

    # Replay cost accounting: the replay reports its own cost, comparable to the source run's.
    replay_meta = replayed_trace.metadata["trace_replay"]
    assert replay_meta["cost_usd"] > 0
    assert replay_meta["source_trace_cost_usd"] > 0
    for step in replayed_trace.steps:
        assert step.model_call.cost_usd is not None
