"""Replay-of-replay must work with no special-casing (docs/DESIGN.md decision #2): a replay's
output is just a trace, so resuming from a replayed trace is exactly the same code path as
resuming from an originally-recorded one.
"""

from __future__ import annotations

from ctx_capture.capture import TraceRecorder

from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import build_baseline_responses, build_replay_entrypoint, record_toy_agent
from trace_replay.engine import resume_from
from trace_replay.modes import ToolModeConfig


def _record_source_trace():
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    return recorder.trace


def test_resuming_from_a_replayed_trace_works():
    source_trace = _record_source_trace()

    gen1 = resume_from(
        source_trace,
        resume_step_index=0,
        entrypoint=build_replay_entrypoint(build_baseline_responses()),
        tool_modes=ToolModeConfig(default="MOCK"),
    )
    assert gen1.metadata["trace_replay"]["source_trace_id"] == source_trace.trace_id

    # gen1 is a normal Trace: resume from it exactly as if it were an originally-recorded trace.
    gen2 = resume_from(
        gen1,
        resume_step_index=2,
        entrypoint=build_replay_entrypoint(build_baseline_responses()),
        tool_modes=ToolModeConfig(default="MOCK"),
    )

    assert gen2.metadata["trace_replay"]["source_trace_id"] == gen1.trace_id
    assert len(gen2.steps) == len(gen1.steps) == 6
    for step_index in range(2):
        assert gen2.steps[step_index].model_dump_json() == gen1.steps[step_index].model_dump_json()
