"""Mutation-safety test. See docs/DESIGN.md "The record/mock boundary" → "Side-effect safety".

A tool flagged mutating must never go LIVE just because the replay's run-wide default is LIVE —
that's exactly the case the design calls out ("never inherited from a default"). It must still
refuse without an explicit, per-tool opt-in; an explicit per_tool override is what "explicit
per-run override" means, and that must actually work (otherwise the gate is just "always block",
which is a different and useless bug).
"""

from __future__ import annotations

import pytest
from ctx_capture.capture import TraceRecorder

from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import build_baseline_responses, record_toy_agent
from trace_replay.engine import resume_from
from trace_replay.modes import ToolModeConfig
from trace_replay.safety import MutationSafetyError


def _record_source_trace():
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    return recorder.trace


def _make_delete_file(calls: list[str]):
    def delete_file(path: str) -> dict:
        calls.append(path)
        return {"deleted": path}

    return delete_file


def test_mutating_tool_refuses_under_global_live_default():
    source_trace = _record_source_trace()
    calls: list[str] = []
    delete_file = _make_delete_file(calls)

    def entrypoint(ctx):
        ctx.recorder.begin_step()
        ctx.tool_router.call("delete_file", {"path": "/etc/important"})
        ctx.recorder.end_step()

    with pytest.raises(MutationSafetyError):
        resume_from(
            source_trace,
            resume_step_index=0,
            entrypoint=entrypoint,
            tool_modes=ToolModeConfig(default="LIVE"),  # global, not per-tool
            live_tool_fns={"delete_file": delete_file},
            tool_safety_registry={"delete_file": True},
        )

    assert calls == [], "mutating tool must never execute when LIVE came from the run-wide default"


def test_mutating_tool_runs_with_explicit_per_tool_override():
    source_trace = _record_source_trace()
    calls: list[str] = []
    delete_file = _make_delete_file(calls)

    def entrypoint(ctx):
        ctx.recorder.begin_step()
        ctx.tool_router.call("delete_file", {"path": "/etc/important"})
        ctx.recorder.end_step()

    resume_from(
        source_trace,
        resume_step_index=0,
        entrypoint=entrypoint,
        tool_modes=ToolModeConfig(default="MOCK", per_tool={"delete_file": "LIVE"}),  # explicit, by name
        live_tool_fns={"delete_file": delete_file},
        tool_safety_registry={"delete_file": True},
    )

    assert calls == ["/etc/important"], "explicit per-tool LIVE override must actually run the tool"
