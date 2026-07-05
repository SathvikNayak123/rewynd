"""Override replay test. See docs/DESIGN.md "The record/mock boundary" → "Divergence handling".

Replays from step 3 with a modified system prompt. Steps 0-2 must be byte-identical to the
source trace (they're copied verbatim — this is ctx-capture's own fidelity guarantee, inherited
for free, not re-derived). The overridden prompt causes the (simulated) live model to ask for
something different starting at step 4, which has no match in the recording: a divergence.
Policy is stub_with_warning for this test, so the run logs the divergence and completes instead
of crashing.
"""

from __future__ import annotations

import json
from typing import Any

from ctx_capture.capture import TraceRecorder

from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import build_baseline_responses, build_override_responses, build_replay_entrypoint, record_toy_agent
from trace_replay.engine import resume_from
from trace_replay.harness import ReplayOverrides
from trace_replay.modes import ToolModeConfig


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _record_source_trace():
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    return recorder.trace


def test_override_from_step_three_preserves_prefix_and_logs_divergence():
    source_trace = _record_source_trace()

    entrypoint = build_replay_entrypoint(build_override_responses())
    tool_modes = ToolModeConfig(default="MOCK", stub_results={"search": "stubbed result"})

    replayed_trace = resume_from(
        source_trace,
        resume_step_index=3,
        entrypoint=entrypoint,
        overrides=ReplayOverrides(system_prompt="OVERRIDE: be a different agent"),
        tool_modes=tool_modes,
        divergence_policy="stub_with_warning",
    )

    # Steps 0-2: byte-identical to the source trace, untouched by the override.
    for step_index in range(3):
        assert (
            replayed_trace.steps[step_index].model_dump_json() == source_trace.steps[step_index].model_dump_json()
        ), f"step {step_index} should be copied verbatim from the source trace"

    # Step 3: overridden system prompt is visible in the injected history, but step 3's own
    # tool call still matches the recording (the override hasn't changed behavior yet).
    step3 = replayed_trace.steps[3]
    assert step3.model_call.messages[0] == {"role": "system", "content": "OVERRIDE: be a different agent"}
    assert step3.tool_calls[0].args_raw == {"query": "q3"}

    # Step 4: the overridden agent asks for something the recording never saw -> divergence,
    # served via STUB per the configured policy, run continues rather than crashing.
    step4 = replayed_trace.steps[4]
    assert step4.tool_calls[0].args_raw == {"query": "q4-override"}
    assert step4.tool_calls[0].result_as_inserted == "stubbed result"

    divergences = replayed_trace.metadata["trace_replay"]["divergences"]
    assert len(divergences) == 1
    assert divergences[0]["step_index"] == 4
    assert divergences[0]["tool_name"] == "search"
    assert divergences[0]["reason"] == "args_mismatch"
    assert divergences[0]["policy_applied"] == "stub_with_warning"
    assert divergences[0]["resolved_mode"] == "STUB"

    # The run completed (didn't crash) and the outcome reflects the override took effect.
    final_answer = replayed_trace.steps[-1].model_call.response["choices"][0]["message"]["content"]
    assert final_answer == "final answer (override)"
    source_final_answer = source_trace.steps[-1].model_call.response["choices"][0]["message"]["content"]
    assert final_answer != source_final_answer
