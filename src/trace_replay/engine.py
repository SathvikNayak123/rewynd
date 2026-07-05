"""resume_from: the replay engine. See docs/DESIGN.md "Replay semantics spec".

Steps before the resume point are copied verbatim from the source trace (byte-identical by
construction — no re-derivation, no re-execution) into the new trace. Live execution starts at
`resume_step_index`, seeded with the exact `messages` array the source trace sent to the model
for that step (the frozen world), with `overrides` applied. The whole run is captured by a fresh
ctx-capture `TraceRecorder`, so the output is a normal trace — replay-of-replay needs no special
case.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from ctx_capture.capture import TraceRecorder
from ctx_capture.schema import Trace

from trace_replay.costing import CostFn, apply_cost_to_live_steps, default_cost_fn, estimate_trace_cost
from trace_replay.divergence import DivergencePolicy
from trace_replay.harness import Entrypoint, ReplayContext, ReplayOverrides
from trace_replay.loader import check_schema_version
from trace_replay.modes import ToolModeConfig
from trace_replay.router import ToolRouter


def _apply_system_prompt(messages: list[dict[str, Any]], system_prompt: str) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["content"] = system_prompt
            return
    messages.insert(0, {"role": "system", "content": system_prompt})


def resume_from(
    source_trace: Trace,
    resume_step_index: int,
    entrypoint: Entrypoint,
    *,
    overrides: ReplayOverrides | None = None,
    tool_modes: ToolModeConfig | None = None,
    divergence_policy: DivergencePolicy = "fail_fast",
    live_tool_fns: dict[str, Callable[..., Any]] | None = None,
    tool_safety_registry: dict[str, bool] | None = None,
    cost_fn: CostFn = default_cost_fn,
    recorder_factory: Callable[..., TraceRecorder] = TraceRecorder,
) -> Trace:
    check_schema_version(source_trace)

    if not (0 <= resume_step_index < len(source_trace.steps)):
        raise ValueError(
            f"resume_step_index {resume_step_index} out of range for a {len(source_trace.steps)}-step trace"
        )
    resume_step = source_trace.steps[resume_step_index]
    if resume_step.model_call is None:
        raise ValueError(f"step {resume_step_index} has no recorded model_call to resume from")

    overrides = overrides or ReplayOverrides()
    tool_modes = tool_modes or ToolModeConfig()

    recorder = recorder_factory(agent_name=source_trace.agent_name, agent_version=source_trace.agent_version)
    for step in source_trace.steps[:resume_step_index]:
        recorder.trace.steps.append(step.model_copy(deep=True))

    injected_messages = copy.deepcopy(resume_step.model_call.messages)
    if overrides.system_prompt is not None:
        _apply_system_prompt(injected_messages, overrides.system_prompt)

    router = ToolRouter(
        source_trace=source_trace,
        recorder=recorder,
        tool_modes=tool_modes,
        divergence_policy=divergence_policy,
        live_tool_fns=live_tool_fns or {},
        tool_safety_registry=tool_safety_registry or {},
    )
    ctx = ReplayContext(
        messages=injected_messages,
        tool_router=router,
        overrides=overrides,
        resume_step_index=resume_step_index,
        recorder=recorder,
    )

    entrypoint(ctx)

    live_cost_usd = apply_cost_to_live_steps(recorder.trace, resume_step_index, cost_fn)
    source_cost_usd = estimate_trace_cost(source_trace, cost_fn)

    recorder.trace.metadata["trace_replay"] = {
        "source_trace_id": source_trace.trace_id,
        "source_step_index": resume_step_index,
        "tool_modes": {
            "default": tool_modes.default,
            "per_tool": dict(tool_modes.per_tool),
            "per_step": {k: dict(v) for k, v in tool_modes.per_step.items()},
        },
        "divergence_policy": divergence_policy,
        "divergences": [d.to_dict() for d in router.divergences],
        "cost_usd": live_cost_usd,
        "source_trace_cost_usd": source_cost_usd,
    }
    return recorder.trace
