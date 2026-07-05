"""ToolRouter: the seam of the integration contract (docs/DESIGN.md "Integration contract"). The
harness's entrypoint routes every tool call through `ctx.tool_router.call(tool_name, args)`
instead of calling tools directly — this is where MOCK/LIVE/STUB, divergence handling, and the
mutation-safety gate all live, invisible to the agent code itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from ctx_capture.capture import TraceRecorder
from ctx_capture.schema import ToolCall, Trace

from trace_replay.divergence import (
    Divergence,
    DivergenceError,
    DivergencePolicy,
    classify_miss,
    find_recorded_tool_call,
)
from trace_replay.modes import ToolModeConfig, resolve_mode
from trace_replay.safety import check_mutation_safety, is_mutating


def _first_recorded_call_for_tool(source_trace: Trace, tool_name: str) -> ToolCall | None:
    for step in source_trace.steps:
        for tc in step.tool_calls:
            if tc.tool_name == tool_name:
                return tc
    return None


@dataclass
class ToolRouter:
    source_trace: Trace
    recorder: TraceRecorder
    tool_modes: ToolModeConfig
    divergence_policy: DivergencePolicy = "fail_fast"
    live_tool_fns: dict[str, Callable[..., Any]] = field(default_factory=dict)
    tool_safety_registry: dict[str, bool] = field(default_factory=dict)
    divergences: list[Divergence] = field(default_factory=list)

    def _current_step_index(self) -> int:
        return self.recorder.trace.steps[-1].step_index

    def _record_tool_call(self, tool_name: str, args_raw: dict[str, Any], value: Any) -> None:
        now = datetime.now(timezone.utc)
        tool_call = ToolCall(
            tool_call_id=str(uuid4()),
            tool_name=tool_name,
            args_raw=args_raw,
            result_as_returned=value,
            result_as_inserted=value,
            started_at=now,
            ended_at=now,
        )
        self.recorder.trace.steps[-1].tool_calls.append(tool_call)

    def _call_live(self, tool_name: str, args_raw: dict[str, Any]) -> Any:
        fn = self.live_tool_fns.get(tool_name)
        if fn is None:
            raise KeyError(f"no live implementation registered for tool {tool_name!r}")
        return fn(**args_raw)

    def call(self, tool_name: str, args_raw: dict[str, Any]) -> Any:
        step_index = self._current_step_index()
        resolved = resolve_mode(self.tool_modes, step_index, tool_name)

        if resolved.mode == "MOCK":
            recorded = find_recorded_tool_call(self.source_trace, step_index, tool_name, args_raw)
            if recorded is not None:
                self._record_tool_call(tool_name, args_raw, recorded.result_as_inserted)
                return recorded.result_as_inserted
            reason = classify_miss(self.source_trace, step_index, tool_name)
            return self._handle_divergence(step_index, tool_name, args_raw, reason)

        if resolved.mode == "LIVE":
            check_mutation_safety(
                tool_name,
                "LIVE",
                resolved.level,
                recorded_tool_call=_first_recorded_call_for_tool(self.source_trace, tool_name),
                tool_safety_registry=self.tool_safety_registry,
            )
            result = self._call_live(tool_name, args_raw)
            self._record_tool_call(tool_name, args_raw, result)
            return result

        # STUB
        stub_value = self.tool_modes.stub_results.get(tool_name)
        self._record_tool_call(tool_name, args_raw, stub_value)
        return stub_value

    def _handle_divergence(self, step_index: int, tool_name: str, args_raw: dict[str, Any], reason: str) -> Any:
        mutating = is_mutating(
            tool_name,
            recorded_tool_call=_first_recorded_call_for_tool(self.source_trace, tool_name),
            tool_safety_registry=self.tool_safety_registry,
        )
        policy = self.divergence_policy

        if policy == "fail_fast" or (policy == "live" and mutating):
            raise DivergenceError(step_index, tool_name, reason)  # type: ignore[arg-type]

        if policy == "live":
            served = self._call_live(tool_name, args_raw)
            resolved_mode = "LIVE"
        else:  # stub_with_warning
            served = self.tool_modes.stub_results.get(tool_name)
            resolved_mode = "STUB"

        self.divergences.append(
            Divergence(
                step_index=step_index,
                tool_name=tool_name,
                args_raw=args_raw,
                reason=reason,  # type: ignore[arg-type]
                policy_applied=policy,
                resolved_mode=resolved_mode,
            )
        )
        self._record_tool_call(tool_name, args_raw, served)
        return served
