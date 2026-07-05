"""Divergence detection + policy. See docs/DESIGN.md "The record/mock boundary" → "Divergence
handling".

A divergence is a MOCK-mode tool call, at or after the resume point, for which no matching
recorded call exists in the source trace: the agent asked a different question than the original
run did. Detection matches canonical-JSON-serialized `args_raw` against the recorded call at the
same step_index — the same rigor ctx-capture's own fidelity test applies to `messages`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ctx_capture.schema import Trace, ToolCall

DivergencePolicy = Literal["fail_fast", "live", "stub_with_warning"]
DivergenceReason = Literal["step_beyond_source_trace", "no_recorded_call", "args_mismatch"]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class DivergenceError(Exception):
    def __init__(self, step_index: int, tool_name: str, reason: DivergenceReason) -> None:
        self.step_index = step_index
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"divergence at step {step_index}, tool {tool_name!r} ({reason}); "
            "replay halted (fail_fast policy)"
        )


@dataclass
class Divergence:
    step_index: int
    tool_name: str
    args_raw: dict[str, Any]
    reason: DivergenceReason
    policy_applied: DivergencePolicy
    resolved_mode: str  # "LIVE" | "STUB"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "args_raw": self.args_raw,
            "reason": self.reason,
            "policy_applied": self.policy_applied,
            "resolved_mode": self.resolved_mode,
        }


def find_recorded_tool_call(source_trace: Trace, step_index: int, tool_name: str, args_raw: dict[str, Any]) -> ToolCall | None:
    if step_index >= len(source_trace.steps):
        return None
    step = source_trace.steps[step_index]
    target = _canonical(args_raw)
    for tc in step.tool_calls:
        if tc.tool_name == tool_name and _canonical(tc.args_raw) == target:
            return tc
    return None


def classify_miss(source_trace: Trace, step_index: int, tool_name: str) -> DivergenceReason:
    if step_index >= len(source_trace.steps):
        return "step_beyond_source_trace"
    step = source_trace.steps[step_index]
    if not any(tc.tool_name == tool_name for tc in step.tool_calls):
        return "no_recorded_call"
    return "args_mismatch"
