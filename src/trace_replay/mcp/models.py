"""MCP tool I/O DTOs. See docs/DESIGN.md "MCP surface spec". FastMCP derives each tool's
outputSchema and structuredContent from these return-type annotations, same convention as
ctx-capture's own ctx_capture.mcp.models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReplayOverridesInput(BaseModel):
    system_prompt: str | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    injected_context_edits: dict[str, Any] = Field(default_factory=dict)


class ToolModesInput(BaseModel):
    default: str = "MOCK"
    per_tool: dict[str, str] = Field(default_factory=dict)
    per_step: dict[int, dict[str, str]] = Field(default_factory=dict)
    stub_results: dict[str, Any] = Field(default_factory=dict)


class ReplayableTraceSummary(BaseModel):
    trace_id: str
    agent_name: str | None
    created_at: str
    step_count: int
    replayable: bool
    reason_not_replayable: str | None = None


class ListReplayableTracesResult(BaseModel):
    traces: list[ReplayableTraceSummary]
    next_cursor: str | None = None


class ReplayFromStepResult(BaseModel):
    replay_id: str
    # v1 executes synchronously in-process (see server.py) so status is always terminal by the
    # time this returns: "completed" | "diverged" | "failed". The field still supports
    # "queued"/"running" so a future async job runner can populate them without a shape change.
    status: str
    new_trace_id: str | None = None


class DivergenceView(BaseModel):
    step_index: int
    tool_name: str
    args_raw: dict[str, Any]
    reason: str
    policy_applied: str
    resolved_mode: str


class GetReplayStatusResult(BaseModel):
    replay_id: str
    status: str
    new_trace_id: str | None
    current_step_index: int | None = None
    divergences: list[DivergenceView]
    cost_usd: float
    error: str | None = None


class ActionEntry(BaseModel):
    step_index: int
    tool_name: str
    args_raw: dict[str, Any]


class ActionsDiff(BaseModel):
    matched: list[ActionEntry]
    only_in_a: list[ActionEntry]
    only_in_b: list[ActionEntry]


class TokenDelta(BaseModel):
    prompt: int
    completion: int


class OutcomeDiff(BaseModel):
    trace_a_outcome: str | None
    trace_b_outcome: str | None
    changed: bool


class DiffRunsResult(BaseModel):
    trace_a: str
    trace_b: str
    actions_diff: ActionsDiff | None = None
    token_delta: TokenDelta | None = None
    cost_delta_usd: float | None = None
    outcome_diff: OutcomeDiff | None = None
    truncated: bool = False


class SetToolModeResult(BaseModel):
    ok: bool
    applied_scope: str  # "global" | "step"
