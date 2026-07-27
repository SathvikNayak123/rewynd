"""FastMCP server exposing the 5 designed tools. See docs/DESIGN.md "MCP surface spec". Same
structuredContent/pagination/size-cap conventions as ctx-capture (its pagination helpers are
reused directly, not re-implemented) — see docs/DESIGN.md decision #4.

v1 executes replays synchronously in-process: `replay_from_step` runs `resume_from` to
completion before returning, so `get_replay_status` is a plain read of the bookkeeping row
rather than a poll against a background job. This is a deliberate simplification for a v1 whose
harness entrypoints are already in-process Python callables (see docs/DESIGN.md "Integration
contract") — nothing in the tool contract has to change if a future version makes it async.
"""

from __future__ import annotations

from typing import Any, Callable

from ctx_capture.mcp.pagination import canonical_size
from ctx_capture.storage.repository import TraceRepository
from mcp.server.fastmcp import FastMCP

from trace_replay.costing import default_cost_fn
from trace_replay.divergence import DivergenceError
from trace_replay.engine import resume_from
from trace_replay.harness import Entrypoint, ReplayOverrides
from trace_replay.loader import schema_major_supported
from trace_replay.mcp.diffing import compute_actions_diff, compute_cost_delta, compute_outcome_diff, compute_token_delta
from trace_replay.mcp.models import (
    DiffRunsResult,
    DivergenceView,
    GetReplayStatusResult,
    ListReplayableTracesResult,
    ReplayableTraceSummary,
    ReplayFromStepResult,
    ReplayOverridesInput,
    SetToolModeResult,
    ToolModesInput,
)
from trace_replay.mcp.replay_store import SQLiteReplayStore
from trace_replay.modes import ToolModeConfig
from trace_replay.safety import MutationSafetyError, is_mutating

DEFAULT_MAX_RESPONSE_BYTES = 50_000
_VALID_MODES = {"MOCK", "LIVE", "STUB"}


def _tool_mode_config_from_input(tool_modes: ToolModesInput | None) -> ToolModeConfig:
    if tool_modes is None:
        return ToolModeConfig()
    return ToolModeConfig(
        default=tool_modes.default,
        per_tool=dict(tool_modes.per_tool),
        per_step=({int(k): dict(v) for k, v in tool_modes.per_step.items()}),
        stub_results=dict(tool_modes.stub_results),
    )


def _overrides_from_input(overrides: ReplayOverridesInput | None) -> ReplayOverrides:
    if overrides is None:
        return ReplayOverrides()
    return ReplayOverrides(
        system_prompt=overrides.system_prompt,
        model=overrides.model,
        params=dict(overrides.params),
        injected_context_edits=dict(overrides.injected_context_edits),
    )


def _first_recorded_call_for_tool(source_trace, tool_name: str):
    for step in source_trace.steps:
        for tc in step.tool_calls:
            if tc.tool_name == tool_name:
                return tc
    return None


def _shrink_diff_to_fit(result: DiffRunsResult, cap: int) -> DiffRunsResult:
    result = result.model_copy(deep=True)
    if result.actions_diff is None:
        return result
    while canonical_size(result.model_dump(mode="json")) > cap and (
        result.actions_diff.matched or result.actions_diff.only_in_a or result.actions_diff.only_in_b
    ):
        result.truncated = True
        ad = result.actions_diff
        if ad.matched and len(ad.matched) >= max(len(ad.only_in_a), len(ad.only_in_b)):
            ad.matched.pop()
        elif ad.only_in_a and len(ad.only_in_a) >= len(ad.only_in_b):
            ad.only_in_a.pop()
        elif ad.only_in_b:
            ad.only_in_b.pop()
        else:
            ad.matched.pop()
    return result


def create_server(
    trace_repo: TraceRepository,
    replay_store: SQLiteReplayStore,
    entrypoints: dict[str, Entrypoint],
    live_tool_fns: dict[str, Callable[..., Any]] | None = None,
    tool_safety_registry: dict[str, bool] | None = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    name: str = "trace-replay",
) -> FastMCP:
    live_tool_fns = live_tool_fns or {}
    tool_safety_registry = tool_safety_registry or {}

    mcp = FastMCP(
        name=name,
        instructions="Resume a recorded agent trace from any step with modified prompt/model/params; "
        "everything before that step is held constant.",
    )

    def _replayability(trace_id: str) -> tuple[bool, str | None]:
        trace = trace_repo.get_trace(trace_id)
        if not schema_major_supported(trace.schema_version):
            return False, f"unsupported schema_version {trace.schema_version}"
        if not trace.steps:
            return False, "trace has no steps"
        if trace.agent_name not in entrypoints:
            return False, f"no harness entrypoint registered for agent_name {trace.agent_name!r}"
        return True, None

    @mcp.tool()
    def list_replayable_traces(
        agent_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> ListReplayableTracesResult:
        """List traces available for replay, flagged with whether trace-replay can actually
        resume them (supported schema version, has steps, a harness is registered for the
        recording agent). Cursor-paginated."""
        limit = max(1, min(limit, 200))
        rows, next_cursor = trace_repo.list_traces(agent_name=agent_name, since=since, until=until, limit=limit, cursor=cursor)
        summaries = []
        for row in rows:
            replayable, reason = _replayability(row.trace_id)
            summaries.append(
                ReplayableTraceSummary(
                    trace_id=row.trace_id,
                    agent_name=row.agent_name,
                    created_at=row.created_at,
                    step_count=row.step_count,
                    replayable=replayable,
                    reason_not_replayable=reason,
                )
            )
        return ListReplayableTracesResult(traces=summaries, next_cursor=next_cursor)

    @mcp.tool()
    def replay_from_step(
        trace_id: str,
        step_index: int,
        overrides: ReplayOverridesInput | None = None,
        tool_modes: ToolModesInput | None = None,
        divergence_policy: str = "fail_fast",
    ) -> ReplayFromStepResult:
        """Resume a trace from step_index with the given overrides. Everything before
        step_index is held constant (copied verbatim); the harness runs live from there. Returns
        a replay_id — see get_replay_status for divergences and cost."""
        source_trace = trace_repo.get_trace(trace_id)
        entrypoint = entrypoints.get(source_trace.agent_name)
        if entrypoint is None:
            raise ValueError(f"no harness entrypoint registered for agent_name {source_trace.agent_name!r}")

        tool_mode_config = _tool_mode_config_from_input(tool_modes)
        tool_modes_dict = {
            "default": tool_mode_config.default,
            "per_tool": tool_mode_config.per_tool,
            "per_step": tool_mode_config.per_step,
            "stub_results": tool_mode_config.stub_results,
        }
        replay_id = replay_store.create(trace_id, step_index, tool_modes_dict, divergence_policy)

        try:
            replayed_trace = resume_from(
                source_trace,
                step_index,
                entrypoint,
                overrides=_overrides_from_input(overrides),
                tool_modes=tool_mode_config,
                divergence_policy=divergence_policy,  # type: ignore[arg-type]
                live_tool_fns=live_tool_fns,
                tool_safety_registry=tool_safety_registry,
                cost_fn=default_cost_fn,
            )
        except (DivergenceError, MutationSafetyError) as e:
            replay_store.fail(replay_id, str(e))
            return ReplayFromStepResult(replay_id=replay_id, status="failed", new_trace_id=None)

        trace_repo.save(replayed_trace)
        divergences = replayed_trace.metadata["trace_replay"]["divergences"]
        status = "diverged" if divergences else "completed"
        replay_store.complete(
            replay_id,
            status=status,
            new_trace_id=replayed_trace.trace_id,
            divergences=divergences,
            cost_usd=replayed_trace.metadata["trace_replay"]["cost_usd"],
        )
        return ReplayFromStepResult(replay_id=replay_id, status=status, new_trace_id=replayed_trace.trace_id)

    @mcp.tool()
    def get_replay_status(replay_id: str) -> GetReplayStatusResult:
        """Poll a replay's outcome: status, resulting trace id, any divergences, and cost."""
        record = replay_store.get(replay_id)
        return GetReplayStatusResult(
            replay_id=record.replay_id,
            status=record.status,
            new_trace_id=record.new_trace_id,
            divergences=[DivergenceView(**d) for d in record.divergences],
            cost_usd=record.cost_usd,
            error=record.error,
        )

    @mcp.tool()
    def diff_runs(
        trace_a: str,
        trace_b: str,
        diff_type: str = "all",
        max_bytes: int | None = None,
    ) -> DiffRunsResult:
        """Trajectory diff between two traces (e.g. a source run and a replay of it): aligned
        action sequence, token/cost deltas, and outcome comparison. diff_type narrows the
        response to "actions" | "tokens" | "cost" | "outcome" | "all" (default)."""
        cap = max_bytes or max_response_bytes
        a = trace_repo.get_trace(trace_a)
        b = trace_repo.get_trace(trace_b)
        if diff_type not in ("actions", "tokens", "cost", "outcome", "all"):
            raise ValueError(f"unknown diff_type: {diff_type!r}")

        result = DiffRunsResult(trace_a=trace_a, trace_b=trace_b)
        if diff_type in ("actions", "all"):
            result.actions_diff = compute_actions_diff(a, b)
        if diff_type in ("tokens", "all"):
            result.token_delta = compute_token_delta(a, b)
        if diff_type in ("cost", "all"):
            result.cost_delta_usd = compute_cost_delta(a, b, default_cost_fn)
        if diff_type in ("outcome", "all"):
            result.outcome_diff = compute_outcome_diff(a, b)

        if canonical_size(result.model_dump(mode="json")) > cap:
            result = _shrink_diff_to_fit(result, cap)
        return result

    @mcp.tool()
    def set_tool_mode(
        replay_id: str,
        tool_name: str,
        mode: str,
        stub_result: Any = None,
        step_index: int | None = None,
        acknowledge_mutating: bool = False,
    ) -> SetToolModeResult:
        """Record a tool-mode override against a replay's stored config. v1 replays run to
        completion synchronously (see module docstring), so this configures what a future replay
        of the same trace/step should use — it does not reach back into an already-finished run.
        Setting LIVE for a tool that resolves mutating (trace flag, local safety registry, or
        unknown-defaults-to-mutating — see trace_replay.safety) is rejected unless called with
        acknowledge_mutating=True: the same explicit-opt-in gate resume_from enforces at
        execution time, applied here too so the config can't be used to pre-stage a bypass."""
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown mode: {mode!r} (expected MOCK, LIVE, or STUB)")
        record = replay_store.get(replay_id)  # raises KeyError if unknown -> protocol error

        if mode == "LIVE" and not acknowledge_mutating:
            source_trace = trace_repo.get_trace(record.source_trace_id)
            recorded_call = _first_recorded_call_for_tool(source_trace, tool_name)
            if is_mutating(tool_name, recorded_tool_call=recorded_call, tool_safety_registry=tool_safety_registry):
                raise MutationSafetyError(
                    tool_name,
                    "set_tool_mode LIVE for a mutating tool requires acknowledge_mutating=True",
                )

        tool_modes = record.tool_modes
        if step_index is not None:
            tool_modes.setdefault("per_step", {}).setdefault(str(step_index), {})[tool_name] = mode
            applied_scope = "step"
        else:
            tool_modes.setdefault("per_tool", {})[tool_name] = mode
            applied_scope = "global"
        if stub_result is not None:
            tool_modes.setdefault("stub_results", {})[tool_name] = stub_result

        replay_store.update_tool_modes(replay_id, tool_modes)
        return SetToolModeResult(ok=True, applied_scope=applied_scope)

    return mcp
