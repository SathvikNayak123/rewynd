"""One-off importer: DeepResearch's own run store (runs/trajectories/tool_calls, its
batch-at-end recorder — see deepresearch/store/recorder.py) -> a ctx-capture Trace.

This is deliberately NOT a general trace-replay feature. DeepResearch doesn't instrument with
ctx-capture, so there's no native trace to load — this reconstructs a best-effort one from
whatever DeepResearch itself persisted. Two honest fidelity gaps, documented rather than papered
over (see docs/CASE_STUDY.md "What the import can't recover"):

1. `messages` is synthesized from DeepResearch's own structured `input`/`output` JSON (its
   trajectories table), not captured verbatim from the wire — DeepResearch's recorder never had
   the raw provider message array to begin with, so this import can't retroactively invent one.
2. Tool-call `result_as_returned`/`result_as_inserted` are DeepResearch's own `result_summary`
   (e.g. `{"n_results": 6}` for a search), not the full tool output — that's genuinely all
   DeepResearch's `record_tool_call` ever stored.

One ctx-capture `Step` per DeepResearch `trajectory` row (plan/worker/reflection/synthesis all
serialize to one LLM call each in DeepResearch's own model); tool_calls attach to the step whose
`span_id` matches.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from ctx_capture.schema import ModelCall, Step, TokenCounts, ToolCall, Trace

_STAGE_MODEL_KEY = {
    "plan": "planner_model",
    "worker": "worker_model",
    "reflection": "reflection_model",
    "synthesis": "synthesis_model",
}


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def import_run(deepresearch_db_path: str, run_id: str) -> Trace:
    conn = sqlite3.connect(deepresearch_db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    run_row = cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run_row is None:
        raise KeyError(f"no run {run_id} in {deepresearch_db_path}")
    config = json.loads(run_row["config"])

    traj_rows = cur.execute(
        "SELECT * FROM trajectories WHERE run_id = ? ORDER BY started_at", (run_id,)
    ).fetchall()
    tool_rows = cur.execute("SELECT * FROM tool_calls WHERE run_id = ?", (run_id,)).fetchall()
    tool_rows_by_span: dict[str, list[sqlite3.Row]] = {}
    for row in tool_rows:
        tool_rows_by_span.setdefault(row["span_id"], []).append(row)

    steps: list[Step] = []
    for index, traj in enumerate(traj_rows):
        started_at = _parse_dt(traj["started_at"])
        ended_at = _parse_dt(traj["ended_at"]) if traj["ended_at"] else None
        input_obj = json.loads(traj["input"]) if traj["input"] else {}
        output_obj = json.loads(traj["output"]) if traj["output"] else {}

        model_call = ModelCall(
            provider="anthropic",
            model=config.get(_STAGE_MODEL_KEY.get(traj["stage"], ""), "unknown"),
            params={},
            messages=[{"role": "user", "content": json.dumps(input_obj)}],
            response={"choices": [{"message": {"role": "assistant", "content": json.dumps(output_obj)}}]},
            token_counts=TokenCounts(
                prompt_tokens=traj["tokens_in"] or 0,
                completion_tokens=traj["tokens_out"] or 0,
                total_tokens=(traj["tokens_in"] or 0) + (traj["tokens_out"] or 0),
            ),
            cost_usd=traj["cost_usd"],
            latency_ms=traj["latency_ms"],
        )

        tool_calls = [
            ToolCall(
                tool_call_id=str(row["id"]),
                tool_name=row["tool_name"],
                args_raw=json.loads(row["args"]) if row["args"] else {},
                result_as_returned=json.loads(row["result_summary"]) if row["result_summary"] else None,
                result_as_inserted=json.loads(row["result_summary"]) if row["result_summary"] else None,
                started_at=started_at,
                ended_at=ended_at,
                error=None if row["success"] else "tool call failed (success=0)",
            )
            for row in tool_rows_by_span.get(traj["span_id"], [])
        ]

        steps.append(
            Step(
                step_index=index,
                started_at=started_at,
                ended_at=ended_at,
                model_call=model_call,
                tool_calls=tool_calls,
            )
        )

    return Trace(
        agent_name="deepresearch-worker",
        metadata={
            "trace_replay": {
                "imported_from": "deepresearch",
                "source_run_id": run_id,
                "source_db": deepresearch_db_path,
                "benchmark_name": run_row["benchmark_name"],
                "fidelity_note": (
                    "best-effort import from DeepResearch's own batch-recorded trajectories/"
                    "tool_calls, not a native ctx-capture capture — messages are synthesized "
                    "from structured input/output JSON, and tool results are DeepResearch's own "
                    "result_summary, not the full tool output. See docs/CASE_STUDY.md."
                ),
            }
        },
        steps=steps,
    )
