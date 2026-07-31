"""Drives the DeepResearch case study end to end. See docs/CASE_STUDY.md.

Three modes, because picking a resume step is a judgement call that depends on what the source run
actually did — the previous version of this script hardcoded one run id and step index from a
single dogfood session, which silently stopped meaning anything once DeepResearch's topology
changed and step indices shifted:

    # 1. what runs are in the DeepResearch store?
    python examples/deepresearch_case_study/run_case_study.py --list-runs

    # 2. what steps does one run have, and which look worth resuming from?
    python examples/deepresearch_case_study/run_case_study.py --list-steps <run_id>

    # 3. import that run and replay from the step you picked
    python examples/deepresearch_case_study/run_case_study.py \
        --run-id <run_id> --step <n> --sub-question "When was X born?"

Requires `DEEPRESEARCH_LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`: the replay makes one real
subagent-stage LLM call (small real cost, same as the original run's per-node cost), and
DeepResearch's ReAct tool-calling path is OpenAI-compatible only. It also needs a local
DeepResearch checkout that already ran the source benchmark (docs/CASE_STUDY.md "Reproducing").

Paths default to where this was dogfooded but are overridable, so the script isn't machine-bound:

    DEEPRESEARCH_DB=/path/to/run_store.db \
    DEEPRESEARCH_CORPUS=/path/to/corpus.json \
    python examples/deepresearch_case_study/run_case_study.py --run-id ... --step ...
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository

from examples.deepresearch_case_study.import_run import import_run
from examples.deepresearch_case_study.replay_harness import build_entrypoint
from trace_replay.engine import resume_from
from trace_replay.harness import ReplayOverrides
from trace_replay.modes import ToolModeConfig

DEEPRESEARCH_DB = os.environ.get(
    "DEEPRESEARCH_DB", r"C:\Users\user\Desktop\DeepResearch\deepresearch_case_study.db"
)
CORPUS_PATH = os.environ.get(
    "DEEPRESEARCH_CORPUS",
    r"C:\Users\user\Desktop\DeepResearch\data\corpus\musique\2hop__635544_110949.json",
)
TRACE_DB_PATH = os.environ.get(
    "TRACE_DB", str(Path(__file__).parent / "case_study_traces.db")
)


def _require_db() -> None:
    if not os.path.exists(DEEPRESEARCH_DB):
        raise SystemExit(
            f"DeepResearch run store not found at {DEEPRESEARCH_DB!r}. Set DEEPRESEARCH_DB to a "
            "run store produced by the benchmark run in docs/CASE_STUDY.md 'Reproducing' — note "
            "DeepResearch writes wherever DATABASE_URL points, default ./deepresearch.db."
        )


def list_runs() -> None:
    """Every run in the store, newest first, with the tell for whether it's usable: a run whose
    trajectories never got flushed (crash mid-run — see docs/CASE_STUDY.md Finding 1) has rows in
    `runs` but zero in `trajectories`, and there is nothing to import from it."""
    _require_db()
    conn = sqlite3.connect(DEEPRESEARCH_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT r.run_id, r.status, r.created_at, r.benchmark_name,
               (SELECT COUNT(*) FROM trajectories t WHERE t.run_id = r.run_id) AS n_traj,
               (SELECT COALESCE(SUM(cost_usd), 0) FROM trajectories t WHERE t.run_id = r.run_id) AS cost,
               -- the run's question isn't a `runs` column; it's the plan stage's input payload
               (SELECT t.input FROM trajectories t WHERE t.run_id = r.run_id
                 ORDER BY t.started_at LIMIT 1) AS first_input
        FROM runs r ORDER BY r.created_at DESC
        """
    ).fetchall()
    if not rows:
        raise SystemExit(f"no runs in {DEEPRESEARCH_DB}")
    for row in rows:
        flag = "" if row["n_traj"] else "   <- no trajectories, not importable"
        try:
            question = str(json.loads(row["first_input"] or "{}").get("question", ""))[:60]
        except ValueError:
            question = ""
        print(
            f'{row["run_id"]}  {row["status"]:<10} steps={row["n_traj"]:<4} '
            f'${row["cost"]:.4f}  {(row["benchmark_name"] or "-"):<10} {question!r}{flag}'
        )


def list_steps(run_id: str) -> None:
    """Imports the run and prints one line per resulting ctx-capture step, so a resume index can
    be chosen against what the trace actually contains rather than a remembered number."""
    _require_db()
    trace = import_run(DEEPRESEARCH_DB, run_id)
    print(f"{len(trace.steps)} steps in run {run_id}\n")
    for index, step in enumerate(trace.steps):
        call = step.model_call
        try:
            asked = json.loads(call.messages[0]["content"])
        except (ValueError, KeyError, IndexError):
            asked = {}
        label = asked.get("question") or asked.get("sub_question") or ""
        cost = f"${call.cost_usd:.5f}" if call.cost_usd else "$0"
        print(f"[{index:>2}] {call.model:<20} {cost:>9}  {str(label)[:70]!r}")


def import_only(run_id: str) -> None:
    """Imports one DeepResearch run into the ctx-capture trace store and stops. Used to stage both
    sides of a comparison — e.g. a pre-rebuild run out of one run store and a post-rebuild run out
    of another — into a single store so `diff_runs` can see both."""
    _require_db()
    trace = import_run(DEEPRESEARCH_DB, run_id)
    SQLiteTraceRepository(TRACE_DB_PATH).save(trace)
    cost = sum(s.model_call.cost_usd or 0 for s in trace.steps if s.model_call)
    print(f"imported {run_id}\n  from    {DEEPRESEARCH_DB}\n  trace_id {trace.trace_id}")
    print(f"  {len(trace.steps)} steps, ${cost:.5f}")


def diff(trace_a: str, trace_b: str, outcome_stage: str | None = "synthesis") -> None:
    """Prints the same comparison the `diff_runs` MCP tool returns, for two traces already in the
    store. Deliberately reuses trace_replay.mcp.diffing rather than recomputing anything here, so
    the CLI output and the MCP tool can't drift apart."""
    from trace_replay.costing import default_cost_fn
    from trace_replay.mcp.diffing import (
        compute_actions_diff,
        compute_cost_delta,
        compute_outcome_diff,
        compute_token_delta,
    )

    repo = SQLiteTraceRepository(TRACE_DB_PATH)
    a, b = repo.get_trace(trace_a), repo.get_trace(trace_b)

    print(f"A  {a.trace_id}  {len(a.steps)} steps  ({a.agent_name})")
    print(f"B  {b.trace_id}  {len(b.steps)} steps  ({b.agent_name})\n")

    tokens = compute_token_delta(a, b)
    print(f"token delta   prompt {tokens.prompt:+d}   completion {tokens.completion:+d}")
    print(f"cost delta    ${compute_cost_delta(a, b, default_cost_fn):+.5f}\n")

    actions = compute_actions_diff(a, b)
    print(
        f"actions       {len(actions.matched)} matched, "
        f"{len(actions.only_in_a)} only in A, {len(actions.only_in_b)} only in B"
    )
    for entry in actions.only_in_a[:5]:
        print(f"  - A only  step {entry.step_index}  {entry.tool_name}  {str(entry.args_raw)[:70]}")
    for entry in actions.only_in_b[:5]:
        print(f"  + B only  step {entry.step_index}  {entry.tool_name}  {str(entry.args_raw)[:70]}")

    # Compare synthesis-to-synthesis, not last-step-to-last-step: the rebuild moved reflection to
    # after synthesis, so the two traces don't end with the same kind of step.
    outcome = compute_outcome_diff(a, b, outcome_stage)
    print(f"\noutcome changed: {outcome.changed}  (compared at stage={outcome_stage!r})")
    print(f"  A: {str(outcome.trace_a_outcome)[:300]}")
    print(f"  B: {str(outcome.trace_b_outcome)[:300]}")


def replay(run_id: str, step_index: int, sub_question: str) -> None:
    _require_db()
    print("Importing DeepResearch run", run_id, "...")
    source_trace = import_run(DEEPRESEARCH_DB, run_id)
    repo = SQLiteTraceRepository(TRACE_DB_PATH)
    repo.save(source_trace)
    print(f"Imported as ctx-capture trace_id={source_trace.trace_id}, {len(source_trace.steps)} steps")

    if not 0 <= step_index < len(source_trace.steps):
        raise SystemExit(
            f"--step {step_index} out of range: this run has {len(source_trace.steps)} steps. "
            "Run --list-steps to see them."
        )

    original_step = source_trace.steps[step_index]
    print(f"\nOriginal step {step_index} ({original_step.model_call.messages}):")
    print(" ", original_step.model_call.response["choices"][0]["message"]["content"][:300])

    print(f"\nReplaying from step {step_index} with sub_question={sub_question!r} ...")
    replayed_trace = resume_from(
        source_trace,
        step_index,
        build_entrypoint(CORPUS_PATH),
        overrides=ReplayOverrides(injected_context_edits={"sub_question": sub_question}),
        tool_modes=ToolModeConfig(default="LIVE"),
        tool_safety_registry={"search": False, "fetch": False},
    )
    repo.save(replayed_trace)
    print(f"\nReplayed trace_id={replayed_trace.trace_id}")
    print("metadata.trace_replay:", replayed_trace.metadata["trace_replay"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-runs", action="store_true", help="list runs in the DeepResearch store")
    parser.add_argument("--list-steps", metavar="RUN_ID", help="import a run and list its steps")
    parser.add_argument("--import-run", metavar="RUN_ID", help="import a run into the trace store and stop")
    parser.add_argument("--diff", nargs=2, metavar=("TRACE_A", "TRACE_B"), help="diff two stored traces")
    parser.add_argument("--run-id", help="DeepResearch run to import and replay")
    parser.add_argument("--step", type=int, help="step index to resume from")
    parser.add_argument("--sub-question", help="the reworded sub-question to replay with")
    parser.add_argument(
        "--from-db",
        help="DeepResearch run store to read this invocation (overrides DEEPRESEARCH_DB) — needed "
        "when the two sides of a comparison live in different run stores",
    )
    args = parser.parse_args()

    if args.from_db:
        global DEEPRESEARCH_DB
        DEEPRESEARCH_DB = args.from_db

    if args.list_runs:
        list_runs()
    elif args.list_steps:
        list_steps(args.list_steps)
    elif args.import_run:
        import_only(args.import_run)
    elif args.diff:
        diff(*args.diff)
    elif args.run_id and args.step is not None and args.sub_question:
        replay(args.run_id, args.step, args.sub_question)
    else:
        parser.error(
            "pass --list-runs, --list-steps RUN_ID, --import-run RUN_ID, --diff A B, "
            "or all of --run-id/--step/--sub-question"
        )


if __name__ == "__main__":
    main()
