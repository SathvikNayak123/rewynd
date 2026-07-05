"""Drives the DeepResearch case study end to end. See docs/CASE_STUDY.md.

    python examples/deepresearch_case_study/run_case_study.py

Requires ANTHROPIC_API_KEY (the replay makes one real worker-stage LLM call — small real cost,
same as the original run's per-worker cost, ~$0.01-0.02).
"""

from __future__ import annotations

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

DEEPRESEARCH_DB = r"C:\Users\user\Desktop\DeepResearch\deepresearch_dogfood.db"
SOURCE_RUN_ID = "df557b49fd804e3795d0f16c3c211f34"  # the Ratata question, real live run
CORPUS_PATH = r"C:\Users\user\Desktop\DeepResearch\data\corpus\musique\2hop__635544_110949.json"
RESUME_STEP_INDEX = 5  # worker:194ca732 — "Johan Ekelund's date of birth", the step that found nothing
NEW_SUB_QUESTION = "When was Johan Ekelund born?"

TRACE_DB_PATH = str(Path(__file__).parent / "case_study_traces.db")


def main() -> None:
    print("Importing DeepResearch run", SOURCE_RUN_ID, "...")
    source_trace = import_run(DEEPRESEARCH_DB, SOURCE_RUN_ID)
    repo = SQLiteTraceRepository(TRACE_DB_PATH)
    repo.save(source_trace)
    print(f"Imported as ctx-capture trace_id={source_trace.trace_id}, {len(source_trace.steps)} steps")

    original_step = source_trace.steps[RESUME_STEP_INDEX]
    print(f"\nOriginal step {RESUME_STEP_INDEX} ({original_step.model_call.messages}):")
    print(" ", original_step.model_call.response["choices"][0]["message"]["content"][:300])

    print(f"\nReplaying from step {RESUME_STEP_INDEX} with sub_question={NEW_SUB_QUESTION!r} ...")
    replayed_trace = resume_from(
        source_trace,
        RESUME_STEP_INDEX,
        build_entrypoint(CORPUS_PATH),
        overrides=ReplayOverrides(injected_context_edits={"sub_question": NEW_SUB_QUESTION}),
        tool_modes=ToolModeConfig(default="LIVE"),
        tool_safety_registry={"search": False, "fetch": False},
    )
    repo.save(replayed_trace)
    print(f"\nReplayed trace_id={replayed_trace.trace_id}")
    print("metadata.trace_replay:", replayed_trace.metadata["trace_replay"])


if __name__ == "__main__":
    main()
