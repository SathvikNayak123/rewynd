"""diff_runs' outcome comparison must not be fooled when two traces end with different kinds of
step. This is the concrete case that motivated `outcome_stage` (docs/CASE_STUDY.md): DeepResearch's
topology rebuild moved reflection to *after* synthesis, so a pre-rebuild trace ends with the report
and a post-rebuild trace ends with a reflection verdict. Comparing last-step-to-last-step reports
"the outcome changed" even when both runs produced the identical report.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from ctx_capture.schema import ModelCall, Step, Trace

from trace_replay.mcp.diffing import compute_outcome_diff

_REPORT = "Mauro Scocco was born on 11 September 1962."
_VERDICT = '{"has_gaps": false}'


def _step(index: int, stage: str, content: str, *, tag_stage: bool = True) -> Step:
    return Step(
        step_index=index,
        started_at=datetime.now(timezone.utc),
        model_call=ModelCall(
            provider="test",
            model="test-model",
            params={"stage": stage} if tag_stage else {},
            messages=[{"role": "user", "content": "q"}],
            response={"choices": [{"message": {"role": "assistant", "content": content}}]},
        ),
    )


def _trace(*steps: Step) -> Trace:
    return Trace(agent_name="test-agent", steps=list(steps))


@pytest.fixture
def old_topology() -> Trace:
    """...plan, worker, synthesis — the report is the last step."""
    return _trace(_step(0, "plan", "{}"), _step(1, "synthesis", _REPORT))


@pytest.fixture
def new_topology() -> Trace:
    """...plan, synthesis, reflection — the report is no longer the last step."""
    return _trace(_step(0, "plan", "{}"), _step(1, "synthesis", _REPORT), _step(2, "reflection", _VERDICT))


def test_last_step_default_reports_spurious_change(old_topology, new_topology):
    """The pre-existing behaviour, pinned deliberately: with no outcome_stage, the report is
    compared against the reflection verdict and the diff claims a change that didn't happen."""
    diff = compute_outcome_diff(old_topology, new_topology)
    assert diff.trace_a_outcome == _REPORT
    assert diff.trace_b_outcome == _VERDICT
    assert diff.changed is True


def test_outcome_stage_compares_like_for_like(old_topology, new_topology):
    diff = compute_outcome_diff(old_topology, new_topology, outcome_stage="synthesis")
    assert diff.trace_a_outcome == _REPORT
    assert diff.trace_b_outcome == _REPORT
    assert diff.changed is False


def test_outcome_stage_still_detects_a_real_change(old_topology):
    changed = _trace(
        _step(0, "plan", "{}"),
        _step(1, "synthesis", "Someone else entirely was born in 1970."),
        _step(2, "reflection", _VERDICT),
    )
    diff = compute_outcome_diff(old_topology, changed, outcome_stage="synthesis")
    assert diff.changed is True


def test_falls_back_to_last_step_when_stage_absent(old_topology, new_topology):
    """A stage name that doesn't appear must not blank the outcome out — it degrades to the
    default, so asking for a stage can never make a diff worse than not asking."""
    diff = compute_outcome_diff(old_topology, new_topology, outcome_stage="nonexistent")
    assert diff.trace_a_outcome == _REPORT
    assert diff.trace_b_outcome == _VERDICT


def test_untagged_traces_fall_back(old_topology):
    """Traces captured without stage tags (params={}) still diff, via last-step."""
    untagged = _trace(_step(0, "plan", "{}", tag_stage=False), _step(1, "synthesis", _REPORT, tag_stage=False))
    diff = compute_outcome_diff(untagged, old_topology, outcome_stage="synthesis")
    assert diff.trace_a_outcome == _REPORT
    assert diff.changed is False


def test_last_matching_stage_wins():
    """When a stage repeats (a retried synthesis), the final one is the outcome."""
    trace = _trace(
        _step(0, "synthesis", "first attempt"),
        _step(1, "synthesis", "final attempt"),
        _step(2, "reflection", _VERDICT),
    )
    diff = compute_outcome_diff(trace, trace, outcome_stage="synthesis")
    assert diff.trace_a_outcome == "final attempt"
