"""diff_runs: trajectory diff between two traces. See docs/DESIGN.md "Replay semantics spec" —
this is not `diff_step_contexts`' index-aligned per-step diff (ctx-capture's tool, for two steps
*within* one trace); two runs can diverge in length, so actions are aligned with a sequence
matcher rather than compared index-for-index.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from ctx_capture.schema import Trace

from trace_replay.costing import CostFn, estimate_trace_cost
from trace_replay.mcp.models import ActionEntry, ActionsDiff, OutcomeDiff, TokenDelta


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def flatten_actions(trace: Trace) -> list[dict[str, Any]]:
    return [
        {"step_index": step.step_index, "tool_name": tc.tool_name, "args_raw": tc.args_raw}
        for step in trace.steps
        for tc in step.tool_calls
    ]


def compute_actions_diff(trace_a: Trace, trace_b: Trace) -> ActionsDiff:
    actions_a = flatten_actions(trace_a)
    actions_b = flatten_actions(trace_b)
    keys_a = [_canonical(a) for a in actions_a]
    keys_b = [_canonical(b) for b in actions_b]

    matched: list[ActionEntry] = []
    only_in_a: list[ActionEntry] = []
    only_in_b: list[ActionEntry] = []

    matcher = SequenceMatcher(None, keys_a, keys_b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched.extend(ActionEntry(**a) for a in actions_a[i1:i2])
        else:
            if tag in ("delete", "replace"):
                only_in_a.extend(ActionEntry(**a) for a in actions_a[i1:i2])
            if tag in ("insert", "replace"):
                only_in_b.extend(ActionEntry(**b) for b in actions_b[j1:j2])

    return ActionsDiff(matched=matched, only_in_a=only_in_a, only_in_b=only_in_b)


def compute_token_delta(trace_a: Trace, trace_b: Trace) -> TokenDelta:
    def totals(trace: Trace) -> tuple[int, int]:
        prompt = sum(s.model_call.token_counts.prompt_tokens for s in trace.steps if s.model_call)
        completion = sum(s.model_call.token_counts.completion_tokens for s in trace.steps if s.model_call)
        return prompt, completion

    prompt_a, completion_a = totals(trace_a)
    prompt_b, completion_b = totals(trace_b)
    return TokenDelta(prompt=prompt_b - prompt_a, completion=completion_b - completion_a)


def compute_cost_delta(trace_a: Trace, trace_b: Trace, cost_fn: CostFn) -> float:
    return estimate_trace_cost(trace_b, cost_fn) - estimate_trace_cost(trace_a, cost_fn)


def _text_of(step) -> str | None:
    choices = step.model_call.response.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content")
    return None


def _final_text_outcome(trace: Trace, outcome_stage: str | None = None) -> str | None:
    """The trace's outcome text. Default is the last step that made a model call, which is right
    whenever both traces end with the same kind of step.

    `outcome_stage` exists because that assumption breaks across a topology change: if one trace
    ends with a synthesis step and the other moved synthesis earlier and now ends with a
    post-synthesis reflection step, comparing last-step-to-last-step compares a report against a
    verdict and reports a spurious change. Passing a stage name selects the last step tagged with
    it (via `model_call.params["stage"]`, which is where a capture or importer records the
    agent's own stage label) on each side, so the comparison stays apples-to-apples. Traces
    without stage tags, or without that stage, fall back to last-step — so this never makes an
    existing diff worse, it only refuses to be fooled when the information is there."""
    if outcome_stage is not None:
        tagged = [
            step
            for step in trace.steps
            if step.model_call is not None and (step.model_call.params or {}).get("stage") == outcome_stage
        ]
        if tagged:
            return _text_of(tagged[-1])

    for step in reversed(trace.steps):
        if step.model_call is None:
            continue
        return _text_of(step)
    return None


def compute_outcome_diff(trace_a: Trace, trace_b: Trace, outcome_stage: str | None = None) -> OutcomeDiff:
    outcome_a = _final_text_outcome(trace_a, outcome_stage)
    outcome_b = _final_text_outcome(trace_b, outcome_stage)
    return OutcomeDiff(trace_a_outcome=outcome_a, trace_b_outcome=outcome_b, changed=outcome_a != outcome_b)
