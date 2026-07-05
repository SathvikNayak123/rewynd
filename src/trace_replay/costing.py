"""Replay cost accounting. See docs/DESIGN.md "Evaluating the server itself" — a replay must
report its own cost, separate from and comparable to the source run's.

ctx-capture's capture SDK never populates `ModelCall.cost_usd` (it's left `None`; pricing is out
of scope for a capture layer). trace-replay fills that gap for its own accounting with a
pluggable `cost_fn(token_counts) -> usd`, applied only where `cost_usd` isn't already set, so a
provider-supplied cost is never overridden.
"""

from __future__ import annotations

from typing import Callable

from ctx_capture.schema import TokenCounts, Trace

CostFn = Callable[[TokenCounts], float]

# Placeholder per-1k-token rates. Swap via the `cost_fn` param for real provider pricing.
DEFAULT_PRICE_PER_1K_PROMPT_USD = 0.003
DEFAULT_PRICE_PER_1K_COMPLETION_USD = 0.015


def default_cost_fn(token_counts: TokenCounts) -> float:
    return (
        (token_counts.prompt_tokens / 1000) * DEFAULT_PRICE_PER_1K_PROMPT_USD
        + (token_counts.completion_tokens / 1000) * DEFAULT_PRICE_PER_1K_COMPLETION_USD
    )


def apply_cost_to_live_steps(trace: Trace, resume_step_index: int, cost_fn: CostFn) -> float:
    """Fills in `cost_usd` for steps at/after resume_step_index that don't have one set.
    Returns the summed cost across those steps."""
    total = 0.0
    for step in trace.steps:
        if step.step_index < resume_step_index or step.model_call is None:
            continue
        if step.model_call.cost_usd is None:
            step.model_call.cost_usd = cost_fn(step.model_call.token_counts)
        total += step.model_call.cost_usd
    return total


def estimate_trace_cost(trace: Trace, cost_fn: CostFn) -> float:
    """Read-only cost estimate for a whole trace: uses recorded cost_usd where present, else
    estimates from token_counts. Never mutates the trace."""
    total = 0.0
    for step in trace.steps:
        if step.model_call is None:
            continue
        total += step.model_call.cost_usd if step.model_call.cost_usd is not None else cost_fn(step.model_call.token_counts)
    return total
