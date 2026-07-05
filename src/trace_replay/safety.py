"""Mutation-safety gate. See docs/DESIGN.md "The record/mock boundary" → "Side-effect safety".

Tools flagged mutating are never auto-LIVE. ctx-capture's schema (v1.0) has no `is_mutating`
field on `ToolCall` yet — a schema addendum is proposed (docs/DESIGN.md), coordinated and
additive. Until it lands, resolution falls back to a local tool-safety registry the caller
supplies. Resolution order, most authoritative first:

  1. `tool_call.is_mutating` on the matched recorded call, when present and non-None (forward
     compatible: `getattr` with a None default means this picks up the field automatically the
     moment ctx-capture ships it, no code change needed here).
  2. The local `tool_safety_registry` (tool_name -> bool) the replay run supplies.
  3. Unknown -> treated as mutating. An unlabeled tool never gets a free pass to LIVE.

A mutating tool may only go LIVE when the LIVE decision came from an *explicit* per-tool or
per-step override (ResolutionLevel != "default") — never from the run-wide default, and never
as a divergence fallthrough (divergence.py enforces the latter).
"""

from __future__ import annotations

from typing import Any

from trace_replay.modes import ResolutionLevel


class MutationSafetyError(Exception):
    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"refusing LIVE execution of mutating tool {tool_name!r}: {reason}")


def is_mutating(
    tool_name: str,
    *,
    recorded_tool_call: Any = None,
    tool_safety_registry: dict[str, bool] | None = None,
) -> bool:
    trace_flag = getattr(recorded_tool_call, "is_mutating", None) if recorded_tool_call is not None else None
    if trace_flag is not None:
        return bool(trace_flag)

    registry = tool_safety_registry or {}
    if tool_name in registry:
        return registry[tool_name]

    return True  # unknown defaults to mutating


def check_mutation_safety(
    tool_name: str,
    mode: str,
    level: ResolutionLevel,
    *,
    recorded_tool_call: Any = None,
    tool_safety_registry: dict[str, bool] | None = None,
) -> None:
    """Raises MutationSafetyError if this LIVE resolution is not allowed for a mutating tool."""
    if mode != "LIVE":
        return
    if not is_mutating(tool_name, recorded_tool_call=recorded_tool_call, tool_safety_registry=tool_safety_registry):
        return
    if level == "default":
        raise MutationSafetyError(
            tool_name,
            "LIVE resolved from the run-wide default, not an explicit per-tool/per-step override",
        )
