"""Replay mode resolution. See docs/DESIGN.md "The record/mock boundary" → "Modes".

Precedence, most specific wins: (step_index, tool_name) override > tool_name override >
replay-run default (MOCK unless changed). The resolution *level* that produced the mode is
tracked alongside the mode itself, because the mutation-safety gate cares which level supplied
a LIVE decision (see safety.py) — a tool can only go LIVE on a mutating call if that LIVE came
from an explicit per-tool/per-step override, never from the run-wide default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ToolMode = Literal["MOCK", "LIVE", "STUB"]
ResolutionLevel = Literal["per_step", "per_tool", "default"]


@dataclass
class ToolModeConfig:
    default: ToolMode = "MOCK"
    per_tool: dict[str, ToolMode] = field(default_factory=dict)
    per_step: dict[int, dict[str, ToolMode]] = field(default_factory=dict)
    stub_results: dict[str, object] = field(default_factory=dict)


@dataclass
class ResolvedMode:
    mode: ToolMode
    level: ResolutionLevel


def resolve_mode(config: ToolModeConfig, step_index: int, tool_name: str) -> ResolvedMode:
    per_step_modes = config.per_step.get(step_index)
    if per_step_modes and tool_name in per_step_modes:
        return ResolvedMode(mode=per_step_modes[tool_name], level="per_step")
    if tool_name in config.per_tool:
        return ResolvedMode(mode=config.per_tool[tool_name], level="per_tool")
    return ResolvedMode(mode=config.default, level="default")
