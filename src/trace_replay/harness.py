"""The integration contract. See docs/DESIGN.md "Integration contract".

trace-replay does not run agent code by magic — it runs the user's agent through a fixed
entrypoint: `def entrypoint(ctx: ReplayContext) -> None`. The user's agent must already be
instrumented with the ctx-capture SDK and must route tool calls through `ctx.tool_router` instead
of calling tools directly. v1 scope: single-agent, linear, Python, instrumented with ctx-capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ctx_capture.capture import TraceRecorder

from trace_replay.router import ToolRouter


@dataclass
class ReplayOverrides:
    system_prompt: str | None = None
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    injected_context_edits: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayContext:
    messages: list[dict[str, Any]]
    tool_router: ToolRouter
    overrides: ReplayOverrides
    resume_step_index: int
    recorder: TraceRecorder


Entrypoint = Callable[[ReplayContext], None]
