"""Replay-run bookkeeping. See docs/DESIGN.md decision #5 "Replay-run storage": a small SQLite
table separate from ctx-capture's own trace storage, which trace-replay depends on directly for
trace data. This table only tracks the *run* (status, config, resulting trace id, divergences,
cost) — never trace content itself.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS replays (
    replay_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source_trace_id TEXT NOT NULL,
    source_step_index INTEGER NOT NULL,
    new_trace_id TEXT,
    tool_modes_json TEXT NOT NULL,
    divergence_policy TEXT NOT NULL,
    divergences_json TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    created_at TEXT NOT NULL
);
"""


@dataclass
class ReplayRecord:
    replay_id: str
    status: str
    source_trace_id: str
    source_step_index: int
    new_trace_id: str | None
    tool_modes: dict[str, Any]
    divergence_policy: str
    divergences: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None
    created_at: str = ""


class SQLiteReplayStore:
    def __init__(self, path: str = "trace_replay_runs.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA_DDL)
        self._conn.commit()

    def create(
        self,
        source_trace_id: str,
        source_step_index: int,
        tool_modes: dict[str, Any],
        divergence_policy: str,
    ) -> str:
        replay_id = str(uuid4())
        with self._conn:
            self._conn.execute(
                """INSERT INTO replays
                   (replay_id, status, source_trace_id, source_step_index, tool_modes_json,
                    divergence_policy, created_at)
                   VALUES (?, 'running', ?, ?, ?, ?, ?)""",
                (
                    replay_id,
                    source_trace_id,
                    source_step_index,
                    json.dumps(tool_modes),
                    divergence_policy,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return replay_id

    def complete(
        self,
        replay_id: str,
        *,
        status: str,
        new_trace_id: str | None,
        divergences: list[dict[str, Any]],
        cost_usd: float,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """UPDATE replays SET status = ?, new_trace_id = ?, divergences_json = ?, cost_usd = ?
                   WHERE replay_id = ?""",
                (status, new_trace_id, json.dumps(divergences), cost_usd, replay_id),
            )

    def fail(self, replay_id: str, error: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE replays SET status = 'failed', error = ? WHERE replay_id = ?",
                (error, replay_id),
            )

    def update_tool_modes(self, replay_id: str, tool_modes: dict[str, Any]) -> None:
        if self._get_row(replay_id) is None:
            raise KeyError(f"no replay {replay_id}")
        with self._conn:
            self._conn.execute(
                "UPDATE replays SET tool_modes_json = ? WHERE replay_id = ?",
                (json.dumps(tool_modes), replay_id),
            )

    def _get_row(self, replay_id: str) -> tuple | None:
        return self._conn.execute(
            """SELECT replay_id, status, source_trace_id, source_step_index, new_trace_id,
                      tool_modes_json, divergence_policy, divergences_json, cost_usd, error, created_at
               FROM replays WHERE replay_id = ?""",
            (replay_id,),
        ).fetchone()

    def get(self, replay_id: str) -> ReplayRecord:
        row = self._get_row(replay_id)
        if row is None:
            raise KeyError(f"no replay {replay_id}")
        return ReplayRecord(
            replay_id=row[0],
            status=row[1],
            source_trace_id=row[2],
            source_step_index=row[3],
            new_trace_id=row[4],
            tool_modes=json.loads(row[5]),
            divergence_policy=row[6],
            divergences=json.loads(row[7]),
            cost_usd=row[8],
            error=row[9],
            created_at=row[10],
        )
