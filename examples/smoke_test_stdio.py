"""Smoke-tests the real `python -m trace_replay.mcp` subprocess over stdio using the actual MCP
client SDK (not the in-process test harness) — proof the server genuinely starts and serves tools
before anyone points a real chat client at it. Run from the repo root:

    python examples/smoke_test_stdio.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "examples", "demo_ctx_capture.db")
REPLAY_DB_PATH = os.path.join(REPO_ROOT, "examples", "demo_trace_replay_runs.db")


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "trace_replay.mcp",
            "--db",
            DB_PATH,
            "--replay-db",
            REPLAY_DB_PATH,
            "--entrypoints",
            "examples.demo_agent:ENTRYPOINTS",
        ],
        cwd=REPO_ROOT,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("tools:", sorted(t.name for t in tools))

            listed = await session.call_tool("list_replayable_traces", {})
            print("list_replayable_traces isError:", listed.isError)
            print("structuredContent:", listed.structuredContent)


if __name__ == "__main__":
    asyncio.run(main())
