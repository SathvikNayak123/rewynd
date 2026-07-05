"""Per-tool contract: every tool declares outputSchema, and every successful call's
structuredContent validates against it. See docs/DESIGN.md "MCP surface spec".

Drives every tool through one realistic flow (list -> replay -> status -> diff -> set_tool_mode)
since get_replay_status/set_tool_mode need a real replay_id to exercise meaningfully — a static
per-tool args table (like ctx-capture's) doesn't fit here.
"""

from __future__ import annotations

import jsonschema
import pytest

from tests.mcp.helpers import mcp_client

EXPECTED_TOOLS = {
    "list_replayable_traces",
    "replay_from_step",
    "get_replay_status",
    "diff_runs",
    "set_tool_mode",
}


async def test_exactly_five_tools_registered(mcp_server):
    async with mcp_client(mcp_server) as session:
        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}
        assert names == EXPECTED_TOOLS, "tool set drifted from the designed 5 (see docs/DESIGN.md)"


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOLS))
async def test_tool_declares_output_schema(mcp_server, tool_name):
    async with mcp_client(mcp_server) as session:
        tools = (await session.list_tools()).tools
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.outputSchema is not None, f"{tool_name} has no outputSchema"


async def test_all_structured_content_conforms_to_schema(mcp_server, seeded_source_trace_id):
    async with mcp_client(mcp_server) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

        r = await session.call_tool("list_replayable_traces", {})
        assert r.isError is False, r.content
        jsonschema.validate(r.structuredContent, tools["list_replayable_traces"].outputSchema)
        assert r.structuredContent["traces"][0]["replayable"] is True

        r = await session.call_tool("replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0})
        assert r.isError is False, r.content
        jsonschema.validate(r.structuredContent, tools["replay_from_step"].outputSchema)
        replay_id = r.structuredContent["replay_id"]
        new_trace_id = r.structuredContent["new_trace_id"]
        assert r.structuredContent["status"] == "completed"

        r = await session.call_tool("get_replay_status", {"replay_id": replay_id})
        assert r.isError is False, r.content
        jsonschema.validate(r.structuredContent, tools["get_replay_status"].outputSchema)

        r = await session.call_tool("diff_runs", {"trace_a": seeded_source_trace_id, "trace_b": new_trace_id})
        assert r.isError is False, r.content
        jsonschema.validate(r.structuredContent, tools["diff_runs"].outputSchema)

        r = await session.call_tool("set_tool_mode", {"replay_id": replay_id, "tool_name": "search", "mode": "STUB"})
        assert r.isError is False, r.content
        jsonschema.validate(r.structuredContent, tools["set_tool_mode"].outputSchema)
        assert r.structuredContent == {"ok": True, "applied_scope": "global"}
