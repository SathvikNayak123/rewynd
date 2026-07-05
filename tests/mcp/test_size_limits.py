"""diff_runs never silently exceeds its byte cap — it comes back labeled truncated: true instead.
Same convention as ctx-capture's tests/mcp/test_size_limits.py.
"""

from __future__ import annotations

from ctx_capture.mcp.pagination import canonical_size

from tests.mcp.helpers import mcp_client


async def test_diff_runs_caps_oversized_diff(small_cap_mcp_server, seeded_source_trace_id):
    async with mcp_client(small_cap_mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0}
        )
        new_trace_id = replay_result.structuredContent["new_trace_id"]

        result = await session.call_tool(
            "diff_runs", {"trace_a": seeded_source_trace_id, "trace_b": new_trace_id, "max_bytes": 100}
        )
        assert result.isError is False, result.content
        sc = result.structuredContent
        assert sc["truncated"] is True
        assert canonical_size(sc) <= 100 * 6  # bounded, not unbounded — generous slack for wrapper fields


async def test_well_under_cap_is_not_marked_truncated(mcp_server, seeded_source_trace_id):
    async with mcp_client(mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0}
        )
        new_trace_id = replay_result.structuredContent["new_trace_id"]

        result = await session.call_tool(
            "diff_runs", {"trace_a": seeded_source_trace_id, "trace_b": new_trace_id}
        )
        assert result.structuredContent["truncated"] is False
