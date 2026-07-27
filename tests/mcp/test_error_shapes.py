"""Invalid input never crashes the server — it comes back as a protocol-level CallToolResult
with isError=True, never a raised exception the client has to catch specially. Same convention
as ctx-capture's own tests/mcp/test_error_shapes.py.
"""

from __future__ import annotations

import pytest

from tests.mcp.helpers import mcp_client


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("diff_runs", {"trace_a": "does-not-exist", "trace_b": "also-does-not-exist"}),
        ("get_replay_status", {"replay_id": "does-not-exist"}),
        ("set_tool_mode", {"replay_id": "does-not-exist", "tool_name": "search", "mode": "MOCK"}),
    ],
)
async def test_unknown_id_is_a_protocol_error_not_a_crash(mcp_server, tool_name, args):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(tool_name, args)
        assert result.isError is True
        assert result.content, "error result must carry a human-readable message"


async def test_replay_from_step_with_no_registered_harness_is_error(mcp_server_no_entrypoint, trace_repo):
    from ctx_capture.capture import TraceRecorder

    from tests.fixtures.fake_client import FakeOpenAIClient
    from tests.fixtures.toy_agent import build_baseline_responses, record_toy_agent

    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    trace_repo.save(recorder.trace)

    async with mcp_client(mcp_server_no_entrypoint) as session:
        result = await session.call_tool(
            "replay_from_step", {"trace_id": recorder.trace.trace_id, "step_index": 0}
        )
        assert result.isError is True

        listed = await session.call_tool("list_replayable_traces", {})
        row = listed.structuredContent["traces"][0]
        assert row["replayable"] is False
        assert "no harness entrypoint" in row["reason_not_replayable"]


async def test_invalid_diff_type_is_error(mcp_server, seeded_source_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "diff_runs",
            {"trace_a": seeded_source_trace_id, "trace_b": seeded_source_trace_id, "diff_type": "bogus"},
        )
        assert result.isError is True


async def test_invalid_tool_mode_is_error(mcp_server, seeded_source_trace_id):
    async with mcp_client(mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0}
        )
        replay_id = replay_result.structuredContent["replay_id"]
        result = await session.call_tool(
            "set_tool_mode", {"replay_id": replay_id, "tool_name": "search", "mode": "BOGUS"}
        )
        assert result.isError is True


async def test_set_tool_mode_live_on_mutating_tool_requires_acknowledgment(mcp_server, seeded_source_trace_id):
    """set_tool_mode is config-only (docs/DESIGN.md), but a stored LIVE-for-a-mutating-tool
    config becomes a live decision the moment a future replay reads it back — so it must be
    gated the same way resume_from gates it at execution time (trace_replay.safety): rejected by
    default (the tool here is unlabeled, so it defaults to mutating), only proceeding with an
    explicit acknowledge_mutating=True."""
    async with mcp_client(mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0}
        )
        replay_id = replay_result.structuredContent["replay_id"]

        refused = await session.call_tool(
            "set_tool_mode", {"replay_id": replay_id, "tool_name": "search", "mode": "LIVE"}
        )
        assert refused.isError is True

        acknowledged = await session.call_tool(
            "set_tool_mode",
            {"replay_id": replay_id, "tool_name": "search", "mode": "LIVE", "acknowledge_mutating": True},
        )
        assert acknowledged.isError is False, acknowledged.content
        assert acknowledged.structuredContent["ok"] is True
