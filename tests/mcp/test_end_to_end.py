"""End-to-end: record a toy run -> replay it via an MCP tool call -> diff via an MCP tool call,
all through a real, scripted MCP client session. See DESIGN.md session task list, "MCP surface".

Two scenarios: a full-MOCK replay from step 0 (should reproduce the source run with zero
divergence and zero deltas) and an overridden replay from step 3 (should diverge and show a
changed outcome) — the same two behaviors test_fidelity.py / test_override_replay.py proved at
the engine level, now proven again end-to-end through the actual MCP protocol surface.
"""

from __future__ import annotations

from tests.mcp.helpers import mcp_client


async def test_full_mock_replay_reproduces_source_with_no_divergence(mcp_server, seeded_source_trace_id):
    async with mcp_client(mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step", {"trace_id": seeded_source_trace_id, "step_index": 0}
        )
        assert replay_result.isError is False, replay_result.content
        replay_sc = replay_result.structuredContent
        assert replay_sc["status"] == "completed"
        new_trace_id = replay_sc["new_trace_id"]

        status_result = await session.call_tool("get_replay_status", {"replay_id": replay_sc["replay_id"]})
        status_sc = status_result.structuredContent
        assert status_sc["status"] == "completed"
        assert status_sc["divergences"] == []
        assert status_sc["cost_usd"] > 0

        diff_result = await session.call_tool(
            "diff_runs", {"trace_a": seeded_source_trace_id, "trace_b": new_trace_id}
        )
        assert diff_result.isError is False, diff_result.content
        diff_sc = diff_result.structuredContent

        assert diff_sc["outcome_diff"]["changed"] is False
        assert diff_sc["outcome_diff"]["trace_a_outcome"] == diff_sc["outcome_diff"]["trace_b_outcome"] == "final answer"
        assert diff_sc["actions_diff"]["only_in_a"] == []
        assert diff_sc["actions_diff"]["only_in_b"] == []
        assert len(diff_sc["actions_diff"]["matched"]) == 5  # one tool call per step, steps 0-4
        assert diff_sc["token_delta"] == {"prompt": 0, "completion": 0}
        assert diff_sc["cost_delta_usd"] == 0


async def test_overridden_replay_diverges_and_diff_shows_changed_outcome(
    override_mcp_server, seeded_source_trace_id
):
    async with mcp_client(override_mcp_server) as session:
        replay_result = await session.call_tool(
            "replay_from_step",
            {
                "trace_id": seeded_source_trace_id,
                "step_index": 3,
                "overrides": {"system_prompt": "OVERRIDE: be a different agent"},
                "tool_modes": {"default": "MOCK", "stub_results": {"search": "stubbed result"}},
                "divergence_policy": "stub_with_warning",
            },
        )
        assert replay_result.isError is False, replay_result.content
        replay_sc = replay_result.structuredContent
        assert replay_sc["status"] == "diverged"
        new_trace_id = replay_sc["new_trace_id"]

        status_result = await session.call_tool("get_replay_status", {"replay_id": replay_sc["replay_id"]})
        status_sc = status_result.structuredContent
        assert status_sc["status"] == "diverged"
        assert len(status_sc["divergences"]) == 1
        assert status_sc["divergences"][0]["step_index"] == 4
        assert status_sc["divergences"][0]["reason"] == "args_mismatch"

        diff_result = await session.call_tool(
            "diff_runs", {"trace_a": seeded_source_trace_id, "trace_b": new_trace_id}
        )
        diff_sc = diff_result.structuredContent
        assert diff_sc["outcome_diff"]["changed"] is True
        assert diff_sc["outcome_diff"]["trace_a_outcome"] == "final answer"
        assert diff_sc["outcome_diff"]["trace_b_outcome"] == "final answer (override)"
        # step 4's action ("q4-override") only appears in trace_b; the recorded ("q4") only in trace_a.
        assert len(diff_sc["actions_diff"]["only_in_a"]) == 1
        assert len(diff_sc["actions_diff"]["only_in_b"]) == 1
        assert diff_sc["actions_diff"]["only_in_a"][0]["args_raw"] == {"query": "q4"}
        assert diff_sc["actions_diff"]["only_in_b"][0]["args_raw"] == {"query": "q4-override"}
