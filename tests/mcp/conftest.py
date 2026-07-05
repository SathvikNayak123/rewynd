from __future__ import annotations

import pytest
from ctx_capture.capture import TraceRecorder
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository

from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import (
    build_baseline_responses,
    build_override_responses,
    build_replay_entrypoint,
    record_toy_agent,
)
from trace_replay.mcp.replay_store import SQLiteReplayStore
from trace_replay.mcp.server import create_server


@pytest.fixture
def trace_repo(tmp_path):
    return SQLiteTraceRepository(str(tmp_path / "traces.db"))


@pytest.fixture
def replay_store(tmp_path):
    return SQLiteReplayStore(str(tmp_path / "replays.db"))


@pytest.fixture
def seeded_source_trace_id(trace_repo) -> str:
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_baseline_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    record_toy_agent(recorder, capturing_client)
    trace_repo.save(recorder.trace)
    return recorder.trace.trace_id


@pytest.fixture
def mcp_server(trace_repo, replay_store):
    """Harness scripted to reproduce the baseline recording exactly — for full-mock/fidelity-style
    replay flows through MCP."""
    entrypoints = {"toy-agent": build_replay_entrypoint(build_baseline_responses())}
    return create_server(trace_repo, replay_store, entrypoints, max_response_bytes=50_000)


@pytest.fixture
def override_mcp_server(trace_repo, replay_store):
    """Harness scripted to diverge from step 4 onward — for divergence-triggering replay flows."""
    entrypoints = {"toy-agent": build_replay_entrypoint(build_override_responses())}
    tool_safety_registry = {"delete_file": True}
    return create_server(
        trace_repo, replay_store, entrypoints, tool_safety_registry=tool_safety_registry, max_response_bytes=50_000
    )


@pytest.fixture
def small_cap_mcp_server(trace_repo, replay_store):
    entrypoints = {"toy-agent": build_replay_entrypoint(build_baseline_responses())}
    return create_server(trace_repo, replay_store, entrypoints, max_response_bytes=300)


@pytest.fixture
def mcp_server_no_entrypoint(trace_repo, replay_store):
    """No harness registered for 'toy-agent' — for testing the not-replayable / no-harness paths."""
    return create_server(trace_repo, replay_store, entrypoints={}, max_response_bytes=50_000)
