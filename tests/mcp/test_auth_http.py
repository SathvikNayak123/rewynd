"""Bearer auth for streamable HTTP mode. trace-replay reuses ctx-capture's auth middleware
directly (see docs/DESIGN.md "MCP surface spec" — same transport/auth decision) rather than
re-implementing it, so this test is really confirming the wiring in __main__.py's build, not the
middleware logic itself (which ctx-capture already tests).
"""

from __future__ import annotations

from ctx_capture.mcp.auth import build_http_app
from starlette.testclient import TestClient

TOKEN = "s3cr3t-test-token"


def _app(trace_repo, replay_store):
    from trace_replay.mcp.server import create_server

    mcp = create_server(trace_repo, replay_store, entrypoints={})
    return build_http_app(mcp, bearer_token=TOKEN)


def test_missing_authorization_header_is_rejected(trace_repo, replay_store):
    with TestClient(_app(trace_repo, replay_store)) as client:
        response = client.post("/mcp", json={})
        assert response.status_code == 401


def test_correct_token_passes_the_auth_layer(trace_repo, replay_store):
    with TestClient(_app(trace_repo, replay_store)) as client:
        response = client.post("/mcp", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code != 401
