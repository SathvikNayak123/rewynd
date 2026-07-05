"""Run the trace-replay MCP server. See docs/DESIGN.md "MCP surface spec" / "Transport" (same
stdio + streamable-HTTP + bearer-auth decision as ctx-capture; its auth middleware is reused
directly rather than re-implemented).

  python -m trace_replay.mcp --entrypoints mypkg.replay:ENTRYPOINTS
  python -m trace_replay.mcp --entrypoints mypkg.replay:ENTRYPOINTS --transport http --port 8001 --bearer-token secret

--entrypoints points at a module-level dict[str, Entrypoint] (agent_name -> harness entrypoint) —
the integration contract's harness callables, see docs/DESIGN.md "Integration contract". These
are the user's own agent code and can't be inferred generically, so they're supplied by import
path rather than hardcoded.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from ctx_capture.mcp.auth import build_http_app
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository

from trace_replay.mcp.replay_store import SQLiteReplayStore
from trace_replay.mcp.server import DEFAULT_MAX_RESPONSE_BYTES, create_server


def _import_entrypoints(spec: str) -> dict:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"error: --entrypoints must be 'module:attr', got {spec!r}")
    module = importlib.import_module(module_name)
    entrypoints = getattr(module, attr)
    if not isinstance(entrypoints, dict):
        raise SystemExit(f"error: {spec} is not a dict[str, Entrypoint]")
    return entrypoints


def main() -> None:
    parser = argparse.ArgumentParser(prog="trace-replay-mcp")
    parser.add_argument("--db", default="ctx_capture.db", help="ctx-capture SQLite trace database path")
    parser.add_argument("--replay-db", default="trace_replay_runs.db", help="replay-run bookkeeping SQLite path")
    parser.add_argument(
        "--entrypoints",
        required=True,
        help="'module:attr' pointing at a dict[str, Entrypoint] (agent_name -> harness entrypoint)",
    )
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("TRACE_REPLAY_BEARER_TOKEN"),
        help="required for --transport http; also read from TRACE_REPLAY_BEARER_TOKEN",
    )
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    entrypoints = _import_entrypoints(args.entrypoints)

    trace_repo = SQLiteTraceRepository(args.db)
    replay_store = SQLiteReplayStore(args.replay_db)
    mcp = create_server(trace_repo, replay_store, entrypoints, max_response_bytes=args.max_response_bytes)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    if not args.bearer_token:
        print("error: --transport http requires --bearer-token or TRACE_REPLAY_BEARER_TOKEN", file=sys.stderr)
        raise SystemExit(2)

    import uvicorn

    app = build_http_app(mcp, args.bearer_token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
