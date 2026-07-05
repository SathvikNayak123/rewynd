# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/). Depends on ctx-capture's `schema_version` `1.0.x` —
see [CLAUDE.md](CLAUDE.md).

## [0.1.0] - 2026-07-05

Initial release.

### Added

- Replay engine (`trace_replay.engine.resume_from`): resumes a ctx-capture trace from any step
  with overridden `system_prompt`/`model`/`params`/`injected_context_edits`. Steps before the
  resume point are copied verbatim (byte-identical, not re-derived); the exact `messages` array
  ctx-capture recorded for the resume step becomes the injected history for the live harness —
  the frozen-past guarantee falls straight out of ctx-capture's own fidelity promise.
- Replay modes per tool call (`trace_replay.modes`): MOCK (default) / LIVE / STUB, resolved with
  precedence `(step, tool) > tool > run default`.
- Divergence detection and policy (`trace_replay.divergence`): canonical-JSON arg matching
  against the source trace; `fail_fast` (default) / `live` / `stub_with_warning`, with every
  divergence logged into the new trace's `metadata.trace_replay.divergences`.
- Mutation-safety gate (`trace_replay.safety`): a tool never goes LIVE from the run-wide default
  if it's flagged (or unknown-defaults-to) mutating — only an explicit per-tool/per-step override
  can do that, and it's never reachable via divergence fallthrough.
- Integration contract (`trace_replay.harness`): a fixed `entrypoint(ctx: ReplayContext)` callable
  the user's already-ctx-capture-instrumented agent implements, routing tool calls through
  `ctx.tool_router` — chosen over process-level VCR-style interception for the same
  honesty-over-magic reasoning ctx-capture used for SDK-first capture over proxy capture.
- Replay cost accounting (`trace_replay.costing`): a pluggable `cost_fn(token_counts) -> usd`
  fills `cost_usd` for live steps and estimates a comparable total for the source trace.
- Every replay is captured as a **new** trace via the ctx-capture SDK — replay-of-replay needs no
  special case, proven by `tests/test_replay_of_replay.py`.
- MCP server (`trace_replay.mcp.create_server`) exposing 5 tools — `list_replayable_traces`,
  `replay_from_step`, `get_replay_status`, `diff_runs`, `set_tool_mode` — over stdio and
  streamable-HTTP transports, with bearer-token auth on HTTP (ctx-capture's auth middleware
  reused directly, not re-implemented).
- `diff_runs`: trajectory diff between two traces of possibly different length, using
  `difflib.SequenceMatcher` over flattened tool-call sequences — aligned actions, token/cost
  deltas, and outcome comparison.
- Byte-size pagination and the 50KB response cap reused directly from ctx-capture
  (`ctx_capture.mcp.pagination`) — same conventions, not a parallel implementation.
- The replay fidelity test (CI-blocking): replaying step 0 of a recorded run in full-MOCK mode
  reproduces the exact trajectory and final answer — trace-replay's equivalent of ctx-capture's
  own capture-fidelity acceptance test.
- Override + divergence test: replaying from a later step with a modified system prompt keeps the
  untouched prefix byte-identical and logs (rather than crashes on) the resulting divergence.
- Mutation-safety test: a mutating tool set LIVE by the run-wide default still refuses; an
  explicit per-tool override still works.
- `trace-replay-mcp` console-script entry point (`python -m trace_replay.mcp` / `uvx trace-replay`).

### Known limitations (tracked for a future release)

- Replays execute synchronously in-process in v1 — `replay_from_step` runs to completion before
  returning; `get_replay_status` reads a finished (or failed) row rather than polling a live job.
  Nothing in the tool contract needs to change for a future async job runner.
- `set_tool_mode` only updates a replay's stored config in v1; since execution is synchronous, it
  can't reach back into an already-finished run (there's no in-flight window to steer).
- Single-agent, linear, Python agents only — see [docs/DESIGN.md § Non-goals](docs/DESIGN.md).
- No Postgres-backed replay-run store yet (SQLite only, matching ctx-capture's own v0.1.0 state).
