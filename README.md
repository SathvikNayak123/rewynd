# trace-replay

Resume a recorded agent run from any step with a different prompt, model, or param — and see
what changes — without re-running the whole thing from scratch. An MCP server built on
[ctx-capture](https://github.com/SathvikNayak123/ctx-capture)'s trace schema; it doesn't define its own trace
format, it consumes ctx-capture's.

## What it is

Debugging an agent normally means re-running it end to end and hoping it fails the same way
again — slow, expensive, and non-deterministic. trace-replay freezes everything before the step
you're debugging and lets you change one thing (prompt, model, params) and run forward from
there.

**Features:**

- **Resume from any step**, with everything before it held byte-identical to the source trace —
  not re-run, not re-derived, just copied.
- **Three modes per tool call**: `MOCK` (default — serve the recorded result, free and
  deterministic), `LIVE` (actually execute it), `STUB` (serve a value you supply). Configurable
  globally, per tool, or per step.
- **Divergence handling** — if a replayed tool call doesn't match anything in the recording (the
  agent asked something new), the default is fail-fast: the replay halts and tells you exactly
  where, instead of drifting silently into an unmocked, possibly costly tail.
- **Mutation safety** — a tool never goes LIVE just because the run's default is LIVE; it needs an
  explicit, named override. Unlabeled tools are treated as mutating, not safe.
- **Replay cost accounting** — every replay reports its own cost against a comparable estimate for
  the original run.
- **Replay-of-replay** — a replay's output is just another trace, so resuming from a replay works
  with no special case.
- **5 MCP tools**: `list_replayable_traces`, `replay_from_step`, `get_replay_status`, `diff_runs`,
  `set_tool_mode` — stdio and streamable-HTTP transports, bearer auth on HTTP.

Real bug found with this tool, isolated in one $0.01 replay instead of a $0.08 full re-run.

## Setup

### Prerequisites

- **Python ≥ 3.10.**
- **[ctx-capture](https://github.com/SathvikNayak123/ctx-capture)** — trace-replay consumes its
  trace schema and SDK. Not on PyPI yet, so both install from source (step 1).
- **An agent instrumented with ctx-capture, and at least one recorded trace** — trace-replay
  resumes recorded runs; with an empty trace DB there is nothing to replay. (No agent yet? The
  included demo agent seeds a real trace in step 3.)
- **One small adapter in your agent code** — the "integration contract" (step 2). This is
  deliberate, not incidental: replay needs an explicit seam to decide MOCK/LIVE/STUB per tool
  call, which zero-touch monkeypatching can't express honestly (see docs/DESIGN.md, decision #3).
- **Any MCP client** — Claude Code, Claude Desktop, Cursor, etc.

### 1. Install both from source

```bash
git clone https://github.com/SathvikNayak123/ctx-capture && pip install -e ./ctx-capture
git clone https://github.com/SathvikNayak123/trace-replay && pip install -e "./trace-replay[dev]"
```

### 2. Implement the integration contract (once per agent)

Your agent needs to already be instrumented with ctx-capture, plus one adapter function that
trace-replay calls to resume it — routing tool calls through `ctx.tool_router` instead of
calling them directly is the only change to existing agent code:

```python
from trace_replay.harness import ReplayContext

def entrypoint(ctx: ReplayContext) -> None:
    messages = ctx.messages                      # frozen prefix, injected verbatim from the trace
    model = ctx.overrides.model or "my-model"    # resume-point overrides
    client = ctx.recorder.wrap_client(build_my_client())  # replay output is itself a trace
    # ...your existing agent loop, except tools go through the router:
    result = ctx.tool_router.call(tool_name, args)   # trace-replay resolves MOCK/LIVE/STUB here

ENTRYPOINTS = {"my-agent": entrypoint}
```

See [examples/demo_agent.py](examples/demo_agent.py) for a complete working one (the
trip-planner used below), and docs/DESIGN.md → "Integration contract" for the full spec.

### 3. Record (or seed) a source trace

Seed one real trace with the included demo agent (a tiny trip-planner that searches then
answers — see [examples/demo_agent.py](examples/demo_agent.py)):

```bash
cd trace-replay
python examples/seed_demo_trace.py
# trace_id = c740c832-bb5f-402b-8223-a5c9388bc69f
```

### 4. Run the MCP server

Point it at the trace DB and your entrypoints (here, the demo agent's):

```bash
python -m trace_replay.mcp \
  --db examples/demo_ctx_capture.db \
  --entrypoints examples.demo_agent:ENTRYPOINTS
```

`--db` is the ctx-capture trace database (read-only source of truth); replay bookkeeping goes to
a separate SQLite file (`--replay-db`, default `trace_replay_runs.db`) so replay state can never
corrupt source recordings. HTTP transport: add `--transport http --port 8001 --bearer-token
<token>` (stdio is the default).

### 5. Connect an MCP client and replay

Point an MCP client at it (`.mcp.json` in this repo already does, for Claude Code/Desktop) and
call `replay_from_step`:

```json
{"trace_id": "c740c832-...", "step_index": 2, "overrides": {"system_prompt": "Prefer budget-friendly trails."}}
```

Steps 0-1 come back byte-identical to the original; step 2 onward re-runs live with the new
prompt — the original answer was "Recommend Mount Elbert for an experienced hiker; permit not
required," the replayed one is "Recommend Bear Lake Trailhead (free, no permit) for a
budget-conscious hiker." Call `diff_runs` on the two trace ids to see that outcome change plus the
token/cost delta directly. `uvx trace-replay` will work the same way once published to PyPI.

## Tool reference

| Tool | Answers |
|---|---|
| `list_replayable_traces(agent_name?, since?, until?, limit?, cursor?)` | "What recorded runs can I resume?" — only traces whose agent has a registered entrypoint. |
| `replay_from_step(trace_id, step_index, overrides?, tool_modes?, divergence_policy?)` | "Resume this run from step N with X changed" — prefix held constant, live from there; returns a `replay_id`. |
| `get_replay_status(replay_id)` | "What happened to that replay?" — status, new trace id, divergence count, cost. |
| `diff_runs(trace_a, trace_b, diff_type?, outcome_stage?)` | "What changed between the original and the replay?" — aligned actions, token/cost deltas, outcome comparison. |
| `set_tool_mode(replay_id, tool_name, mode, stub_result?, step_index?, acknowledge_mutating?)` | "Make this tool MOCK / LIVE / STUB for a future replay of the same trace" — going LIVE on a mutating tool requires the explicit acknowledgement flag. |

Replay output is a normal ctx-capture trace, so it's browsable through ctx-capture's existing
`trace://{id}` resources — no separate replay-viewing surface.

## Limitations

- **Single-agent, linear, Python only** — no multi-agent replay, no branching UI, no non-Python
  agents in v1.
- **Divergence explosion on early-step replays** — the earlier you resume, the more of the run is
  live, and the more likely the agent asks something the recording never saw. Late-step replays
  (the common case) don't have this problem.
- **Synchronous in v1** — `replay_from_step` runs to completion before returning; no background
  job runner yet.
- **No time-travel for external world state** — a LIVE tool call hits the real, current world, not
  a reconstruction of what it looked like during the original run.
- **`set_tool_mode` is config-only** — it can't reach into an already-finished synchronous run to
  change its behavior after the fact.
- **No Postgres-backed replay-run store yet** — SQLite only.

Full list with reasoning: [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
