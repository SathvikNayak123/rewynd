# trace-replay

**Resume a recorded agent run from any step with a different prompt, model, or param — and see
what changes — without re-running the whole thing and hoping it fails the same way twice.**

An MCP server for deterministic agent-run replay, built directly on
[ctx-capture](https://github.com/ORG/ctx-capture)'s trace schema. It does not define its own trace
format; it consumes ctx-capture's.

## The pain

Debugging an agent today means re-running the whole thing and hoping it fails the same way again.
That's slow (full re-execution, real model latency), expensive (every debug iteration re-spends
the original run's tokens), and non-deterministic — retrieval, tool output, and the model itself
can all answer differently on the next run, so "did my fix work" gets confounded with "did the
world just answer differently this time."

What you actually want is narrower: **freeze everything before the step you're debugging, change
one thing at that step, and run forward from there.** ctx-capture already captures the byte-exact
record of what happened at every step. trace-replay is the other half — replay that record
forward.

## 60-second quickstart

**Requires [ctx-capture](https://github.com/ORG/ctx-capture)** — it isn't on PyPI yet, so install
it from source first. trace-replay consumes its trace schema and SDK directly; there's no
standalone mode.

```bash
git clone https://github.com/ORG/ctx-capture && pip install -e ./ctx-capture
git clone https://github.com/SathvikNayak123/trace-replay && pip install -e "./trace-replay[dev]"
```

Seed one real trace with the included demo agent (a tiny trip-planner that searches then
answers — see [examples/demo_agent.py](examples/demo_agent.py)):

```bash
cd trace-replay
python examples/seed_demo_trace.py
# Seeded source trace at examples/demo_ctx_capture.db
# trace_id = c740c832-bb5f-402b-8223-a5c9388bc69f
```

Run the MCP server, pointed at that trace DB and the demo agent's harness entrypoint:

```bash
python -m trace_replay.mcp \
  --db examples/demo_ctx_capture.db \
  --entrypoints examples.demo_agent:ENTRYPOINTS
```

Point an MCP client at it (`.mcp.json` in this repo already does, for Claude Code/Desktop) and
call `replay_from_step`:

```json
{"trace_id": "c740c832-...", "step_index": 2, "overrides": {"system_prompt": "Prefer budget-friendly trails."}}
```

That's it — steps 0-1 are held byte-identical, step 2 onward re-runs live with the new prompt.
`uvx trace-replay` works the same way once published (see [Packaging](#packaging)).

## The record/mock boundary

This is the whole product, so it gets a diagram instead of just prose.

```
source trace:   [step 0]──[step 1]──[step 2]──[step 3]──[step 4]──[step 5: final answer]

resume_from(trace, step_index=3, overrides={system_prompt: "..."})

new trace:      [step 0]──[step 1]──[step 2] │ [step 3]──[step 4]──[step 5: final answer]
                └──────────┬───────────────┘ │ └──────────────┬───────────────────────┘
                    copied verbatim,          │        live: harness runs for real.
                 byte-identical to source     │    Model calls are LIVE by definition —
                  trace — never re-run,       │    that's the experiment. Each tool call
                  never re-derived             │    resolves MOCK / LIVE / STUB against
                                                │    the source trace's own recording.
                                          resume point:
                                 injected history = source trace's own
                                 step[3].model_call.messages, verbatim —
                                 the frozen world becomes the harness's
                                 starting context, with `overrides` applied
                                 on top (system_prompt replaced, params
                                 shallow-merged so untouched keys — even a
                                 random seed — stay exactly as recorded)
```

Three things are true at once, and the whole design exists to keep them true simultaneously:

1. **The past is frozen.** Steps before the resume point aren't replayed, mocked, or
   reconstructed — they're the literal source-trace `Step` objects, copied into the new trace.
   Nothing about them can drift, because nothing about them runs.
2. **The model call at/after the resume point is always live.** That's not a mode you can turn
   off — mocking the *model* would defeat the point of resuming at all. What's mockable is
   everything the model's live output triggers: tool calls.
3. **Every tool call from the resume point on resolves to exactly one of three modes**, in this
   precedence — `(step, tool)` override → `tool` override → run-wide default (MOCK unless you
   change it):
   - **MOCK** (default) — serve the recorded `result_as_inserted` for a matching call. Free, fast,
     deterministic. This is what makes replaying feel like "the same run, but different" instead
     of "a whole new run."
   - **LIVE** — actually execute the tool.
   - **STUB** — serve a value you supply, for "what if this had returned X" without reproducing X
     for real.

   A tool call whose args don't match anything in the recording is a **divergence** — the agent
   asked a question the original run never saw. Default policy is **fail-fast**: the replay halts
   and tells you exactly where and why, rather than silently drifting into an unmocked, possibly
   costly, possibly non-deterministic tail. `live` and `stub_with_warning` are opt-in
   alternatives — see [`docs/DESIGN.md` § Divergence handling](docs/DESIGN.md).

   Mutating tools never go LIVE just because the run's default is LIVE — only an explicit,
   named-by-tool override counts, and that gate can't be bypassed by a divergence fallthrough
   either. An unlabeled tool (no `is_mutating` info anywhere) is treated as mutating, not safe.

## Integration contract

trace-replay doesn't run your agent by magic — process-level interception (monkeypatching the
model client / tool functions, VCR-style) was considered and rejected, for the same reason
ctx-capture rejected proxy-based capture: it's the exact place a fidelity tool starts quietly
lying about what's actually mocked. Instead, your agent implements one small entrypoint:

```python
def entrypoint(ctx: ReplayContext) -> None:
    # ctx.messages            — the injected history: source_trace.steps[resume_step].model_call.messages
    # ctx.tool_router.call()  — route every tool call through this instead of calling it directly
    # ctx.overrides           — system_prompt / model / params / injected_context_edits
    # ctx.resume_step_index   — where live execution starts
    # ctx.recorder            — the ctx-capture TraceRecorder building the new trace
    ...
```

That's the one adaptation asked of already-instrumented agent code: build your model client as
usual and wrap it with `ctx.recorder.wrap_client(...)`, but route tool calls through
`ctx.tool_router.call(tool_name, args)` instead of calling them directly — that's the seam where
MOCK/LIVE/STUB resolution happens, invisible to the rest of your agent logic. See
[examples/demo_agent.py](examples/demo_agent.py) for a complete, runnable one, and
[`docs/DESIGN.md` § Integration contract](docs/DESIGN.md) for the full argument.

v1 scope: **single-agent, linear, Python, already instrumented with ctx-capture.**

## Fidelity & safety guarantees

Three properties, each backed by a test that's meant to fail loudly if the property breaks:

| Guarantee | Proven by |
|---|---|
| Full-MOCK replay from step 0 reproduces the exact trajectory and final answer | [tests/test_fidelity.py](tests/test_fidelity.py) — CI-blocking forever, same status as ctx-capture's own capture-fidelity test |
| Steps before the resume point are byte-identical to the source trace; a real divergence is logged, not crashed on | [tests/test_override_replay.py](tests/test_override_replay.py) |
| A mutating tool never goes LIVE from the run-wide default; an explicit per-tool override still works | [tests/test_mutation_safety.py](tests/test_mutation_safety.py) |
| Replay-of-replay needs no special case — a replay's output is just a trace | [tests/test_replay_of_replay.py](tests/test_replay_of_replay.py) |
| The MCP surface itself: 5 tools, `outputSchema`/`structuredContent` contract, size caps, and a real record→replay→diff flow through a scripted MCP client | [tests/mcp/](tests/mcp/) |

## Replay cost accounting

ctx-capture's capture SDK never populates `cost_usd` (pricing isn't a capture-layer concern).
trace-replay fills that gap for its own accounting: a pluggable `cost_fn(token_counts) -> usd`
prices every live step in a replay, and estimates a comparable total for the source trace the same
way — so `get_replay_status` and `diff_runs` can both answer "did my fix cost more or less,"
not just "did the answer change." See [`trace_replay/costing.py`](src/trace_replay/costing.py).

## Non-goals

- **No multi-agent replay** — single-agent step fidelity and the record/mock boundary are already
  substantial scope; multi-agent needs a causality/correlation design that's premature before the
  single-agent core is proven.
- **No branching timeline UI** — MCP clients are the UI, same stance as ctx-capture. `diff_runs`
  gives a client everything it needs to build one if it wants.
- **No time-travel for external world state** — trace-replay freezes what the agent *observed*
  (recorded tool results), not the real systems those tools talked to. A LIVE tool call in a
  replay hits the real, current world.
- **No non-Python agents in v1** — the integration contract is a Python callable protocol,
  matching ctx-capture's own Python-first capture mechanism.

See [`docs/DESIGN.md` § Non-goals](docs/DESIGN.md) for the reasoning behind each.

## Honest limitations

- **Divergence explosion on early-step replays.** The earlier you resume, the more of the run is
  live, and the more chances the agent has to ask something the recording never saw — an
  overridden system prompt applied at step 0 can diverge almost immediately, making MOCK mode's
  "cheap, deterministic replay" benefit mostly theoretical for that resume point. Late-step
  replays (the common case — "what if step 7 had a different prompt") don't have this problem;
  very-early-step replays with a broad override are close to just running the agent fresh, and
  should be expected to diverge often. `fail_fast` surfaces this immediately rather than letting
  it burn budget silently — but it doesn't make the underlying tradeoff go away.
- **Single-agent only, and synchronous in v1.** `replay_from_step` runs to completion in-process
  before returning; there's no background job runner yet, so a very long live replay blocks the
  MCP call for its whole duration. Nothing in the tool contract needs to change for that to become
  async later (see [CHANGELOG.md](CHANGELOG.md)).
- **No Postgres-backed replay-run store yet** — SQLite only, matching where ctx-capture itself is
  at v0.1.0.
- **`set_tool_mode` is config-only in v1** — it updates a replay's stored tool-mode config, but
  can't reach into an already-finished synchronous run to change its behavior after the fact.

## Packaging

Semantic versioning; current release `0.1.0`. `python -m build` + `uvx trace-replay` /
`pip install trace-replay` once published — see
[.github/workflows/release.yml](.github/workflows/release.yml), which publishes to PyPI via
trusted publishing on any `v*` tag push. See [CHANGELOG.md](CHANGELOG.md) for what's in this
release and what's explicitly not yet.

## License

MIT — see [LICENSE](LICENSE).
