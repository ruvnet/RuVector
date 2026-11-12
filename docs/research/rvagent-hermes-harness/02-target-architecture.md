# Target Architecture: rvAgent as a Hermes-Class, Self-Evolving Rust Harness

Design principle (from the benchmark literature): **structure transfers,
prose doesn't**. Everything below is structural — tools, loop mechanics,
memory layers, seams — not prompt wording.

---

## 1. The Loop (rvagent-core)

Replace today's blocking 4-node state machine with an **event-streaming
loop** (the ADR-139 direction, matching Claude Code's decompiled design and
Hermes's single-class-many-modes pattern):

```rust
// One loop, many frontends (CLI, TUI, ACP, A2A runner, WASM, headless)
pub trait AgentLoop {
    fn run(&mut self, input: LoopInput) -> impl Stream<Item = AgentEvent>;
}

pub enum AgentEvent {
    TurnStart { .. }, ModelDelta { text: String },          // streaming
    ToolCallStart { id, name, args }, ToolCallEnd { id, result },
    PermissionRequest { .. },                                // HITL surfaces here
    CompactionStart { .. }, CompactionEnd { lineage: SessionLineage },
    SubagentSpawned { .. }, SubagentResult { .. },
    Checkpoint { id }, TurnEnd { usage: Usage }, Done { state: AgentState },
    Error { recoverable: bool, .. },
}
```

Non-negotiable loop behaviors (each maps to a measured Hermes/Harness-Bench
lesson):

1. **Tools go to the model.** `ChatModel::complete(&[Message], &[ToolDefinition])`
   and `tools` on the Anthropic/Gemini request bodies. (Fixes D1.)
2. **Tool errors are tool results.** Never `?`-abort the loop on a failed
   tool; feed the error text back as a `Message::tool` so the model can
   recover. ("Execution alignment" — the dominant Harness-Bench factor.)
3. **Real parallel tool execution** via the existing (unused)
   `parallel_execute` JoinSet+Semaphore path. (Fixes D4.)
4. **Real streaming**: SSE parsing in the backends, `impl Stream<Item =
   StreamChunk>`, incremental TUI render.
5. **Few, high-fidelity core tools.** Keep the 9 built-ins; Hermes wins
   benchmarks with `terminal` + `file` only. Everything else arrives via
   MCP (rvagent-mcp client) and ruflo's `wasm_agent_compose` allowlist —
   never as bespoke tool sprawl.
6. **One canonical type system.** `rvagent-core`'s `AgentState`/`Message`/
   `Tool` become the only definitions; middleware and subagents consume
   them. (Fixes D3 — the blocker for everything else.)
7. **Async middleware.** `Middleware::wrap_model_call` becomes async; the
   default pipeline is constructed and wired in every entrypoint. (Fixes D2.)

## 2. Cache-First Prompt Assembly (new: `prompt_builder` in rvagent-core)

Hermes's 91–97% cache hit rate comes from a discipline, not a feature:

```
[stable]   identity + tool guidance + skill summaries   ← changes only on config change
[context]  AGENTS.md / project context files            ← changes only on file change
[volatile] memory digest + todos + timestamp            ← the ONLY tier that moves
```

- The stable tier is emitted with `cache_control` breakpoints (the existing
  `PromptCachingMiddleware` + a new `cache_control` field on `ApiRequest`).
- Enumerate cache-breaking events (model switch, memory-file change,
  context-file change) and log them — cache hit rate becomes a first-class
  metric in `Metrics`.
- The system prompt never mutates mid-conversation; volatile data rides in
  the tier boundary, and compaction respects the cache boundary.

## 3. Compaction with Lineage (wire + upgrade `SummarizationMiddleware`)

- Sentinel triggers **before** the hard limit (Hermes pattern): summarize
  middle turns, keep head (stable prompt) and tail (recent turns) intact.
- Durable facts extracted to the memory layer under a **hard char budget**
  (forcing curation, per Hermes's 3,575-char discipline).
- **Lineage**: compacted sessions record parent/child chains in the session
  store so summaries trace back to raw turns (enables replay + Darwin's
  trajectory harvesting).
- Upgrade path: ADR-252 coherence-weighted compaction
  (`ruvector-agent-memory`) replaces naive char-window summarization — a
  RuVector-native capability Hermes doesn't have.
- Replace the chars/4 token estimate with a real tokenizer + per-model
  context-window table.

## 4. Four-Layer Memory (mirror Hermes, back with RuVector/ruflo substrate)

| Layer | Hermes | rvagent implementation |
|---|---|---|
| Prompt | MEMORY.md/USER.md, hard budget | `MemoryMiddleware` (AGENTS.md, exists) + write-back with budget enforcement |
| Episodic | SQLite+FTS5 over sessions | session store + **ruflo's `.swarm/memory.db`** via AgentDB bindings; honor ADR-323 provenance (`agent_output`, `tool_result`) and the WAL-sidecar guard |
| Procedural | skills (markdown, agentskills.io) | `SkillsMiddleware` + `skills_bridge` (exists; already speaks Claude Code + Codex formats) with progressive disclosure at runtime |
| User model | Honcho profile | optional; defer |

**Kill the fake HNSW.** The current hash-based pseudo-embedding middleware
is worse than nothing (locality-insensitive, unfounded perf claims). Replace
with real `ruvector-core`/`@ruvector` embeddings behind the existing feature
gate, or use ruflo's hybrid retrieval over MCP. FTS-first, ANN-second — the
Hermes lesson is that deterministic local search beats a vector DB for a
local-first harness.

## 5. Skills + The Learning Loop (the differentiator)

This is where the three systems interlock, and it directly closes ruflo's
self-identified top gap ("DISTILL never emits shareable SKILL.md"):

```
witnessed trajectory (rvagent witness chain, SHA3-256 tool-call entries)
        │ trigger: 5+ tool calls / error recovery / user correction
        ▼
skill synthesis (auxiliary model distills trajectory → SKILL.md)
        │ progressive disclosure: summary in [stable] tier, body on demand
        ▼
in-use self-patching (string-patch edits, skill_manage tool)
        │
        ▼ offline
metaharness Darwin/flywheel: GEPA-style evolution of skills + policy genome
  - fitness from rvagent headless runs (SWE-bench/Terminal-Bench/TBLite)
  - promotion via harness-loop gates (OBSERVE→QUALIFY→BENCHMARK→VERIFY→CANARY→ACCEPT)
  - Ed25519-signed champion manifests (out-of-loop signing)
        │
        ▼
SONA/ReasoningBank consolidation (EWC++, ADR-271 genome recipe)
  + optional weight-eft: SFT/DPO distillation of gold trajectories into cheap-tier LoRA
```

The **policy genome seam** is metaharness's `SWE_POLICY_SYSTEM` pattern:
rvagent accepts an evolved operating policy as an appended stable-tier
block (`--policy-file` / env), so Darwin mutates behavior without touching
loop code. Genome surfaces: policy text, temperature schedule, compaction
thresholds, tool config, skill set, `EwcConfig` (ADR-271).

## 6. Subagents (make ADR-097 real)

- `TaskTool` → `SubAgentOrchestrator` actually spawns `AgentLoop` instances
  via JoinSet with a concurrency semaphore; results stream back as
  `SubagentResult` events; CRDT merge + validators (already written) run on
  join.
- Per-subagent model override (ADR-139's `CLAUDE_CODE_SUBAGENT_MODEL`
  equivalent) — routing decision can come from ruflo's Thompson bandit via
  MCP, or the local 3-tier heuristic.
- Subagent state isolation via the existing `CowStateBackend` fork/merge.
- Cross-*process/machine* delegation is already solved: the A2A crate.
  Plug the real loop in as the A2A `TaskRunner` (replacing `InMemoryRunner`)
  and rvagent instances federate with budgets, policy, recursion guards,
  and signed identity for free — a capability Hermes does not have.

## 7. Hooks, Permissions, Checkpoints

- **Hooks:** adopt metaharness `crates/kernel`'s 10-event taxonomy +
  `Allow/Deny/Ask/Defer` decision-merge (it's `#![forbid(unsafe_code)]`,
  serde-typed, no-I/O — mirror it per ADR-256 "borrow concepts", or take it
  as an *optional* dep consistent with ADR-150 symmetry). External hook
  processes use the ruflo `hook-handler.cjs` subprocess contract so one
  hook ecosystem serves both.
- **Permissions:** wire `HumanInTheLoopMiddleware` to the event stream
  (`PermissionRequest` event ↔ approval reply), add permission modes and
  per-tool allow/deny rules (ADR-139's 6-mode model).
- **Checkpoints:** full-state (messages + todos + files + middleware state +
  in-flight tool calls) using the AGI container's unused
  `SegmentType::Checkpoint`; resume/fork from any checkpoint. This is what
  makes Darwin's population runs cheap (metaharness measured 39.3% cost
  saved on resume, synthetic).
- **Sandbox depth:** keep env-sanitized local shell as the fast path, add a
  Docker backend (read-only root, dropped caps — Hermes's secure default)
  and honor ADR-140's WASM double-sandbox for untrusted agents. The
  `SandboxBackend` trait already models this; it needs implementations.

## 8. Headless Mode (the benchmark/evolution contract)

`rvagent run -p "<prompt>" --output-format json|stream-json --max-budget-usd
--policy-file --checkpoint-dir` — a stateless one-shot invocation emitting
the event stream as JSONL + a final result envelope (cost, tokens, tool
calls, patch). This single interface is what lets:
- metaharness Darwin use rvagent as a solver backend (exactly how
  `handoff-solver.mjs` shells out to `claude -p` today),
- ruflo's worker daemon spawn rvagent instead of headless Claude,
- CI run frozen-eval gates.

## 9. Integration Seams (explicit contracts)

### rvagent ⇄ ruflo
| Seam | Mechanism | Status |
|---|---|---|
| Turn loop hosting | `@ruvector/rvagent-wasm` (`WasmAgent`, `JsModelProvider`) + a NAPI sibling for native speed | WASM exists; NAPI new |
| Tool surface | `wasm_agent_compose` safe-allowlist of ruflo's 305 MCP tools (destructive gate) | exists |
| Memory | shared `.swarm/memory.db` / `.rvf`; ADR-323 provenance mandatory | bindings exist (AgentDB/ruvector are Rust-origin) |
| Lifecycle | hook-handler.cjs subprocess contract; Capability Brain entry with maturity/authority/risk metadata | contract exists |
| Routing | consume `[TASK_MODEL_RECOMMENDATION]` signals; report outcomes to the bandit | signal exists |

### rvagent ⇄ metaharness
| Seam | Mechanism | Status |
|---|---|---|
| Host adapter | `host-rvagent` (10th host): `generateConfig(spec)` emits rvagent config + skills + policy | new (adapter interface is trivial) |
| Evolution | headless JSON contract as Darwin solver; policy-genome env/file seam; flywheel promotion gates | pattern exists (`SWE_POLICY_SYSTEM`, `claude -p` precedent) |
| Governance | Ed25519 witness formats aligned (kernel witness ↔ rvagent witness chain ↔ RVF witness per ADR-106 Phase 4) | partial |
| Kernel reuse | mirror hooks/claims/dispatch types (ADR-256: concepts, not required dependency) | new |

Both directions preserve the ADR-150 invariant: **every integration is
optional and degrades gracefully**. rvagent must run standalone; ruflo must
run without rvagent; metaharness must stay removable.

## 10. What "best SOTA harness" concretely means here

1. **Beat the Claw-SWE-Bench harness spread**: rvagent + a fixed open model
   scores within the Hermes/OpenClaw band (≥70% on the 350-instance set),
   validated with metaharness's conformant apparatus (Wilson CIs, gold
   Docker eval, packaged submissions — no self-graded claims).
2. **Beat Hermes where it's weak**: native-speed kernel (startup, token
   throughput, parallel tools), in-browser WASM deployment, federated
   multi-agent execution with cryptographic identity/budgets (A2A), and
   coherence-weighted compaction (ADR-252) instead of char windows.
3. **Match Hermes where it's strong**: the closed learning loop — measured
   as ≥30% token reduction on repeat-task suites after 20+ synthesized
   skills, with Darwin-evolved skill populations as the upgrade Hermes
   only gets offline.
