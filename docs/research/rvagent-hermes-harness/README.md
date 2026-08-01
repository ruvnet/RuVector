# rvAgent as a Hermes-Class Harness — Research & Architecture Proposal

**Date:** 2026-08-01
**Status:** Research complete, implementation proposed
**Related:** ADR-093..107 (rvAgent), ADR-139 (Claude Code intelligence), ADR-150 (ruflo metaharness surfaces), ADR-159 (A2A), ADR-211/252 (agent memory), ADR-256 (metaharness concepts), ADR-260/266/271 (Darwin evolution)

## The Question

> How can we implement rvagent more like the Hermes harness, integrated with
> ruvnet/metaharness and ruvnet/ruflo, to create the best SOTA harness in the world?

## The Answer in One Paragraph

The Hermes agent (NousResearch/hermes-agent, MIT, ~175K stars) proved two things:
(1) **the harness is worth up to 27 points of SWE-bench pass@1** — more than most
model upgrades — and (2) the winning differentiator is not one-shot benchmark
score but a **closed learning loop**: trajectories distilled into self-patching
skills, evolved offline by GEPA, yielding ~40% faster/cheaper repeat tasks.
RuVector already believes this thesis — ADR-260/266/271's "freeze the model,
evolve the harness" is the same idea Hermes ships. What no one ships yet is a
**native-speed, memory-safe, WASM-portable harness kernel with an evolutionary
optimizer and a swarm coordination plane attached**. That is exactly the seam
where rvagent (Rust execution kernel) + metaharness (Darwin evolution + eval
apparatus + governance) + ruflo (memory substrate, hooks bus, MCP surface,
model routing, swarm) combine into something none of the incumbent harnesses
— Hermes included — can match. The catch: rvagent's core loop is currently
scaffold-grade (it cannot even send tool schemas to the model), so the path
starts with foundation repair, not features.

## Ecosystem Fit (who does what)

```
┌────────────────────────────────────────────────────────────────────┐
│  metaharness  — the harness FACTORY & EVOLVER                      │
│  mints harnesses (9 hosts incl. Hermes), Darwin/flywheel evolves   │
│  policy genomes, Ed25519 witness governance, SWE/Terminal-Bench    │
│  apparatus. Generated harnesses ship NO agent loop today.          │
└───────────────▲────────────────────────────────────────────────────┘
                │ evolves genome / benchmarks / signs
┌───────────────┴────────────────────────────────────────────────────┐
│  rvagent  — the EXECUTION KERNEL (this proposal)                   │
│  Rust agent loop: tools, streaming, compaction+lineage, skills,    │
│  subagents, checkpoints, hooks, budget, witness. Ships as native   │
│  CLI + NAPI + WASM. Fills the loop-shaped hole both siblings       │
│  currently outsource to Claude Code.                               │
└───────────────▲────────────────────────────────────────────────────┘
                │ memory / hooks / routing / tool surface via MCP
┌───────────────┴────────────────────────────────────────────────────┐
│  ruflo  — the COORDINATION PLANE                                   │
│  .swarm/memory.db + hnsw.index + .rvf substrate (ADR-323           │
│  provenance), 305 MCP tools, hooks lifecycle bus, Thompson-bandit  │
│  model router, swarm/hive-mind. Already hosts rvagent via          │
│  @ruvector/rvagent-wasm (27 wasm_agent_* tools).                   │
└────────────────────────────────────────────────────────────────────┘
```

## Documents

| File | Contents |
|---|---|
| [01-findings.md](01-findings.md) | What we found: Hermes architecture & benchmark evidence; rvagent current-state audit (4 blocking defects); metaharness & ruflo capability maps |
| [02-target-architecture.md](02-target-architecture.md) | The Hermes-class rvagent design: loop, prompt tiers, memory layers, skills, subagents, and the exact integration seams into metaharness and ruflo |
| [03-roadmap.md](03-roadmap.md) | Phased implementation plan (P0 foundations → P4 evolution/SOTA), success gates, proposed ADRs |

## Headline Findings

1. **rvagent's protocol layers are production-grade; its loop is not.**
   `rvagent-a2a` (signed cards, budgets, circuit breakers, recursion guards),
   `rvagent-mcp`, and the security primitives are real and well-tested. But the
   agent loop never sends tool schemas to the model (the `Tools` node is
   unreachable in production), the 19-module middleware pipeline is never wired
   into the CLI, subagents are stubs, streaming does not exist, and three
   incompatible `AgentState`/`Message`/`Tool` type systems block assembly.

2. **Hermes's edge is structural, and every piece has a RuVector-native analog.**
   Cache-first tiered prompts → unwired `PromptCachingMiddleware`; layered
   memory → AGENTS.md + ruflo's memory substrate + ADR-211/252; skills as
   procedural memory → `SkillsMiddleware` + `skills_bridge`; trajectory→skill
   distillation → SONA/ReasoningBank + witness chains; GEPA offline evolution →
   metaharness Darwin/flywheel. The parts exist; nothing is connected.

3. **Both siblings have a loop-shaped hole rvagent should fill.** metaharness
   generates harness *configuration* for nine hosts but no runtime loop; ruflo
   explicitly delegates multi-turn execution to Claude Code (`CLAUDE.md:908`)
   and its only in-house turn loop is… rvagent's WASM build. ruflo's own
   roadmap names "skill synthesis vs Hermes-class agents" its top gap.

4. **The benchmark literature says invest here.** Harness choice swings
   SWE-bench pass@1 by 12.5–27.4 points (Claw-SWE-Bench); "execution
   alignment" and few high-fidelity tools beat sprawling toolsets
   (Harness-Bench); structure transfers across models, prompt wording doesn't
   (Agentic Harness Engineering). A Rust kernel + Darwin evolution attacks all
   three levers at once.
