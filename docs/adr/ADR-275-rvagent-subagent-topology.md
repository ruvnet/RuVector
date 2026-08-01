---
adr: 275
title: "rvAgent Subagent Topology: Single Writer with Auxiliary Intelligence"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-103, ADR-107, ADR-273, ADR-274, ADR-277]
tags: [rvagent, harness, subagents, multi-agent, concurrency, sota]
---

# ADR-275 — rvAgent Subagent Topology: Single Writer with Auxiliary Intelligence

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Evidence base: `docs/research/rvagent-hermes-harness/04-sota-landscape.md` §3.

**This ADR reverses a standing design bet.** `03-roadmap.md` Phase 1.6
specifies "real subagents (JoinSet spawn, semaphore, CoW fork/merge, CRDT
join)". CoW fork plus CRDT join *is* parallel-writer architecture — the one
multi-agent pattern with strong negative evidence for coding specifically.

## 1. Decision

**One writer. Auxiliary intelligence around it. Never parallel writes.**

The subagent boundary is modelled as **a tool that spawns an isolated context
and returns a String** — not as peer agents with a message bus, shared mutable
state, or a mergeable state type.

Two subagent roles are adopted; a third is deferred.

| Role | Status | Shares context? | Writes? |
|---|---|---|---|
| Fresh-context reviewer | Adopt | **No** — deliberately | No |
| Read-only context-gatherer | Adopt | No | No |
| Coordinator / manager | Defer | — | No |
| Parallel writers | **Rejected** | — | — |

## 2. Why

### 2.1 The positive multi-agent result does not transfer to coding

Anthropic's multi-agent research system reports **+90.2%** over single-agent —
and simultaneously that **token usage alone explains 80% of the performance
variance** (95% with tool-call count and model added). Much of the gain is
*buying more compute*, not coordination; the missing control arm is a single
agent at the same 15× budget.

Anthropic states directly that the architecture suits **breadth-first**
questions with independent paths and is **less effective for tightly
interdependent tasks such as coding.**

### 2.2 The skeptical position was revised, not refuted

Cognition's "Don't Build Multi-Agents" (Jun 2025) argued for single-threaded
linear agents because "actions carry implicit decisions" that conflict when
parallelized.

Their April 2026 revision — after a year of production data — did not reverse
this. It refined it:

- **One writer, augmented by auxiliary intelligence. Never parallel writes.**
- Code review loop works: Devin Review catches ~**2 bugs per PR, 58% severe**.
- **Reviewers perform better with NO shared context.** Shorter context → less
  context rot → deeper analysis. This inverts the usual "share everything"
  instinct and is the most actionable finding here.
- Manager delegation ships but "requires extensive context engineering;
  managers default to over-prescription without deep codebase knowledge."

### 2.3 At equal budget, single-agent wins on coding

The 2026 consensus across sources: at **equal token budget**, single-agent
matches or beats multi-agent on multi-hop reasoning. Multi-agent earns its
overhead only on breadth-first, parallel-decomposable, low-state-sharing tasks.
Coding is the canonical *bad* fit — it is the case where sub-results are
interdependent and merge conflicts are semantic, not textual.

A CRDT can merge two edits to the same file without conflict. It cannot make
the *result* coherent. That is precisely the failure Cognition describes.

## 3. Adopted patterns

### 3.1 Fresh-context reviewer

Spawns with **no inherited conversation** — only the diff and the task
statement. Returns findings as a string. Does not write.

The counterintuitive part is load-bearing: do **not** pass the parent's
context. The reviewer's value comes from evaluating the artifact without the
parent's accumulated rationalizations, and from having a short, clean window.

### 3.2 Read-only context-gatherer

Explores, reads, greps; returns a summary string. No shared mutable state, no
write tools in its surface.

Measured (SWE-Edit, Viewer + Editor split): **+2.1 pp resolve, −17.9% cost,
−34.5% main-agent input tokens.**

**Use a cheap model here.** A specialized Qwen3-8B editor matched GPT-5-nano;
putting GPT-5 in that slot gave **+0.4 pp at 5.8× cost.** Model tiering per
subagent role is part of the design, not an optimization.

This is also the cleanest lever on ADR-274 §5 failure #1 (wasted-context
accumulation): exploration output never enters the main window.

### 3.3 Deferred: coordinator

Only after single-agent is solid. Requires heavy context engineering to avoid
over-prescription.

## 4. Rejected: parallel writers, CoW fork/merge, CRDT join

Rejected on evidence for the coding domain. No rigorous positive coding result
exists; the negative evidence is production-scale and from a team that shipped
the architecture and walked it back.

**What is kept from Phase 1.6:** `JoinSet` spawning and semaphore-bounded
concurrency remain — they are how §3.1 and §3.2 subagents run concurrently with
each other. What is dropped is CoW state forking and CRDT merge, because
nothing writes concurrently and therefore nothing needs merging.

This is a substantial simplification: it removes a mergeable state type, the
merge-conflict semantics, and the entire class of bugs where two subagents
make locally-valid but jointly-incoherent edits.

## 5. Concurrency model

Retained from the existing loop and unchanged by this ADR:

- **Read-only tools run concurrently; state-mutating tools run sequentially.**
  This is the industry-convergent split and matches MCP's `readOnlyHint`.
- Bounded concurrency via semaphore; results returned in call order.
- Each tool runs in its own task so a panicking tool surfaces as a tool error
  rather than crashing the loop (already shipped).

## 6. Consequences

**Positive.** Removes the most complex unbuilt subsystem in the roadmap. Buys
nearly all demonstrated multi-agent upside — the reviewer and gatherer are the
two patterns with real production numbers — at a fraction of the complexity.
A `Fn(prompt) -> String` boundary is trivially testable and trivially
replayable (ADR-277).

**Negative.** Forecloses the "swarm of coders on one repo" demo. That demo has
no supporting evidence for coding tasks and would likely produce incoherent
results, so this is a cost worth paying — but it is a visible capability we are
choosing not to build.

**Interaction with ADR-107** (rvagent native swarm/WASM): swarm topology
remains valid for *independent* tasks across separate workspaces. This ADR
constrains concurrent writers **within a single workspace on a single task**,
which is the case the evidence covers.

## 7. Implementation

1. Subagent-as-tool trait: isolated context in, String out
2. Fresh-context reviewer (no inherited history)
3. Read-only gatherer with a write-free tool surface and cheap-tier model
4. Per-role model override
5. Remove CoW fork/merge and CRDT join from the Phase 1.6 scope
