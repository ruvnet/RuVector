---
adr: 274
title: "rvAgent Context Management: Masking over Summarization"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-103, ADR-252, ADR-273, ADR-275, ADR-276]
supersedes_parts_of: [ADR-252]
tags: [rvagent, harness, context, compaction, masking, long-horizon, sota]
---

# ADR-274 — rvAgent Context Management: Masking over Summarization

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Evidence base: `docs/research/rvagent-hermes-harness/04-sota-landscape.md` §4.

**This ADR reverses a standing design bet.** `03-roadmap.md` Phase 1.3
(middle-turn summarization), Phase 2.5 (ADR-252 coherence-weighted
compaction), and the shipped default middleware pipeline all rely on LLM
summarization as the primary context strategy. The evidence says that is close
to the worst available option.

## 1. Decision

1. **Observation masking is the default** context strategy, not summarization.
2. **Masked observations keep a dereferenceable ID** (addressable recall).
3. **Programmatic tool calling** is added as a first-class capability.
4. **Invariants are re-injected verbatim after every compaction**, never
   summarized.
5. **Context features are capability-gated per model tier.**
6. LLM summarization is retained only as a **rubric-guided fallback**.

## 2. Why the reversal

**Observation masking matches or beats LLM summarization at roughly half the
cost** (JetBrains, 250-turn SWE-bench trajectories, NeurIPS 2025 workshop:
+2.6% solve rate at 52% lower cost on Qwen3-Coder 480B).

The mechanism matters more than the number: **LLM summarization extended
trajectories 13–15%** by destroying natural stopping signals. The agent loses
the cue that it already finished something and keeps working. A hybrid cut
cost 7% below pure masking.

A placeholder reading `[test output, 2,847 tokens, exit 1, elided]` is more
useful than a mediocre summary, because it preserves the *shape* of history
without fabricating its contents.

**Addressable Recall Compaction** (arXiv 2607.25066) improves on plain masking
by leaving an ID the agent can dereference on demand. It beat full-context,
sliding window, LLM summary, structured state, *and* RAG memory — NIAH 99.40%
vs 88.12% for the best baseline. This removes the main objection to masking
(irreversible loss).

### 2.1 The largest lever is not compaction at all

On LOCA-bench, six context strategies were compared across four frontier
models at 128k context. **Programmatic tool calling — the model writes code
that orchestrates tools, so intermediate outputs never enter context — was the
only strategy positive on every model tested** (+6.0 to +13.3 points).

Summarize-and-continue compaction was the *weakest* of the obvious strategies
(+2.6 points on the model where it helped most).

This is absent from the current roadmap and is the single highest-value
context item.

### 2.2 Compaction silently erases invariants

"Governance Decay" (arXiv 2606.22528) shows safety constraints and
system-prompt instructions **erode through successive compaction cycles with no
failure signal.** This is not jailbreaking — it emerges from the compression
architecture itself.

Therefore: the task statement, acceptance criteria, safety constraints, and
system prompt are **re-emitted byte-identical** after each compaction. They are
never inputs to a summarizer. Cheap, and directly counters a documented
mechanism.

### 2.3 Context features can make weaker models worse

On LOCA-bench, giving DeepSeek-V3.2 a memory tool dropped it from 10.7% to
**8.0%**, and telling it its remaining context budget dropped it to **4.0%** —
while both features *helped* GPT-5.2 and Gemini-3-Flash.

Per-model-tier feature flags are therefore mandatory, not optional. A feature
that helps the frontier tier must not be enabled by default for weaker models.

## 3. Design

### 3.1 Masking

Replace old tool observations with typed placeholders carrying: tool name,
elided byte/token count, exit status where applicable, and a recall ID.
**Reasoning steps and actions are kept verbatim** — only observations are
masked. Keep the last N observations in full.

### 3.2 Addressable recall

Append-only, ID-addressable log of full observations. The placeholder carries
the handle; a `recall` tool dereferences it. No re-execution, no similarity
search, no embedding index.

### 3.3 Thresholds

Compact at **70–80% of nominal context**, not 95%. Treat nominal window as
roughly **2× the reliable working budget** — degradation begins well under half
the advertised window, and it is silent. Anthropic's own API default compacts
at 150k on a 200k-class window, which is the most useful real-world datapoint.

Do not pursue larger context windows as a substitute: sweeps plateau ~114k and
documented ceilings sit at 96–112k (ADR-273 §4).

### 3.4 Fallback summarization

When summarization is used, the **rubric is the load-bearing part**, not the
tool. Preserve explicitly: task statement, files touched, failing tests,
decisions made, unresolved issues. Offering a compaction tool without rubric
guidance produces uneven behavior; a paragraph of guidance closes the gap.

Guard against recursive summarization of summaries — a documented and fixed bug
in Codex's rewrite.

### 3.5 Prompt-cache interaction

Compaction invalidates every cached prefix downstream of the break. Keep a
byte-stable system prefix and place cache breakpoints *before* the volatile
region. Design for 10+ compaction cycles per session.

## 4. Retrieval policy

Layered escalation, not a choice between grep and semantic:

1. **ripgrep** — known symbol, exact identifier, error string
2. **Structural search** (ast-grep / LSP / `syn`) — callers of X, impls of Y
3. **Semantic** — only for conceptual cross-cutting queries

**Never route a short keyword query to a dense retriever.** CoREB (arXiv
2605.04615) shows short keyword queries — "auth flow", "user service", exactly
the shape agents emit — collapse essentially every embedding model to near-zero
nDCG@10. This explains the 2025 industry migration away from vector search in
coding agents mechanistically.

Summarize search output by default: return paths and match counts, require a
second call to read contents.

Any index over code the agent is editing is **stale by construction**. If one
is added later it needs invalidation-on-write or explicit staleness bounds.

## 5. What breaks first in long runs

Design priority follows the observed failure order:

1. **Wasted-context accumulation → attention dilution** — earliest, universal,
   invisible. Unaided coding agents waste ~1-in-3 file reads.
2. **History error accumulation** — the largest single driver. Process-level
   failures are **72.5%** of long-horizon failures (HORIZON, arXiv 2604.11978).
   Errors compound *between* steps. Non-linear: sharp collapse past a
   domain-specific threshold.
3. **Compaction-induced loss** — self-inflicted by the mitigation for #1.
4. **Goal / identity drift** — *downstream* of 1–3, not independent.
5. **Hallucinated state** — what #3 looks like when the summary is confidently
   wrong rather than merely lossy.

The evidence does **not** support treating goal drift as the primary problem.
It is the observable end-stage of context and error problems, which is why
this ADR targets 1–3 and ADR-273 targets error compounding.

## 6. Consequences

**Positive.** Cheaper and faster than summarization (no model call on the hot
path). Deterministic, so it does not itself become a source of nondeterminism
in replay (ADR-277). No fabrication risk. Preserves stopping signals.

**Negative.** ADR-252 (coherence-weighted compaction) is demoted from the
default path to a fallback. Work already done there is not wasted — it becomes
the rubric-guided fallback of §3.4 — but it is no longer the primary strategy.

The `summarization` middleware currently in the shipped default pipeline must
be reordered behind masking, or removed from the default set.

**Risk.** Masking loses information that a good summary would have retained.
Mitigated by §3.2 addressable recall: the information is still there and still
reachable.

## 7. Implementation order

1. Observation masking with last-N-in-full (ADR-273 floor item 2)
2. Verbatim invariant re-injection (§2.2) — cheap, high consequence
3. Addressable recall log + `recall` tool (§3.2)
4. Programmatic tool calling (§2.1) — highest value, largest effort
5. Capability gating (§2.3)
6. Demote `summarization` middleware to fallback (§3.4)
