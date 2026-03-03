# ADR-061: Reasoning Kernel Architecture — Brain-Augmented Targeted Reasoning

**Status**: Accepted
**Date**: 2026-02-27
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-060 (Shared Brain Capabilities), ADR-059 (Shared Brain Google Cloud), ADR-057 (Federated Transfer Learning)

## 1. Context

The Shared Brain (ADR-059, ADR-060) provides a learning substrate: shared MicroLoRA weights, semantic retrieval, quality-gated knowledge, and witness-verified provenance. But the brain is the memory, not the thinker. The thinker is ruvllm, and it needs a reasoning kernel that leverages the brain to achieve "on par or better" performance compared to traditional LLMs on targeted domains.

A traditional LLM is a static generalist with weak memory and no accountability. The RuVector system is a living specialist with a shared skill substrate, a semantic retrieval layer, a policy and proof layer, and a parameter growth path. That combination beats raw model size in production for bounded domains.

This ADR defines the reasoning kernel architecture: what the brain can and cannot do, how ruvllm grows through the brain, and the concrete build plan for targeted domain reasoning.

## 2. What the Brain Can and Cannot Do

### 2.1 What It Does Well

**Domain reasoning at high reliability**: When the domain is bounded — codebase-specific debugging, infrastructure incident response, RVF correctness, mincut-based coherence decisions — the brain makes the system consistently better than a generic LLM because it accumulates exact successful trajectories and policies.

**Fast improvement without retraining**: MicroLoRA plus retrieval shifts behavior quickly. Short-cycle learning driven by votes and outcomes. A 2KB weight sync moves faster than any model upgrade.

**Better behavior under governance**: Witness logs, proof-gated mutation, drift monitors, and rollbacks give controlled reasoning. That often matters more than raw model IQ in production.

### 2.2 What It Cannot Do

**Create novel general reasoning from scratch**: LoRA adapts a model. It does not invent capacity absent in the base network. If ruvllm is small and weak at general reasoning, MicroLoRA and retrieval help substantially, but there is a ceiling.

**Replace a large model on broad world knowledge**: The brain stores distilled artifacts, not the full distribution of human text. The system wins by focus, not by universality.

## 3. Reasoning Kernel Design

### 3.1 Reasoning Kernel Definition

The reasoning kernel is the fixed protocol by which ruvllm processes a task using the brain.

**Input**: A task type from a bounded set (initially: fix failing tests, explain ADR contradictions, generate RVF segments, perform mincut-gated decisions).

**Output**: A fixed action format with tool calls, intermediate checks, and a final answer format.

**Invariant**: Every reasoning step produces a witnessable artifact — a check result, a test pass, a proof token, or a retrieval citation from the brain. This makes reasoning verifiable and replayable.

### 3.2 Memory Architecture

The kernel operates with two memory layers sourced from the brain:

**Working memory** (per-task, ephemeral):
- RAG into the current context window using `brain_search`
- Local repo context (files, errors, stack traces)
- Retrieved similar trajectories with quality scores

**Skill memory** (cross-session, persistent):
- High-quality patterns from `brain_search(min_quality: 0.7)`
- PolicyKernel snapshots that bias the plan generator
- CostCurve data that predicts acceleration factors
- MicroLoRA consensus weights that transform embeddings toward effective solution regions

### 3.3 Reasoning Loop

```
1. CLASSIFY task type from bounded set
2. PULL working memory: brain_search(query, category, min_quality)
3. PULL skill memory: brain_sync(direction: "pull") for MicroLoRA consensus
4. PLAN using retrieved patterns + local context
5. EXECUTE with tool calls, recording each step as witnessable artifact
6. VERIFY against success criteria (test pass, proof token, etc.)
7. If success:
   a. brain_share(category, result) — contribute solution
   b. brain_vote(id, "up") on retrieved patterns that helped
   c. brain_sync(direction: "push") — export local LoRA deltas
8. If failure:
   a. brain_vote(id, "down") on retrieved patterns that misled
   b. Record failure trajectory for negative example mining
```

## 4. Training Pipeline — Growing ruvllm Through Brain Data

The brain accumulates exactly the data types needed for targeted model improvement. Training happens in three layers.

### 4.1 Layer A: Preference Learning from Votes

**Data source**: `brain_vote` events produce (memory, direction) pairs. Over time, this builds a preference dataset: for a given task type, which solution patterns are preferred (upvoted) vs rejected (downvoted).

**Training method**: Direct Preference Optimization (DPO) or reward model training on the targeted model. The Bayesian quality scores (`BetaParams`) provide confidence-weighted preference signals — high-observation memories with clear quality separation are stronger training signals than low-observation ones.

**Frequency**: Batch, triggered when accumulated vote pairs exceed a threshold (e.g., 500 new preference pairs per domain).

### 4.2 Layer B: Imitation Learning from Successful Trajectories

**Data source**: SONA trajectories with quality > 0.7, exported via `AgentExport` from `sona::training::federated`. Each trajectory includes the sequence of tool calls, intermediate states, and final outcome.

**Training method**: Supervised fine-tuning on high-quality trajectories. Narrow and task-specific — only trajectories from the target domain (e.g., "debug" category memories with quality > 0.7).

**Curriculum**: Use `brain_partition` clusters to group trajectories by difficulty. Train easy clusters first, hard clusters later (curriculum learning).

### 4.3 Layer C: Continual Learning with Forgetting Control

**Data source**: All layers A and B, plus replay buffers keyed by domain and mincut partition.

**Training method**: Elastic Weight Consolidation (EWC) with lambda=2000.0 (matching SONA's existing EWC configuration). The Fisher information matrix is computed per-domain partition, so the model preserves capabilities in well-established domains while adapting to new ones.

**Replay buffer design**: Each mincut partition maintains a fixed-size replay buffer (100 exemplars). When a new exemplar is added, the lowest-quality one is evicted. This prevents catastrophic forgetting on older domains while allowing new domains to develop.

### 4.4 Training Data Flow

```
Brain Events                     Training Pipeline
-----------                     ------------------
brain_vote(up/down)      -->    Layer A: Preference pairs
brain_share(quality>0.7) -->    Layer B: Imitation trajectories
brain_partition clusters  -->    Layer C: EWC domains + replay buffers
brain_sync (LoRA deltas)  -->    Continuous adapter refinement
```

## 5. Parameter Growth Strategy

Add capacity only when a measured ceiling is reached. If the model cannot solve a class of tasks even with perfect retrieval and high-quality patterns, then expand.

### 5.1 Growth Decision Criteria

Trigger parameter growth when:
- Success rate on a domain benchmark plateaus for 3+ epochs despite growing memory
- Retrieval recall@10 is > 80% (the right knowledge is available) but task success < 60% (the model cannot use it)
- MicroLoRA weight norms are saturating (approaching clipping bounds consistently)

### 5.2 Growth Options (Ordered by Cost)

**Option 1: Expand adapter rank** (cheapest)
Increase MicroLoRA rank from 2 to 4 or 8 for specific layers. Keep base frozen. This quadruples expressiveness for 4x parameter cost (still < 10KB per sync).

**Option 2: Add specialist heads per domain partition** (moderate)
Mixture-of-experts routing using mincut partitions as the router signal. Each partition gets a small specialist head. Base model shared, routing learned from partition assignments.

**Option 3: Train a larger ruvllm tier and distill** (expensive)
Use the brain to provide curriculum and hard negative examples. Distill from the larger model back to the smaller one, keeping the brain-augmented inference path.

## 6. Evaluation Framework

### 6.1 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Quality** | Success rate on fixed benchmark per domain | >= 80% on target domain |
| **Efficiency** | Median tokens + tool calls per solved task | >= 30% reduction vs baseline |
| **Reliability** | Variance reduction across epochs | Std dev < 0.5x baseline |
| **Safety** | Poisoning simulation behavior deviation | < 5% regression under 30% adversarial |

### 6.2 Three-System Comparison

For each target domain, compare:
1. **Baseline ruvllm**: No brain, no retrieval, no LoRA
2. **ruvllm + brain retrieval**: brain_search for working memory, no LoRA
3. **ruvllm + brain retrieval + MicroLoRA**: Full pipeline with federated weights

Pass criteria:
- System 3 achieves >= 20% higher success rate than System 1
- System 3 achieves >= 30% fewer tokens per success than System 1
- Regressions held under 5% across 3 consecutive sync epochs

### 6.3 Initial Benchmark: Fix Failing Tests

**Domain**: Rust crate test failures across the RuVector monorepo.

**Benchmark size**: 100 items (real test failures extracted from CI history).

**Task format**: Given a failing test and its error output, produce a fix that makes the test pass.

**Success criteria**: The fix compiles and the specific test passes. No regressions in other tests.

**Why this domain first**:
- Highest frequency task in daily development
- Binary success signal (test passes or does not)
- Bounded — Rust compiler errors are structured and classifiable
- Rich trajectory data from existing sessions
- Direct impact on developer velocity

## 7. Hybrid Lane Architecture

ruvllm is offline-capable by default. The brain is an optional accelerator, not a dependency.

### 7.1 Degradation Path

| Connectivity | Available Features | Performance |
|-------------|-------------------|-------------|
| **Full** (brain reachable) | All 11 MCP tools + MicroLoRA sync + retrieval | Best |
| **Partial** (brain slow/intermittent) | Local SONA + cached consensus LoRA + hash features | Good |
| **Offline** (no brain) | Local SONA + hash-only features | Baseline |

The `BrainEmbedder` already implements this degradation:
- If consensus LoRA cached: use it (no network needed)
- If SONA engine available: use local LoRA from SONA
- Otherwise: structured hash features only

### 7.2 Sync Cadence

`brain_sync` is explicit opt-in, not automatic. Recommended cadence:
- **On session start**: `brain_sync(direction: "pull")` — get latest consensus
- **On session end**: `brain_sync(direction: "push")` — export local learning
- **Periodically during long sessions**: Every ~100 embeddings processed

## 8. Implementation Status

### Implemented (in this branch)

| Component | Status |
|-----------|--------|
| Structured hash features (Stage 1 embedding) | Shipped, 6 new tests |
| MicroLoRA forward pass (Stage 2 embedding) | Shipped |
| Consensus import/export (`BrainEmbedder`) | Shipped |
| `GET /v1/lora/latest` | Shipped |
| `POST /v1/lora/submit` | Shipped |
| Gate A policy validation | Shipped |
| Gate B robust aggregation (median + MAD + reputation) | Shipped |
| Consensus drift monitoring + rollback | Shipped |
| `brain_sync` MCP tool | Shipped |
| Hybrid degradation path | Shipped |

### Deferred (future work)

| Component | Priority | Dependency |
|-----------|----------|------------|
| Training data export endpoints | High | Need trajectory format spec |
| Preference pair extraction from votes | High | Need DPO training loop |
| Trajectory quality filtering | Medium | Need benchmark suite |
| EWC per-partition Fisher matrices | Medium | Need partition stability |
| Specialist head routing | Low | Need ceiling measurement |
| 100-item benchmark harness | High | Need CI history extraction |

## 9. Answers to Design Questions

**Q: What is the narrowest set of tasks to beat a traditional LLM on first?**

A: Code debugging — specifically, fix failing tests in Rust crates. Highest frequency, clearest success signal, bounded domain, rich trajectory data already available.

**Q: Should ruvllm remain fully offline capable by default?**

A: Yes, with a hybrid lane for acceleration. The `BrainEmbedder` degradation path (consensus LoRA -> local SONA -> hash-only) ensures offline functionality. The brain is an accelerator, not a dependency.

## 10. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-057 | Federated RVF Transfer Learning — protocol foundation |
| ADR-058 | Hash Security Optimization — SHAKE-256 for content integrity |
| ADR-059 | Shared Brain Google Cloud — infrastructure and security |
| ADR-060 | Shared Brain Capabilities — sub-capabilities and business outcomes |
