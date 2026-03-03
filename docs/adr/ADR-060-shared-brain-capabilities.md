# ADR-060: Shared Brain Capabilities — Federated MicroLoRA Intelligence Substrate

**Status**: Accepted
**Date**: 2026-02-27
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud Deployment), ADR-057 (Federated RVF Transfer Learning)

## 1. Context

The Shared Brain (ADR-059) turns isolated Claude Code sessions into a continuously improving, public, zero-trust learning substrate. The federated MicroLoRA communication bridge — where tiny shared weights (2KB per sync) move faster than any single model upgrade — enables seven distinct capabilities that were previously impossible with isolated sessions.

This ADR documents the sub-capabilities, practical use cases, business outcomes, and architectural considerations that emerge from the deployed system.

## 2. Core Capabilities

### 2.1 Cold Start Disappears

A brand-new session can land on the right approach in minutes because it downloads the current MicroLoRA consensus plus high-quality priors and policies. The "first time tax" — where every new repo, bug class, or architecture decision requires re-discovery — is eliminated.

**Mechanism**: `brain_sync(direction: "pull")` downloads consensus LoRA weights + `brain_search` retrieves relevant prior knowledge. The structured hash features provide immediate lexical n-gram-level matching; the MicroLoRA transform adds learned semantic refinement on top.

**Primary metric**: Median time to first relevant hit at top-5 on a fixed labeled query set, measured continuously. Secondary: median tool calls until first correct fix. Both tracked per-epoch to observe convergence.

**Measurable outcome**: New sessions converge on known solution patterns without rediscovering them. The acceleration is proportional to the quality and density of the knowledge graph in the relevant domain.

### 2.2 Debugging Becomes Cumulative

When one session finds a reliable fix pattern for a failure mode, the brain learns that trajectory. Future sessions retrieve it by meaning, not by exact text match. Debugging becomes a shared skill library that grows with every session.

**Mechanism**: The SONA engine records debugging trajectories as `LearnedPattern` entries with quality scores. Successful fixes (high quality feedback via `brain_vote(direction: "up")`) strengthen the corresponding embedding clusters. The MicroLoRA adapts to prefer embeddings near high-quality fix patterns.

**Example flow**:
1. Session A discovers that `tokio::spawn` deadlocks when combined with `parking_lot::RwLock` in certain patterns
2. Session A shares the fix pattern via `brain_share(category: "debug", ...)`
3. Session B encounters a similar deadlock
4. Session B's `brain_search` retrieves Session A's fix — not because the code is identical, but because the structured hash features capture the shared lexical tokens ("deadlock", "rwlock", "spawn") and the MicroLoRA has learned the semantic proximity between those tokens and the solution space

### 2.3 A Shared "Taste" for Good Solutions

Votes plus quality gating train the MicroLoRA to prefer solution patterns that actually worked downstream. Over time the system learns what "good" looks like for the ecosystem — not what is popular, but what is effective.

**Invariant: Quality signals are sourced only from authenticated contributors.** Voting is a write operation requiring API key authentication and contributor pseudonym derivation. This prevents the vote stream — which directly influences MicroLoRA gradient signals — from becoming the easiest poisoning path. Anonymous or unauthenticated users cannot influence quality scores.

**Mechanism**: The Bayesian quality score (`BetaParams { alpha, beta }`) accumulates vote evidence. Memories with `quality_score.mean() < 0.3` after `observations() >= 5` are auto-archived. The MicroLoRA's gradient signal is quality-weighted: high-quality patterns get stronger representation in the embedding space, shifting the consensus toward effective solutions.

**Anti-popularity bias**: Quality gating is based on Bayesian posteriors with flat priors, not raw vote counts. A memory with 3 upvotes and 0 downvotes (quality=0.8) is not assumed superior to one with 50 upvotes and 10 downvotes (quality=0.85). The Beta distribution's uncertainty naturally favors well-tested knowledge.

### 2.4 Transfer Learning Across Domains

TransferPrior and PolicyKernel segments enable moving learning from "code review" into "security audit" or from "infra ops" into "edge deployment" with measurable acceleration and confidence.

**Mechanism**: `brain_transfer(source_domain, target_domain)` uses `DomainExpansionEngine::initiate_transfer()` with sqrt-dampened priors (prevents overfit to source domain). `TransferVerification::verify()` requires both conditions: target improved AND source not regressed. `CostCurve` tracks the acceleration factor.

**Cross-domain examples**:
- Rust ownership patterns (source) -> TypeScript lifetime management patterns (target)
- REST API security (source) -> GraphQL security (target)
- Kubernetes deployment (source) -> Edge container deployment (target)

**Failure mode**: Transfer can inject incorrect analogies when domains are superficially similar but structurally different. Transfer priors are never auto-applied. They require explicit `brain_transfer` invocation, downstream verification in the target domain (target_after > target_before), and non-regression in the source domain (source_after >= source_before). If either condition fails, the transfer is rejected and the `TransferVerification.promotable` flag is false.

### 2.5 A Public Marketplace of Small, Auditable Intelligence Artifacts

Because everything is an RVF container with witness chains, knowledge can be exchanged as signed, replayable, revocable units. This is the basis for monetization, governance, and compliance.

**Canonical RVF container layout: exactly ten segments per memory** (from ADR-059 Section 6):

| # | Type | Content |
|---|------|---------|
| 1 | `MANIFEST (0x05)` | Segment directory |
| 2 | `VEC_SEG (0x01)` | SONA embedding (f32 x dim) |
| 3 | `META_SEG (0x07)` | Title, content, tags, category |
| 4 | `TransferPrior (0x30)` | Domain priors (if any) |
| 5 | `PolicyKernel (0x31)` | Best config snapshot |
| 6 | `CostCurve (0x32)` | Acceleration data |
| 7 | `FederatedManifest (0x33)` | Export metadata |
| 8 | `DiffPrivacyProof (0x34)` | Noise attestation |
| 9 | `RedactionLog (0x35)` | PII strip attestation |
| 10 | `WITNESS_SEG (0x0A)` | SHAKE-256 operation chain |

Ed25519 signature covers segments 1-10. This layout is locked — code comments and test assertions reference exactly ten segments.

**Auditability**: Every operation (share, vote, delete, transfer, sync) is recorded in the witness chain. Any participant can verify the chain independently.

### 2.6 An Anti-Poisoning Collective Memory

Robust aggregation, mincut partitioning, drift monitors, and reputation gating let the system absorb noisy public input without letting any single actor steer the centroid.

**Seven layers of defense**:
1. **Input sanitization**: 12-pattern PII strip, schema validation, size limits
2. **Server-side verification**: Re-check PII, witness chain, signature, embedding bounds
3. **DoS protection**: BudgetTokenBucket rate limiting, challenge nonces
4. **Byzantine aggregation**: 2-sigma outlier exclusion, per-parameter median + MAD trimming
5. **Multi-factor reputation**: `composite = accuracy^2 x uptime x stake_weight`, cold start at 0.1
6. **Drift monitoring**: VectorDelta centroid tracking, automatic rollback on excessive drift
7. **Vote authentication**: Quality signals sourced only from authenticated contributors (see Section 2.3)

**MicroLoRA-specific defenses**:
- Gate A policy validation: shape, NaN/Inf, [-2.0, 2.0] clipping bounds, L2 norm < 100, evidence >= 5
- Gate B robust aggregation: per-parameter median, 3xMAD trimming, reputation-weighted mean
- Consensus drift monitoring: L2 distance between raw (unclipped, unnormalized) consensus weight vectors across epochs; rollback threshold 5.0 calibrated for rank-2 on 128-dim (512 total parameters, weights clipped to [-2, 2] before submission). This threshold must be recalibrated if rank or dimension changes.

### 2.7 Always-On Agent Coordination

The brain becomes the local edge brain for real systems. It can watch logs, configs, incidents, and sensor streams, then pull collective learnings on demand while keeping sensitive raw data local.

**Architecture**: The MCP stdio server (`mcp-brain`) runs as a local process within each Claude Code session. It communicates with the Cloud Run backend (`mcp-brain-server`) via HTTPS. Raw data never leaves the local session — only distilled embeddings, LoRA weights, and sanitized metadata are transmitted.

**Edge deployment pattern**:
1. Agent monitors local system (logs, metrics, alerts)
2. When anomaly detected, `brain_search` queries collective for similar patterns
3. If match found, apply fix pattern from collective knowledge
4. `brain_share` contributes the resolved incident as new knowledge
5. `brain_sync` updates local LoRA weights with any new consensus

## 3. Embedding Pipeline Clarifications

### 3.1 Hash Features Are Lexical, Not Semantic

Structured hash features provide lexical and structural similarity. Texts sharing word-level n-grams will have higher cosine similarity than texts with disjoint vocabularies. **Semantics come only from the MicroLoRA transform**, which is learned from quality signals over time. A reader should not assume that hashing alone captures meaning — it captures co-occurrence structure that the LoRA refines into approximate semantic proximity.

### 3.2 Symmetric Pipeline for Store and Query

Both stored vectors and query vectors pass through the same pipeline stages in the same order:

```
text -> structured_hash_features() -> apply_lora_transform() -> L2_normalize()
```

The DentateGyrus pattern separation layer in `cognitive.rs` is used only for server-side indexing and anomaly detection. It is **not** applied to query embeddings or to stored embedding vectors used for similarity search. This prevents asymmetric drift between index and query representations.

### 3.3 Transfer Learning Boundary Conditions

Transfer priors are dampened by sqrt-scaling and require dual verification (target improved AND source not regressed). Transfers that fail verification are rejected with `TransferVerification.promotable = false`. There is no auto-apply path — every transfer requires explicit invocation and evidence.

## 4. Business Outcomes

### 4.1 Lower Cost Per Solved Task

Fewer retries and faster convergence. When a session can pull relevant prior knowledge and learned LoRA weights instead of discovering solutions from scratch, the number of API calls, tool invocations, and human review cycles decreases.

### 4.2 Higher Consistency Across Projects

Best practices become default. The MicroLoRA consensus naturally encodes the patterns that have been most successful across all contributing sessions. New projects inherit this collective "taste" from the first `brain_sync`.

### 4.3 Defensible Moat

The shared weights and witness-logged outcomes are proprietary experience, not just code. The accumulated LoRA weights represent the collective intelligence of all contributing sessions — a continuously growing asset that cannot be replicated by copying code alone.

### 4.4 Economic Accounting

```
Monthly brain value = (hours_saved x blended_hourly_rate) - cloud_costs - review_time
```

**Initial targets**:
- Reduce median fix time by 2x on "fix failing tests" workflow (30-run A/B test)
- Reduce LLM token spend per fix by 30% due to fewer retries (measured via API call counts in brain-on vs brain-off runs)

**Cloud cost estimate**: Cloud Run at 0-10 instances (2 CPU / 2Gi RAM), Firestore, GCS. Expected < $50/month at moderate traffic. Break-even requires saving approximately 1 developer-hour per month at $50/hr blended rate.

## 5. Target Users

### Primary: Developers Using Claude Code

Individual developers and small teams using Claude Code for daily software engineering tasks. The brain provides immediate value through cold-start elimination and cumulative debugging knowledge.

### Secondary: Enterprise Teams Running Always-On Agents

Teams deploying autonomous agents in production environments. The brain provides collective intelligence for incident response, configuration management, and cross-project coordination.

## 6. Access Model

### Public Read, Gated Write

- **Read**: Search, list, get, drift, partition, status, and lora/latest are publicly accessible with per-identity rate tiers (1000 reads/hr authenticated, 100 reads/hr anonymous)
- **Write**: Share, vote, delete, transfer, lora/submit require API key authentication with contributor pseudonym derivation
- **Voting is a write operation**: Quality signals are sourced only from authenticated contributors. This is a deliberate choice — public voting without identity would be the easiest Sybil/poisoning attack surface
- **Reputation gate**: New contributors start at 0.1 composite reputation. Their contributions are weighted 10x less in aggregation until they accumulate sufficient positive votes
- **System contributors**: Seed data uses `ruvector-seed` pseudonym with max reputation (1.0) and system flag exempting from decay

### Read Abuse Budgeting

Even public read endpoints can dominate cost. Defenses:
- **Query complexity limits**: Search limited to `limit <= 100`, embedding dimension fixed at 128
- **Response caching**: 60-second TTL cache on `/v1/lora/latest` (consensus changes only at epoch boundaries). 30-second cache on `/v1/status` and `/v1/drift`
- **Per-identity rate tiers**: Authenticated: 1000 reads/hr. Anonymous (no API key): 100 reads/hr. Exhausted: 429 with Retry-After header

## 7. Federation Cadence and Stability

### Aggregation Trigger

Aggregation fires on **submission count**: when `pending_submissions >= min_submissions` (default: 3). There is no time-based trigger — aggregation requires evidence from multiple independent sessions.

### Minimum Quorum

A new consensus epoch is published only when:
1. At least `min_submissions` (3) accepted submissions are pending
2. After Gate B trimming, at least `min_submissions` (3) inlier submissions remain
3. If both conditions fail, pending submissions accumulate until the threshold is met

### Rollback Semantics

When drift rollback triggers, **only MicroLoRA consensus weights are rolled back**. The accepted memory set is preserved. Memories are independently quality-gated (Bayesian BetaParams) and witness-verified — rolling them back would destroy valid, independently verified knowledge. The LoRA transform is the only aggregated learned artifact that can diverge from the population; memories are individually validated on ingestion.

After rollback, the pending submission queue is cleared to prevent the same batch from immediately re-triggering the same divergent consensus.

## 8. Acceptance Criteria

### Embedding Quality

- 200 labeled pairs (similar/unrelated)
- After 3 sync epochs with >= 10 contributors each
- Similar pairs should exceed unrelated by >= 0.25 cosine distance
- Search recall@10 should improve >= 30% vs hash-only baseline

### Workflow Acceleration

Pick one workflow (e.g., "fix failing tests in a TypeScript repo"):
- Measure median time to first passing build: brain off vs brain on
- 30 runs each condition
- Target: >= 2x improvement once the brain has a few hundred high-quality memories and a few LoRA sync epochs

### Security: Poisoning Resistance

- Zero PII leakage through embedding or LoRA weight analysis
- Byzantine aggregation excludes > 95% of adversarial submissions in poisoning simulations
- Weight drift rollback triggers correctly when consensus shifts > 5.0 L2 distance
- Poisoning simulation: 30% adversarial submitters must fail to move consensus weights beyond the drift threshold for more than one epoch

### Weekly Regression Suite

Run continuously with:
1. Fixed labeled query set (200 pairs)
2. Fixed workflow harness (e.g., "fix failing tests")
3. Poisoning simulation (30% adversarial contributors)

Pass criteria:
- Recall@10 >= 30% improvement vs hash-only baseline
- Median time to first passing build >= 2x improvement
- Adversarial consensus deviation contained to <= 1 epoch before rollback

## 9. Implementation Status

All capabilities described in this ADR are implemented on branch `feat/adr-030-hash-security-optimization`. Test counts:

| Crate | Tests | Status |
|-------|-------|--------|
| `mcp-brain` (client) | 33 | All passing |
| `mcp-brain-server` (server) | 28 | All passing |
| `mcp-brain` (doc-tests) | 1 | Passing |
| **Total** | **62** | **All passing** |

### Shipped Endpoints

| Method | Path | Category |
|--------|------|----------|
| GET | `/v1/health` | Infrastructure |
| GET | `/v1/challenge` | Auth |
| POST | `/v1/memories` | Write |
| GET | `/v1/memories/search` | Read |
| GET | `/v1/memories/list` | Read |
| GET | `/v1/memories/{id}` | Read |
| POST | `/v1/memories/{id}/vote` | Write |
| DELETE | `/v1/memories/{id}` | Write |
| POST | `/v1/transfer` | Write |
| GET | `/v1/drift` | Read |
| GET | `/v1/partition` | Read |
| GET | `/v1/status` | Read |
| GET | `/v1/lora/latest` | Read |
| POST | `/v1/lora/submit` | Write |

Total: 14 endpoints (8 read, 5 write, 1 infrastructure).

### Shipped MCP Tools

11 tools: `brain_share`, `brain_search`, `brain_get`, `brain_vote`, `brain_transfer`, `brain_drift`, `brain_partition`, `brain_list`, `brain_delete`, `brain_status`, `brain_sync`.

### Deferred

- `GET /v1/lora/stats` (per-epoch aggregation statistics) — deferred until aggregation history is implemented
- `POST /v1/lora/challenge` (proof-of-work challenge for anonymous submitters) — deferred until PoW is required by traffic patterns

## 10. RVF AGI Stack Integration (ADR-075)

The capabilities described in this ADR — PII stripping (Section 2.6, Layer 1), differential privacy (Section 2.5, Segment 8), witness chains (Section 2.5, Segment 10), and RVF container construction (Section 2.5) — were originally documented as design goals. ADR-075 implements these using the production RVF crate stack:

| Capability | Inline (Previous) | RVF Crate (ADR-075) |
|---|---|---|
| PII stripping | 8 string patterns in `verify.rs` | `rvf-federation::PiiStripper` (12 regex rules) |
| Diff privacy | Not implemented | `rvf-federation::DiffPrivacyEngine` (Gaussian, feature-gated) |
| Witness chains | String-step SHAKE-256 in `verify.rs` | `rvf-crypto::create_witness_chain` (73-byte linked entries) |
| Container build | Client-provided RVF bytes only | `rvf-wire::write_segment` (server-side construction) |
| Adversarial detect | Not implemented | `rvf-runtime::is_degenerate_distribution` (log-only) |
| Negative cache | Not implemented | `rvf-runtime::NegativeCache` (query signature blacklist) |

The 10-segment canonical container layout (Section 2.5) is now constructed server-side by the `pipeline.rs` module, with segment count reported in `ShareResponse.rvf_segments`.

## 11. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-057 | Federated RVF Transfer Learning — foundational protocol |
| ADR-058 | Hash Security Optimization — SHAKE-256 content hashing |
| ADR-059 | Shared Brain Google Cloud Deployment — infrastructure and security |
| ADR-075 | Wire Full RVF AGI Stack — replaces inline crypto with production RVF crates |
