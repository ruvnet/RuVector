# ADR-069: Edge-Net and Pi Brain Integration — Distributed Compute Intelligence

**Status**: Proposed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-062 (Brainpedia Architecture), ADR-063 (WASM Executable Nodes), ADR-064 (Pi Brain Infrastructure), ADR-066 (SSE MCP Transport)

## 1. Context

Two complementary systems exist in the ruvector ecosystem, both deployed on Google Cloud Run:

**Pi Brain** (`ruvbrain` at `pi.ruv.io`) is a centralized shared intelligence substrate. It provides a knowledge graph with Bayesian quality scoring, federated MicroLoRA consensus weights, structured hash embeddings, MinCut graph partitioning, and SONA self-optimization. It exposes 22 tools via SSE MCP transport (`/sse`) and a REST API (`/v1/*`). All operations are protected by witness chains and a challenge-nonce authentication system. It runs as a single Rust binary on Cloud Run.

**Edge-Net** (`examples/edge-net/`) is a distributed P2P browser compute network. Browser visitors contribute idle CPU cycles via Web Workers running Rust/WASM. Contributors earn rUv (Resource Utility Vouchers) based on compute donated. The network provides HNSW vector search, MicroLoRA adaptation, federated gradient gossip, entropy-based consensus, collective memory with hippocampal replay, and an adversarial coherence engine (RAC). It uses Pi-Key Ed25519 identity, a CRDT-based credit ledger, and a browser-based MCP server (`WasmMcpServer`).

These systems operate independently. The brain holds curated knowledge but is bottlenecked by single-origin compute for embedding generation, similarity search, and LoRA training. Edge-net has distributed compute capacity but no centralized knowledge substrate to draw from or contribute to. Connecting them creates a flywheel: edge nodes contribute compute that powers brain operations, brain knowledge improves edge node task routing, and rUv earned from brain operations incentivizes sustained edge participation.

### Current Deployment

| Service | URL | Platform |
|---------|-----|----------|
| `ruvbrain` (pi brain) | `https://pi.ruv.io` | Cloud Run, us-central1 |
| `edge-net-dashboard` | `https://edge-net-dashboard-875130704813.us-central1.run.app` | Cloud Run, us-central1 |
| `edge-net-genesis` | `https://edge-net-genesis-875130704813.us-central1.run.app` | Cloud Run, us-central1 |
| `edge-net-relay` | `https://edge-net-relay-875130704813.us-central1.run.app` | Cloud Run, us-central1 |

## 2. Decision

Integrate edge-net's distributed compute network with pi brain's knowledge substrate. Edge-net nodes become first-class brain contributors: they earn rUv for compute that powers embedding generation, vector similarity search, MinCut partitioning, and federated LoRA aggregation. The brain gains horizontally scalable compute; edge-net gains access to curated knowledge and a concrete economic purpose for contributed cycles.

The integration uses the edge-net relay as a bridge between browser WASM nodes and the brain's REST/MCP APIs. Edge-net's existing `WasmMcpServer` speaks MCP and can proxy requests to the brain's SSE MCP transport. Pi-Key identity maps to the brain's contributor pseudonym system (SHAKE-256 derivation) so that edge contributions are attributed and auditable.

## 3. Architecture

### 3.1 Integrated System Diagram

```
 Browser Nodes (WASM)                        Cloud Run Services
 ====================                        ==================

 +-----------+  +-----------+  +-----------+
 | edge-net  |  | edge-net  |  | edge-net  |
 |  node A   |  |  node B   |  |  node C   |    Contributors
 | (browser) |  | (browser) |  | (browser) |    earn rUv for
 +-----+-----+  +-----+-----+  +-----+-----+    brain compute
       |               |               |
       |   WebSocket   |   WebSocket   |
       +-------+-------+-------+-------+
               |                               +-----------------+
        +------+------+                        | edge-net-genesis|
        | edge-net    |   rUv ledger sync      | (Cloud Run)     |
        |   relay     +----------------------->| QDAG ledger     |
        | (Cloud Run) |                        | Node registry   |
        +------+------+                        +-----------------+
               |
               | Brain API bridge
               | (REST or SSE MCP)
               |
        +------+------+                        +-----------------+
        |  ruvbrain   |                        | edge-net        |
        |  (pi brain) |                        |   dashboard     |
        |  pi.ruv.io  |                        | (React, Cloud   |
        |             |                        |  Run)           |
        | Knowledge   |                        +-----------------+
        | graph, LoRA |
        | consensus,  |
        | embeddings  |
        +-------------+
```

### 3.2 Data Flow: Brain Operation Distributed to Edge

```
1. Brain receives search query via MCP or REST
       |
2. Brain decomposes into distributable subtasks
       |
3. Subtasks sent to relay as compute tasks
       |
4. Relay fans out to available edge nodes
       |
5. Edge nodes execute (HNSW search, embedding, etc.)
       |
6. Results return through relay to brain
       |
7. Brain aggregates, applies quality gating
       |
8. Edge nodes credited rUv via genesis ledger
```

### 3.3 MCP Protocol Bridge

Edge-net's `WasmMcpServer` already implements the MCP JSON-RPC protocol with tools like `vector_search`, `generate_embedding`, `credit_balance`, and `coherence_stats`. The brain exposes 22 tools via SSE MCP at `/sse`. The relay bridges these two MCP surfaces:

```
 Browser WASM Node              Relay                   Pi Brain
 +-----------------+    +-------------------+    +------------------+
 | WasmMcpServer   |    |                   |    | SSE MCP Server   |
 | (MessagePort)   |--->| WebSocket in      |    | /sse endpoint    |
 |                 |    | MCP JSON-RPC out  |--->| 22 tools         |
 | Tools:          |    |                   |    |                  |
 | - vector_search |    | Maps edge tools   |    | Tools:           |
 | - embedding     |    | to brain tools    |    | - brain_search   |
 | - lora_forward  |    | + rUv accounting  |    | - brain_share    |
 | - credit_balance|    |                   |    | - brain_sync     |
 +-----------------+    +-------------------+    +------------------+
```

## 4. Integration Points

### 4.1 Identity Mapping

Edge-net uses Pi-Key identity (Ed25519 keys, 314-bit Pi-sized). The brain uses SHAKE-256 contributor pseudonyms derived from API keys. The integration maps between them:

- Edge node's Pi-Key public key is registered with the brain as a contributor identity
- The relay derives a brain-compatible pseudonym from the Pi-Key using SHAKE-256 over the Ed25519 public key bytes
- All brain operations from edge nodes carry this derived pseudonym for attribution
- Witness chains in the brain record the Pi-Key signature alongside the pseudonym

### 4.2 Knowledge Synchronization

Edge-net's collective memory (hippocampal replay, HNSW-indexed patterns) synchronizes with the brain's knowledge graph:

| Edge-Net Component | Brain Component | Sync Direction |
|-------------------|-----------------|----------------|
| `CollectiveMemory` patterns | Brain memories (knowledge graph) | Bidirectional |
| `EntropyConsensus` decisions | Brain quality scores (Bayesian) | Edge -> Brain |
| `NetworkLearning` trajectories | Brain `LearnedPattern` entries | Edge -> Brain |
| MicroLoRA adapter weights | Brain LoRA consensus weights | Brain -> Edge |
| HNSW index state | Brain embedding space | Brain -> Edge (delta sync) |

### 4.3 Entropy Consensus for Quality Voting

Edge-net's entropy-based consensus (Shannon entropy minimization with DeGroot belief mixing) provides a distributed quality signal for brain knowledge. When multiple edge nodes encounter the same brain memory during task execution, their independent quality assessments converge via entropy consensus. The consensus result feeds back to the brain as a distributed Bayesian update, supplementing individual `brain_vote` calls with collective judgment.

### 4.4 RAC Coherence Integration

Edge-net's adversarial coherence engine (RAC, 12 axioms) protects the integrity of shared patterns. When edge nodes share patterns derived from brain knowledge, RAC ensures:

- **Axiom 6 (Disagreement is signal)**: Conflicting edge assessments of brain knowledge trigger investigation
- **Axiom 9 (Quarantine is mandatory)**: Suspicious patterns from untrusted nodes are quarantined before reaching the brain
- **Axiom 11 (Equivocation detectable)**: Merkle tree audit trail prevents nodes from submitting contradictory results

## 5. Distributed Compute Tasks

Brain operations that can be farmed out to edge-net WASM nodes:

### 5.1 Vector Similarity Search

The brain's HNSW index can be partitioned across edge nodes. Each node holds a shard of the index and executes approximate nearest-neighbor queries locally. The relay collects top-K results from multiple shards and merges them.

- **Edge capability**: `HnswIndex` in `edge-net/src/ai/memory.rs` with 150x speedup over naive search
- **WASM SIMD**: `simd128` intrinsics accelerate cosine distance on supporting browsers
- **Shard strategy**: Brain partitions index by domain/namespace; each edge node caches its assigned shard

### 5.2 Embedding Generation

The brain's `structured_hash_features()` generates lexical n-gram embeddings. The MicroLoRA transform adds learned semantic refinement. Both stages can run on edge nodes:

- **Stage 1 (lexical)**: `structured_hash_features()` is pure computation on text input, runs entirely in WASM
- **Stage 2 (LoRA transform)**: MicroLoRA forward pass with rank 1-16, `<50us` for rank-1 on edge-net's `AdapterPool`
- **Consensus weights**: Edge nodes pull the latest LoRA consensus from the brain, ensuring embedding consistency

### 5.3 MinCut Graph Partitioning

The brain's `SubpolynomialMinCut` algorithm for knowledge graph partitioning can be parallelized:

- **Partition strategy**: Each edge node processes a subgraph assigned by the brain
- **Merge**: The relay collects partial cuts and the brain computes the global minimum
- **Use case**: When the knowledge graph grows large, rebalancing partitions benefits from distributed compute

### 5.4 Federated LoRA Training

Edge-net already implements federated learning with Byzantine-tolerant gradient gossip (`GradientGossip` in `edge-net/src/ai/federated.rs`):

- **TopK sparsification**: 90% gradient compression reduces bandwidth
- **Byzantine detection**: 2-sigma outlier exclusion removes malicious gradients
- **Differential privacy**: Gaussian noise injection protects individual contributions
- **Integration**: Edge nodes train MicroLoRA weights on local task patterns; brain aggregates via reputation-weighted federated averaging

### 5.5 Quality Voting

Distributed Bayesian quality updates from edge nodes:

- Each edge node that uses a brain memory during task execution records success/failure
- Results are aggregated via entropy consensus across participating edge nodes
- The consensus vote is submitted to the brain as a weighted Bayesian update
- Brain's quality gating (auto-archive at `quality_score.mean() < 0.3` after 5 observations) benefits from higher observation counts

## 6. Security Model

### 6.1 Identity and Authentication

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| Edge node identity | Pi-Key Ed25519 (314-bit) | Per-node cryptographic identity |
| Brain contributor pseudonym | SHAKE-256 of Pi-Key public key | Brain-compatible attribution |
| Operation signing | Ed25519 signatures on task results | Non-repudiation |
| Brain witness chains | Append-only signed chain | Audit trail for all contributions |

### 6.2 Byzantine Fault Tolerance

Edge nodes are untrusted by default. The system provides multiple layers of protection:

- **Gradient poisoning**: `ByzantineDetector` in federated.rs excludes gradients beyond 2 standard deviations from the reputation-weighted mean
- **Result validation**: Brain cross-checks edge search results against local index for random samples (probabilistic verification)
- **Reputation decay**: Nodes that submit invalid results lose reputation, reducing their influence on future aggregation
- **RAC quarantine**: Patterns from low-reputation nodes enter quarantine before affecting brain state

### 6.3 Sybil Resistance

Sybil resistance uses layered defenses instead of high staking barriers, ensuring the network remains accessible to newcomers while protecting against abuse:

| Defense Layer | Mechanism | Purpose |
|---------------|-----------|---------|
| Proof of Work | First contribution requires completing 1 compute task | Proves real compute capacity; blocks zero-cost identity creation |
| Rate Limiting | 100 writes/hour per identity | Bounds the damage any single Sybil identity can cause |
| Quality Gating | Shared knowledge must pass RAC coherence check (score > 0.5) | Prevents low-quality spam from polluting the brain |
| Progressive Trust | Higher reputation tiers unlock higher rate limits and priority | Long-term good behavior is expensive to fake across many identities |
| Reputation Decay | 1% decay per epoch (see `ReputationCurve.apply_decay()`) | Abandoned Sybil identities lose influence automatically |

Staking remains available as an optional mechanism (see Section 8.3) but is not required for basic participation. Nodes that submit invalid results (as detected by probabilistic verification or Byzantine detection) have their reputation slashed, which is more effective than financial slashing alone because reputation takes sustained effort to rebuild.

### 6.4 Data Privacy

- Edge nodes receive only the data needed for their assigned subtask (need-to-know)
- Search queries are decomposed so no single edge node sees the full query context
- LoRA gradients are protected by differential privacy (Gaussian noise with configurable epsilon)
- The relay strips personally identifiable metadata before forwarding to edge nodes

## 7. API Bridge Design

Three integration options, ordered by implementation simplicity:

### Option A: Direct REST (recommended for Phase 1)

Edge-net nodes call the brain's REST API directly through the relay proxy.

```
Edge Node --[WS]--> Relay --[HTTPS]--> pi.ruv.io/v1/memories/search
                                       pi.ruv.io/v1/memories (POST)
                                       pi.ruv.io/v1/lora/latest
```

- Simplest to implement: relay proxies HTTP requests with Pi-Key authentication headers
- Each edge node's requests are independently rate-limited at the relay
- No persistent connection required between relay and brain

### Option B: Relay-Proxied Batch (recommended for Phase 2)

The relay batches requests from multiple edge nodes into bulk brain API calls.

```
Edge Nodes --[WS]--> Relay --[HTTPS bulk]--> pi.ruv.io/v1/batch
```

- Reduces brain API call count when many edge nodes query simultaneously
- Relay aggregates search queries, deduplicates, and fans results back to requesting nodes
- Requires a `/v1/batch` endpoint on the brain (new development)

### Option C: SSE MCP Bridge (recommended for Phase 3)

The relay maintains a persistent SSE MCP session with the brain and multiplexes edge node requests over it.

```
Edge Nodes --[WS/MCP]--> Relay --[SSE MCP]--> mcp.pi.ruv.io
```

- Full tool access: all 22 brain MCP tools available to edge nodes
- Persistent connection amortizes SSE setup overhead
- Relay maintains a single SSE session per brain region, multiplexing edge requests
- Requires MCP session management in the relay (handles reconnection, session affinity)

## 8. Economic Model

The economic model is designed around two principles: **accessible** (no barriers to entry) and **sustainable** (finite supply with controlled inflation). The previous model required 10-200 rUv staking to participate, creating a chicken-and-egg problem for new users. The revised model eliminates cost barriers entirely and uses contribution rewards to bootstrap the economy.

### 8.1 Free-to-Read, Earn-to-Write

All read operations are FREE. No rUv cost, no staking requirement. This removes the single biggest barrier to adoption. Write operations are also free to perform but earn rUv rewards when quality thresholds are met.

| Operation | Cost | Reward | Conditions |
|-----------|------|--------|------------|
| Search brain | FREE | 0 rUv | -- |
| Get brain status | FREE | 0 rUv | -- |
| List memories | FREE | 0 rUv | -- |
| Share knowledge | FREE | 2 rUv | Quality score > 0.5 after RAC review |
| Vote on quality | FREE | 0.1 rUv | Must have completed >= 1 task |
| Generate embedding | FREE | 1 rUv | Result passes hash verification |
| LoRA gradient | FREE | 5 rUv | Gradient passes Byzantine detection |
| WASM compute | FREE | 0.5-3 rUv | Based on compute time and success rate |

Implementation reference: rewards are credited via `WasmCreditLedger.credit()` in `examples/edge-net/src/credits/mod.rs`. The ledger's CRDT merge (`WasmCreditLedger.merge()`) ensures consistency across P2P nodes.

### 8.2 Contribution Curve (Revised)

The contribution curve rewards early adopters without being extractive. The formula is updated from the existing `ContributionCurve` in `examples/edge-net/src/credits/mod.rs`:

```
multiplier = FLOOR + (MAX_BONUS - FLOOR) * e^(-network_compute / DECAY_CONSTANT)
```

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `MAX_BONUS` | 10x | 5x | Still rewards early adopters, less extractive |
| `FLOOR_MULTIPLIER` | 1x (implicit) | 0.5x | Even mature network still pays something |
| `DECAY_CONSTANT` | 1,000,000 CPU-hours | 1,000,000 CPU-hours | Unchanged |

At genesis (0 compute-hours), multiplier is 5x. As total network compute grows, multiplier decays toward 0.5x. This means:

| Network Compute | Multiplier |
|----------------|------------|
| 0 hours (genesis) | 5.0x |
| 100K hours | 4.6x |
| 500K hours | 2.9x |
| 1M hours | 1.9x |
| 5M hours | 0.5x |

The `ContributionCurve::current_multiplier()` implementation must be updated to use the revised constants. The `FLOOR_MULTIPLIER` ensures contributors always earn something, even when the network is mature.

### 8.3 Reputation Tiers (Revised)

Reputation tiers use the existing `ReputationTier` enum from `examples/edge-net/src/economics/reputation.rs` but add a Newcomer tier and make staking optional:

| Tier | Score | Reward Mult | Staking Required | Access |
|------|-------|-------------|-----------------|--------|
| Newcomer | 0-10 | 0.5x | None | Read-only free, writes earn at 0.5x |
| Bronze | 10-25 | 1.0x | 0 rUv | Full read/write, standard rewards |
| Silver | 25-50 | 1.1x | 10 rUv (optional) | Priority task allocation |
| Gold | 50-75 | 1.25x | 50 rUv (optional) | Price discounts via `ReputationCurve.discount()`, priority |
| Platinum | 75-100 | 1.5x | 100 rUv (optional) | Max discounts, governance voting weight |

Key changes from the previous model:
- **Staking is OPTIONAL.** It provides benefits (price discounts, governance weight) but is not required for basic participation. This eliminates the chicken-and-egg problem where new users need rUv to participate but cannot earn rUv without participating.
- **Newcomer tier added.** Score 0-10 earns at 0.5x rate, providing immediate earning capability.
- **Reward multipliers** are applied via `ReputationTier::reward_multiplier()` which already implements the 1.0x/1.1x/1.25x/1.5x tiers in the codebase.

### 8.4 Halving Schedule

Brain rewards halve every 100K brain operations (cumulative across all nodes). This creates a Bitcoin-like deflationary schedule that ensures finite total rUv supply:

| Epoch | Cumulative Operations | Base Reward Multiplier |
|-------|-----------------------|----------------------|
| 0 | 0 - 100K | 1.0x |
| 1 | 100K - 200K | 0.5x |
| 2 | 200K - 400K | 0.25x |
| 3 | 400K - 800K | 0.125x |
| N | ... | 1/(2^N)x |

The halving multiplier stacks with the contribution curve and reputation tier multipliers:

```
effective_reward = base_reward * contribution_multiplier * tier_multiplier * halving_multiplier
```

This creates urgency for early participation (higher halving multiplier) without the aggressive 10x genesis bonus that the old contribution curve provided.

### 8.5 Protocol Budget

Each epoch has a fixed rUv budget to prevent runaway inflation:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial epoch budget | 1,000,000 rUv | Sufficient for ~200K reward operations at average 5 rUv |
| Budget carry-over | Yes | Unspent budget rolls to next epoch |
| Budget exhaustion behavior | Rewards pause until next epoch | Hard cap prevents inflation |
| Epoch duration | 7 days or 100K operations, whichever comes first | Bounds both time and volume |

The protocol fund is managed by `EconomicEngine.get_protocol_fund()`. The `ComputeAMM` in `examples/edge-net/src/economics/amm.rs` provides secondary revenue through trading fees (0.3%-3% dynamic fee based on pool utilization) that supplement the protocol budget after the bootstrap phase.

### 8.6 Sybil Resistance (Economic)

Instead of requiring high staking barriers (the old model required 10-200 rUv to participate), the revised model uses layered economic defenses:

| Defense | Mechanism | Implementation |
|---------|-----------|----------------|
| Proof of Work | First contribution requires completing 1 compute task | Proves real compute capacity; blocks zero-cost Sybil creation |
| Rate limiting | 100 writes/hour per identity | Bounds damage per identity; implemented at relay level |
| Quality gating | Shared knowledge must pass RAC coherence check | `AdversarialCoherence` score > 0.5 required |
| Progressive trust | Higher tiers unlock higher rate limits and priority | `ReputationCurve.select_nodes_for_task()` weights by reputation |
| Reputation cost | Reputation takes sustained effort to build (diminishing returns in `record_task()`) | More expensive to maintain many Sybil identities than one real identity |

### 8.7 Sustainability Analysis

The revised model is sustainable because total rUv supply converges:

**Finite supply proof.** With halving schedule and fixed epoch budget:
- Epoch 0: up to 1,000,000 rUv minted
- Epoch 1: up to 500,000 rUv
- Epoch N: up to 1,000,000 / 2^N rUv
- Total maximum supply = 1,000,000 * sum(1/2^N for N=0..inf) = 2,000,000 rUv

**Revenue sources by phase:**

| Phase | Duration | Primary Revenue | Secondary Revenue |
|-------|----------|----------------|-------------------|
| Bootstrap (0-10K nodes) | Months 1-6 | Protocol fund subsidy | None |
| Growth (10K-100K nodes) | Months 6-18 | Protocol fund + AMM fees | Brain API consumer payments |
| Self-sustaining (100K+ nodes) | Month 18+ | AMM fees + consumer payments | Liquidity provider fees |

**Deflationary pressure.** As brain knowledge grows, rUv utility increases (more valuable knowledge to access, more compute tasks available). The AMM's constant-product formula (`x * y = k` in `ComputeAMM`) ensures that as demand for compute grows, the rUv price of compute increases, creating natural deflationary pressure.

**Protocol fund sufficiency.** At 2,000,000 rUv maximum supply and projected 50K operations/month at maturity, the protocol fund sustains rewards for 24+ months even without secondary revenue. AMM fees (0.3%-3% of all swaps, tracked via `ComputeAMM.fees_collected`) provide a sustainable revenue stream after bootstrap.

## 9. Implementation Phases

### Phase 1: REST Bridge (Weeks 1-3)

1. Add Pi-Key to brain pseudonym derivation in the relay
2. Implement relay HTTP proxy to brain REST API with rate limiting
3. Add `brain_search` and `brain_share` proxy commands to edge-net's `WasmMcpServer`
4. Edge nodes can search brain knowledge and share patterns via relay
5. rUv accounting for search and share operations

### Phase 2: Distributed Search (Weeks 4-6)

1. Brain partitions HNSW index into shards by namespace
2. Relay distributes shard assignments to edge nodes
3. Edge nodes cache assigned shards and execute local HNSW queries
4. Relay merges top-K results from multiple edge nodes
5. Probabilistic verification: brain spot-checks 5% of edge search results

### Phase 3: Federated Intelligence (Weeks 7-10)

1. SSE MCP bridge between relay and brain
2. Edge nodes pull brain LoRA consensus weights and generate embeddings locally
3. Federated LoRA training: edge nodes contribute gradients from local task patterns
4. Entropy consensus quality voting feeds back to brain Bayesian scores
5. Collective memory synchronization between edge-net and brain knowledge graph

## 10. Monitoring

### 10.1 Integration Metrics

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Edge-to-brain request latency (p99) | Relay logs | > 500ms |
| Brain search delegation rate | Brain metrics | < 10% (not using edge) |
| Edge node participation rate | Genesis ledger | < 50 active nodes |
| Byzantine rejection rate | Relay metrics | > 5% of submissions |
| rUv earned per edge node (daily avg) | Genesis ledger | < 1 rUv (nodes leaving) |
| LoRA consensus drift | Brain LoRA metrics | Cosine distance > 0.1 from last epoch |

### 10.2 Dashboard Integration

The existing `edge-net-dashboard` (React, Cloud Run) is extended with:

- Brain integration status panel (relay connection health, brain API availability)
- Per-node brain contribution metrics (searches executed, embeddings generated, rUv earned)
- Network-wide federated learning progress (consensus convergence, gradient acceptance rate)
- Knowledge flow visualization (patterns shared edge -> brain, knowledge pulled brain -> edge)

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Edge nodes return stale search results from outdated shards | Degraded search quality | Shard version headers; brain rejects results from outdated shards |
| Relay becomes single point of failure | Edge-brain integration offline | Deploy relay in multiple regions; edge nodes failover to direct brain REST |
| LoRA gradient poisoning from compromised edge nodes | Corrupted brain consensus | Byzantine detection (2-sigma exclusion), reputation-weighted aggregation, differential privacy |
| rUv inflation from brain subsidies | Devalued rUv economy | Fixed protocol fund budget per epoch; halving schedule aligned with genesis sunset |
| High browser compute costs deter edge participation | Low node count | Battery-aware throttling, configurable CPU limits (10-50%), clear rUv earning visibility |
| Brain API rate limits block edge relay | Throttled integration | Relay batches requests; brain allowlists relay IP with higher rate limits |
| Privacy leak through search query distribution | User data exposure | Query decomposition, differential privacy on inputs, need-to-know distribution |

## 12. Acceptance Criteria

### Phase 1

- [ ] Edge node can execute `brain_search` through relay and receive results
- [ ] Edge node can execute `brain_share` to publish a pattern to brain knowledge graph
- [ ] Pi-Key identity correctly maps to brain contributor pseudonym
- [ ] rUv credited for search and share operations via genesis ledger
- [ ] Rate limiting prevents single edge node from overwhelming brain API

### Phase 2

- [ ] Brain HNSW index partitioned into at least 4 shards
- [ ] Edge nodes cache assigned shards and return search results within 200ms
- [ ] Merged results from edge nodes match single-origin results at 95% recall
- [ ] Probabilistic verification catches intentionally wrong results > 90% of the time
- [ ] Dashboard shows per-node search contribution metrics

### Phase 3

- [ ] Edge nodes pull LoRA consensus and generate embeddings consistent with brain
- [ ] Federated gradient gossip produces consensus within 5% cosine distance of centralized training
- [ ] Entropy consensus quality votes appear in brain quality scores within one epoch
- [ ] Collective memory sync operates bidirectionally without data loss
- [ ] SSE MCP bridge maintains persistent connection with automatic reconnection

### Economics

- [ ] New users can search brain without any rUv balance or staking
- [ ] Newcomer tier (score 0-10) earns rUv at 0.5x rate
- [ ] `ContributionCurve` updated with `MAX_BONUS = 5.0` and `FLOOR_MULTIPLIER = 0.5`
- [ ] Halving schedule reduces base reward multiplier after every 100K cumulative operations
- [ ] Protocol budget limits per-epoch rUv minting to 1,000,000 rUv
- [ ] Staking is optional (not required for basic participation at Newcomer or Bronze tiers)
- [ ] Rate limiting enforced at 100 writes/hour per identity
- [ ] Quality gating requires RAC coherence score > 0.5 for shared knowledge
- [ ] AMM fee revenue tracked and reported on dashboard

## 13. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-059 | Shared Brain Google Cloud -- brain deployment architecture that edge-net integrates with |
| ADR-060 | Shared Brain Capabilities -- federated MicroLoRA and knowledge substrate that edge-net extends |
| ADR-062 | Brainpedia Architecture -- knowledge pages that edge-net quality voting improves |
| ADR-063 | WASM Executable Nodes -- WASM node system that edge-net browser nodes share technology with |
| ADR-064 | Pi Brain Infrastructure -- Cloud Run deployment, custom domains, persistence layer |
| ADR-066 | SSE MCP Transport -- the MCP transport that the Phase 3 bridge connects to |
