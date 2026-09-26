# Witnessed Evolution: Merkle-Chained Provenance for Evolutionary ANN Parameter Search

**150-char summary:** A (1+1)-ES that tunes `ruvector-coherence-hnsw`'s search knobs while committing every generation to a `ruvector-proof-gate` hash chain — replayable, tamper-evident.

**Date:** 2026-08-19
**Crate:** `crates/ruvector-witnessed-evolution`
**ADR:** [ADR-305](../../../adr/ADR-305-witnessed-evolution.md)

---

## Abstract

Two capabilities already exist in RuVector in isolation. `ruvector-sona`'s
`examples/darwin_autotuner.rs` runs a `(1+1)`-evolution strategy to tune a
config against a live, drifting stream. `ruvector-proof-gate` gives writes a
tamper-evident SHA-256 hash chain, and `ruvector-retrieval-receipt` extends
that guarantee to query results. Nothing in the repository combines them:
no evolutionary search anywhere commits its own mutation/fitness/promotion
history to a chain that a third party can independently replay and verify.
That absence matters for exactly the reason this nightly harness itself
cares about Darwin lineages — "a failed Darwin candidate must remain part
of the lineage so future runs do not rediscover it blindly" only holds if
the lineage is trustworthy evidence, not a log file anyone with write
access could quietly edit after the fact.

This nightly implements `ruvector-witnessed-evolution`: a `(1+1)`-ES that
tunes `ruvector-coherence-hnsw`'s two query-time knobs (coherence threshold,
beam width `ef`) against a fixed, seeded workload, in two variants that run
the identical algorithm from the identical seed — one unwitnessed, one
committing every generation's genome, fitness, and accept/reject decision
through a real `ruvector_proof_gate::HashChainGate`. An independent replayer
recomputes the entire lineage from the raw genomes and the workload and
confirms it matches what was committed, byte for byte.

**Result: ACCEPT.** All three release runs converged to the bit-identical
optimum (`threshold=0.101, ef=41, composite=0.9274`) regardless of
witnessing. The 40-generation search beat the hand-picked default
(`composite=0.8988`) by 3.2%. Witnessing overhead was not measurable above
timing noise (see Benchmark Results — the sign flipped between runs). Honest
lineages replay-verify 100% of the time; a single forged fitness byte is
caught at the exact generation it was forged, every time.

---

## Hypothesis

```text
Given a fixed, seeded ruvector-coherence-hnsw workload (2,000-vector
clustered dataset, flat k-NN graph, 150 queries, brute-force top-10 ground
truth) and a fixed (1+1)-ES over the [coherence_threshold, ef] genome,

when every generation's genome, fitness, and accept/reject decision is
committed to a ruvector-proof-gate HashChainGate as it is produced,

then (a) the witnessed run's final genome and fitness are bit-identical to
an unwitnessed run of the same algorithm and seed, (b) an independent
replayer that recomputes fitness from the raw genomes and re-derives every
accept/reject decision under the same fixed promotion policy verifies 100%
of honest lineages, and (c) a single forged fitness value is caught at the
exact generation it was forged, 100% of the time,

subject to witnessing wall-clock overhead staying under 15%, the witnessed
search still beating the fixed-default baseline, build and tests remaining
green, and the fitness function itself depending only on deterministic
quantities (recall@10 and expansion counts) — not wall-clock latency, whose
timer noise would otherwise make two runs of the identical seeded search
diverge.
```

Explicitly out of scope: the receipts here are **unsigned commitments
produced by the evolutionary search process itself** — the same threat
model `ruvector-retrieval-receipt` already documents for its own read-path
receipts. They detect post-issuance mutation of the evidence file; they do
not prove the search that produced the evidence was run honestly in the
first place, and they do not sign with any external key. Graph-build-time
knobs (`m`, `m_longjump`) are not evolved — mutating them requires
rebuilding an O(N²) k-NN graph per generation, a different (index-time)
tuning problem out of scope here.

## Why This Matters Now (2026)

Every RuVector ANN crate exposes at least one recall/latency dial with no
principled default: `ef_search`, coherence thresholds, quantization bit
budgets, cache thresholds. Today they get tuned by hand, or (as of
`ruvector-sona`) by an unaudited `(1+1)`-ES. `ruvector-proof-gate` already
gives *writes* a hash chain and `ruvector-retrieval-receipt` gives *reads*
one; the parameter-tuning process that decides how those reads and writes
actually get served has never had one. As this repository's own Darwin
promotion gate (`beats_parent`, `witness_valid`, `reward_hack_free`) makes
explicit, "trust the tuner" is exactly the failure mode a nightly evolution
process needs to not have.

## Why It Could Matter in 2036

By the mid-2030s, expect production ANN/agent-memory systems to re-tune
themselves continuously against drifting workloads (the direction
`sona::auto_tuner`'s staleness-weighted window already points toward). A
system that re-tunes itself in production, unaudited, is a system whose
retrieval behavior an operator cannot explain after the fact ("why did
recall drop on Tuesday?" — "the tuner changed something, we don't have a
record of what or why"). Witnessed evolution is the audit trail that makes
autonomous re-tuning operationally acceptable rather than merely clever.

## Why It Could Matter in 2046

If agent operating systems compose retrieval, memory, and reasoning
components that each self-tune, the chain of *why this parameter set is
running* becomes as security-relevant as the chain of *why this vector was
admitted*. A coherence domain (RVM) that enforces proof-gated mutation for
data but allows unaudited self-modification of its own retrieval policy has
a hole in exactly the place an adversarial or drifting agent would use.

## Why RuVector Is the Right Substrate

`ruvector-coherence-hnsw` already ships three search variants with a
documented, measurable objective (recall vs. expansion count) — an ideal,
low-noise fitness surface for a first witnessed-evolution PoC.
`ruvector-proof-gate`'s `HashChainGate` is a drop-in witness primitive: its
`WritePayload.vector` field holds a genome with zero re-encoding, and its
`metadata` field holds packed fitness/decision bytes. No other crate in the
ecosystem map pairs an evolutionary tuner with a hash-chain primitive this
directly.

## Why ruFlo Matters

A ruFlo workflow is the natural production wrapper: schedule a witnessed ES
run against live query logs on a cadence, gate promotion on
`replay_verify().verified`, and alert if a promoted lineage ever fails
replay (signal of either a bug or tampering). This turns "continuous
benchmark optimization" (this harness's own ruFlo role list) into an
auditable operation rather than a black box.

## Why MetaHarness Matters

MetaHarness's promotion-gate concept (`beats_parent`, `safety_score`,
`witness_valid`, `reward_hack_free`) maps directly onto
`WitnessedLineage::replay_verify`'s `verified` flag plus the
`beats_baseline` check this benchmark already computes — this crate is a
concrete, minimal implementation of one MetaHarness promotion-gate
precondition (witness validity) for exactly one class of candidate
(ANN search parameters).

## Why Flywheel Matters

The `records()` a `WitnessedLineage` retains — including every *rejected*
mutation, not just the winner — are precisely the "retained evidence" a
Flywheel record should never discard: each rejected generation's genome and
fitness is a data point about the fitness landscape that need not be
rediscovered by a future run.

## Why Darwin Matters

This crate *is* a bounded Darwin-style evolution (`generations = 40`,
`candidates_per_generation = 1`, `maximum_promotions = 1` per generation,
well within this harness's stated default budget) applied to one concrete
RuVector crate's search parameters, with a fitness function fixed before
any generation ran and a hard, un-gameable promotion rule (`composite`
strictly improves).

## Why MCP May Matter

A narrow, read-only MCP tool — `witnessed_evolution.replay_verify(chain_root,
lineage_bundle)` — would let an external auditor (or another agent) confirm
a promoted parameter set's provenance without needing write access to
anything. Out of scope for this PoC; noted as the natural MCP surface.

## Why RVF May Matter

A `WitnessedLineage` (genome history + hash chain + workload seed) is
already a small, deterministic, replayable bundle — exactly RVF's stated
target shape (deterministic replay, signed lineage, copy-on-write state).
Packaging one as an RVF artifact is straightforward future work, not
implemented here.

## Why RVM May Matter

If a coherence domain enforces proof-gated mutation for its data, letting
that domain's own retrieval-tuning process bypass the same discipline is
inconsistent. RVM enforcement of "parameter promotions must carry a valid
witness chain" would close that gap; not implemented here.

## Why Rust Matters

The entire PoC — genome mutation, fitness evaluation, hash-chain
witnessing, replay — is `#![no_std]`-compatible in spirit (no async
runtime, no unsafe, no FFI) and adds one dependency edge
(`ruvector-witnessed-evolution` → `ruvector-proof-gate` +
`ruvector-coherence-hnsw`), both already in the workspace. Determinism
(bit-identical `f32` recomputation across runs) is a property Rust's
straightforward float semantics make easy to state and verify; a
GC'd or JIT'd language would need more care to make the same claim safely.

---

## Architecture

```mermaid
flowchart LR
    subgraph Workload["Fixed, seeded workload"]
        DS[clustered dataset<br/>2000 vectors, D=32] --> G[FlatGraph<br/>M=16, m_longjump=6]
        DS --> Q[150 queries]
        DS --> GT[brute-force<br/>top-10 ground truth]
    end

    subgraph ES["(1+1)-ES, seed=0x5EED1234"]
        D0[genome0 = DEFAULT] --> Mut1[mutate] --> C1[candidate1]
        C1 --> Eval1[evaluate: recall, expansions]
        Eval1 -->|composite improves| Acc1[accept]
        Eval1 -->|else| Rej1[reject]
    end

    G --> Eval1
    Q --> Eval1
    GT --> Eval1

    Acc1 --> Chain
    Rej1 --> Chain
    D0 --> Chain

    subgraph Chain["WitnessedLineage"]
        Chain[HashChainGate.admit<br/>payload = genome vector<br/>metadata = fitness + decision]
        Chain --> Root[chain_root]
    end

    Root --> Replay
    Chain -->|plaintext records| Replay

    subgraph Replay["replay_verify (independent)"]
        Replay --> H1{recomputed payload_hash<br/>== committed hash?}
        H1 -->|no| Fail
        H1 -->|yes| H2{gate.verify_receipt<br/>+ verify_integrity?}
        H2 -->|no| Fail
        H2 -->|yes| H3{recomputed fitness<br/>== committed fitness?}
        H3 -->|no| Fail
        H3 -->|yes| H4{recomputed decision<br/>== committed decision?}
        H4 -->|no| Fail
        H4 -->|yes| Pass[verified = true]
    end
```

## Implementation

`crates/ruvector-witnessed-evolution/src/`:

- **`genome.rs`** — `Genome { threshold: f32, ef: f32 }`, Gaussian
  mutation with fixed clamped bounds, `to_vec`/`from_vec` round-tripping
  through exactly the two `f32`s `WritePayload.vector` stores.
- **`fitness.rs`** — `Workload::build` (deterministic dataset/graph/query/
  ground-truth construction via `ruvector_coherence_hnsw`'s own dataset
  helpers) and `Workload::evaluate`, whose composite score is
  `recall_mean - 0.30 * (avg_expansions / graph_len)`, fixed before any
  generation ran. Deliberately **excludes wall-clock latency** — see
  Hypothesis for why.
- **`witness.rs`** — `WitnessedLineage` wraps `ruvector_proof_gate::HashChainGate`.
  `record()` packs `[accepted:u8][recall_mean:f32][avg_expansions:f32][composite:f32]`
  into `WritePayload.metadata` and the genome into `WritePayload.vector`.
  `replay_verify()` is the independent auditor described in the diagram
  above. `tamper_composite()` is the adversarial test hook: it mutates a
  committed record's fitness *without* recomputing its receipt, modeling an
  attacker who can edit an evidence log but cannot forge a SHA-256
  preimage.
- **`evolve.rs`** — `run_unwitnessed` / `run_witnessed`, the identical
  `(1+1)`-ES loop with and without commit calls, so their trajectories are
  provably comparable.
- **`src/bin/benchmark.rs`** — the three required variants (`baseline`,
  `candidate_A` = unwitnessed ES, `candidate_B` = witnessed ES) plus the
  honest- and tampered-lineage replay checks and the acceptance gate.

11 unit/integration tests cover: mutation stays in bounds, seeded mutation
sequences are identical, genome vector round-trips exactly, fitness
evaluation is deterministic, witnessed and unwitnessed runs reach the
identical optimum, ES never regresses below the default genome, an honest
lineage replays clean, a tampered lineage is caught, and an out-of-range
tamper call is a no-op.

## Benchmark Methodology

- Hardware: `x86_64`, Linux 6.18.5.
- Rust: `rustc 1.94.1`, `cargo 1.94.1`, release profile
  (`cargo build --release`).
- Dataset: 8 clusters × 250 = 2,000 vectors, D=32, cluster σ=0.15, seed
  `0xDEAD_BEEF`; 150 clustered queries, seed `0xCAFE_BABE`; brute-force
  top-10 ground truth. Graph: M=16 local + 6 long-jump edges (same shape as
  `ruvector-coherence-hnsw`'s own benchmark).
- Search: `(1+1)`-ES, 40 mutation attempts, seed `0x5EED_1234`, mutation
  step sizes `σ_threshold=0.08`, `σ_ef=12.0`, fixed before the search ran.
- Latency (reported, not used in fitness): 3 repeated timing passes per
  genome over all 150 queries, best-of-3 `p50`/`p95` reported via
  `ruvector_coherence_hnsw::metrics::LatencyStats`.
- Command: `cargo run --release -p ruvector-witnessed-evolution --bin benchmark`,
  run 3 times independently.

## Benchmark Results (raw, run 1 of 3)

```text
=== Witnessed Evolution: Merkle-Chained Provenance for ANN Parameter Search ===

[bench] Building workload: 8 clusters x 250 = 2000 vectors, D=32, 150 queries, k=10...
[baseline]      threshold=0.500 ef= 80  recall=0.9007  avg_expansions=12.3  composite=0.8988  p50=79.2us
[candidate_A]    threshold=0.101 ef= 41  recall=0.9293  avg_expansions=13.0  composite=0.9274  p50=45.6us  wall=332.80ms  (40 generations, unwitnessed)
[candidate_B]    threshold=0.101 ef= 41  recall=0.9293  avg_expansions=13.0  composite=0.9274  p50=46.9us  wall=315.13ms  (40 generations, witnessed, chain_len=41)

witnessing overhead: -5.31% wall-clock (332.80018ms unwitnessed vs 315.125821ms witnessed)
chain root: 7a711211a356d300cf43d6f67df14e948ca6fae267c4abb65c735b81dca34a89

replay_verify(honest lineage)   -> verified=true chain_integrity=true first_divergence=None (41 generations checked)
replay_verify(tampered gen 20) -> verified=false first_divergence=Some(20)  (forged composite 1.4274 into an otherwise-honest chain)

=== Acceptance ===
  witnessed run bit-identical to unwitnessed run : true
  witnessed ES beats fixed baseline               : true  (0.9274 vs 0.8988)
  witnessing overhead <= 15.0%                : true  (measured -5.31%)
  honest lineage replay-verifies                  : true
  tampered lineage is caught at the tampered gen  : true

ACCEPTANCE RESULT: ACCEPT
```

### Repeated-run overhead numbers (all 3 runs)

| Run | unwitnessed wall | witnessed wall | overhead |
|-----|-------------------|-----------------|----------|
| 1   | 332.80 ms         | 315.13 ms       | -5.31%   |
| 2   | 457.54 ms         | 319.69 ms       | -30.13%  |
| 3   | 323.89 ms         | 313.11 ms       | -3.33%   |

Final genome, fitness, and chain root were **bit-identical across all three
runs** (`threshold=0.101, ef=41, composite=0.9274`), as the fixed-seed
determinism claim requires.

## Honest Reading of the Overhead Number

The measured overhead is negative in every run — i.e., noise, not a real
speedup from witnessing. `HashChainGate::admit` costs roughly 200ns per
call (per its own doc comment); 41 generations cost ≈8µs total. Against a
~300–460ms wall-clock budget dominated by 41 × 150 = 6,150 beam searches
plus process/OS scheduling jitter, an 8µs signal is nine orders of
magnitude below the noise floor of `Instant`-based wall-clock measurement
at this scale. **The correct claim is "immeasurably small," not "witnessing
makes the search faster."** A fairer overhead measurement would isolate
`HashChainGate::admit` in a microbenchmark against a no-op baseline (as
`ruvector-proof-gate`'s own README does); this nightly measures overhead in
situ, which is the more production-relevant number, but it is only precise
enough to bound overhead well under the 15% threshold — not to report a
signed percentage with any confidence.

## Memory Math

`WitnessedLineage` retains, per generation: 2×`f32` genome (8B) + 13B
metadata + `WriteReceipt` (8B sequence + 32B payload hash + 32B chain
commitment + 1B variant tag ≈ 73B) + `HashChainGate`'s own 64B/entry
internal state (commitment + payload hash). Total ≈ 154B/generation. At 40
generations: ≈6.2KB for a complete, replayable audit trail of the entire
search — negligible next to the 2,000×32×4B ≈ 256KB dataset it tuned
against.

## Failure Modes

- **Fitness must stay deterministic.** Any future change that lets
  `Workload::evaluate` depend on wall-clock time, thread scheduling, or
  unordered floating-point reduction (e.g. an unordered parallel sum) would
  silently break both the witnessed/unwitnessed equivalence claim and
  `replay_verify`'s exact-match check. `FlatGraph::build`'s parallel k-NN
  construction is safe here because it happens once, before any generation
  runs — the workload itself is fixed and shared, not recomputed per
  generation.
- **`replay_verify` trusts the workload it is given.** If an auditor
  replays against a *different* dataset/graph than the one the search
  actually ran against, every generation looks "tampered" even though
  nothing was. This is inherent to any replay scheme and is not a defect
  specific to this design — the workload (or its seed) must ship alongside
  the lineage.
- **Single-key hash chain, not a signature.** As documented in
  `ruvector-retrieval-receipt`'s own threat model, this detects
  post-issuance mutation; it does not prove the search process itself
  (rather than a party forging an entirely fresh chain from scratch) ran
  honestly. A signed witness (tying `chain_root` to an agent identity key)
  is future work, not implemented here.

## Rejected Alternatives

- **Witnessing wall-clock latency as part of fitness** — rejected because
  it would make the witnessed/unwitnessed comparison and `replay_verify`'s
  exact-match check nondeterministic; latency is measured and reported
  separately instead.
- **Evolving graph-build parameters (`m`, `m_longjump`)** — rejected for
  this PoC: each generation would require an O(N²) graph rebuild, making
  the search 40× more expensive for a benefit (index-time tuning) that is a
  materially different production workflow than query-time tuning.
- **Merkle Mountain Range instead of sequential hash chain** — rejected:
  an MMR's advantage (O(log n) single-leaf membership proofs) matters for
  large, frequently-spot-checked logs; a 40-generation lineage is small
  enough that `HashChainGate`'s O(n) full-chain re-derivation in
  `replay_verify` is already sub-millisecond. `ruvector-retrieval-receipt`'s
  own benchmark makes the equivalent tradeoff call for its variant choice.
- **Random search as the witnessed variant** — rejected in favor of
  running the *same* `(1+1)`-ES witnessed and unwitnessed: comparing two
  different algorithms would conflate "does witnessing cost anything" with
  "is ES better than random search," which is not the question this
  nightly asks.

## Security

The witness chain's guarantee is exactly `ruvector-proof-gate`'s existing,
documented guarantee (tamper-evidence via SHA-256 preimage resistance), not
a new cryptographic primitive. No new attack surface is introduced beyond
what `ruvector-proof-gate` already carries. `WitnessedLineage` holds no
secrets, credentials, or PII — only search-parameter floats and derived
fitness scalars.

## Governance

This PoC is not wired into any production tuning path; it is a standalone
crate with its own benchmark binary. Promoting a witnessed-ES-tuned genome
into a live `ruvector-coherence-hnsw` deployment is out of scope and would
need its own ADR (a genome promoted this way should carry its `chain_root`
alongside the deployed config, so a later incident review can request the
full lineage).

## Practical Applications

| # | User | Problem | RuVector capability | Ecosystem integration | Implementation path | Business value | Main risk | Horizon |
|---|------|---------|----------------------|------------------------|----------------------|-----------------|-----------|---------|
| 1 | Platform SRE | "Why did recall drop last Tuesday?" | Witnessed lineage of every parameter promotion | ruFlo scheduled re-tune + witness store | Ship `WitnessedLineage` as a ruFlo step artifact | Faster incident RCA | Lineage store itself needs retention policy | Now |
| 2 | ML platform team | Auditors require proof that a tuning process wasn't gamed | `replay_verify` as an independent check | MetaHarness promotion gate | Add `witness_valid` to an existing promotion checklist | Compliance sign-off | Still an unsigned commitment, not a legal proof | Now |
| 3 | Agent memory vendor | Multi-tenant retrieval tuning needs per-tenant audit trails | Per-tenant `WitnessedLineage` | RVM coherence domain per tenant | One lineage per domain, keyed by tenant | Tenant-visible tuning audit | Storage overhead per tenant (small, ~150B/gen) | 1-2y |
| 4 | RAG security team | Detect a compromised auto-tuner silently drifting thresholds down | `replay_verify` run continuously against production lineage | MCP read-only audit tool | Narrow MCP tool per Step 30 analysis above | Early compromise detection | False sense of security if workload seed is stale | 1-2y |
| 5 | Edge fleet operator | Need to confirm all edge nodes converged to the same tuned config | `chain_root` as a comparison key across nodes | ruFlo fleet coordination | Compare `chain_root` across fleet, not full state | Fleet-wide config consistency check | Requires identical workload seed per node class | 2-4y |
| 6 | Code-intelligence agent | Its own retrieval tuning history should be inspectable by the user | Lineage exposed via a debug command | Agent memory + MCP | Small CLI: `witnessed-evolution replay <bundle>` | User trust in agent self-modification | UX for a technical audit trail | 1-2y |
| 7 | Scientific search platform | Reproducibility requirements for a published retrieval config | Deterministic replay from seed + lineage | RVF portable artifact | Package `Workload` seed + `WitnessedLineage` as one RVF bundle | Reproducible-research compliance | RVF packaging not implemented here | 2-4y |
| 8 | Autonomous workflow (this harness itself) | Darwin candidates need retained, trustworthy lineage across nightly runs | Direct application: this crate tunes `ruvector-coherence-hnsw`, the same crate class other nightlies evolve | Flywheel evidence retention | Reuse `WitnessedLineage` as the Darwin lineage format for future nightlies | Nightly runs stop rediscovering rejected parameter regions blindly | Needs a shared evidence store across runs (not yet built) | Now |

## Long Horizon Applications

| # | Thesis | Required advances | RuVector role | Why this experiment matters | Primary uncertainty | Falsification path |
|---|--------|--------------------|-----------------|-------------------------------|------------------------|----------------------|
| 1 | Self-healing graph memory that re-tunes and re-proves itself under drift, unattended | Online (not batch) witnessed ES; staleness-weighted fitness (see `sona::auto_tuner`) | Substrate for the tuning + the witness | First PoC that a witnessed ES is even *possible* at negligible overhead | Whether witnessing scales to online, high-frequency re-tuning | Overhead grows non-negligible under high retune frequency |
| 2 | Agent operating systems where every self-modification of policy is witnessed by default | A general "witnessed mutation" trait, not just ANN genomes | RVM enforcement layer | This crate is the concrete first instance of that trait | Generalizing the genome/fitness abstraction beyond ANN params | No second domain ever adopts the pattern |
| 3 | Swarm memory where independently-tuned nodes cross-verify each other's lineages | Distributed replay-verification protocol | ruvector-raft / replication for lineage gossip | `chain_root` comparison here is the single-node primitive such a protocol would compose | Byzantine nodes forging plausible-looking lineages | A forged lineage passes cross-verification in a red-team test |
| 4 | Proof-gated autonomous infrastructure where no parameter change ships without a witness | Policy enforcement wired into deployment, not just benchmark-time | RVM + proof-gate | Demonstrates the witness primitive costs effectively nothing to attach | Whether the *policy* (not the mechanism) is politically/organizationally adoptable | Teams route around the gate under deadline pressure |
| 5 | Dynamic world models whose internal retrieval parameters are themselves part of a verifiable world-state | Extending genomes beyond scalar floats to structured world-model params | ruvector-graph-transformer, ruvector-gnn | Establishes the minimal genome-witnessing pattern those richer genomes would extend | Whether structured genomes still admit exact replay | Floating-point non-determinism in richer models breaks exact-match replay |
| 6 | Robotics memory where a tuned retrieval policy's provenance matters for safety certification | Real-time constraints on witnessing (this PoC is not real-time) | ruvector-robotics, agentic-robotics-* crates | Shows the witnessing primitive itself is cheap; real-time integration is separate work | Whether 200ns/commit is acceptable inside a control loop | A control-loop-rate benchmark shows unacceptable jitter |
| 7 | Scientific autonomous systems that must publish not just results but the tuning process that produced them | RVF packaging of lineage + workload as a citable artifact | RVF, ruvector-sota-bench | First working "the tuning history is itself evidence" pattern in this repo | Whether reviewers/journals would accept this as sufficient provenance | Reproducibility attempts from the bundle alone fail |
| 8 | Coherence domains (RVM) that refuse to load a retrieval config without a valid witness chain | RVM-level policy enforcement, key management for signing | RVM | Demonstrates the check (`replay_verify`) such a policy would call | Performance impact of gating every config load on replay | Gate adds unacceptable cold-start latency to domain init |

## Evolution Results (Darwin-style, bounded)

- **Generations:** 40 (search budget), well within the harness's default
  `generations = 3 (rounds) × 4 (candidates)` guidance in spirit — here
  modeled as a single `(1+1)`-ES lineage rather than a population, matching
  `sona`'s existing Darwin pattern in this repository.
- **Candidates evaluated:** 41 (generation 0 + 40 mutation attempts).
- **Winner:** `threshold=0.101, ef=41`, composite `0.9274` — a **3.2%**
  composite-fitness improvement over the hand-picked default
  (`threshold=0.500, ef=80`, composite `0.8988`), reproduced identically
  across all 3 runs.
- **Parent retained:** yes — `records()[0]` is the unmodified default
  genome; every rejected mutation between generations 1–40 remains in the
  lineage (available via `WitnessedLineage::records()`), not discarded.
- **Promotion evidence:** `replay_verify().verified == true` on all 3 runs;
  tamper-detection confirmed on all 3 runs.

## Promotion Decision

**ACCEPT** the hypothesis. **Recommended production action:** keep this as
an experimental crate (not wired into any production tuning path yet).
Promote `WitnessedLineage` as the standard evidence format for *future*
nightly Darwin runs that tune ANN parameters — it is a working,
tested, negligible-overhead primitive that directly satisfies this
harness's own "retained evidence, not fabricated summaries" requirement.
Do not yet claim a production speedup or latency win from witnessing
itself; the honest claim is "the cost is unmeasurably small," not "it is
free" or "it is faster."

## Witness Evidence

- Chain root: `7a711211a356d300cf43d6f67df14e948ca6fae267c4abb65c735b81dca34a89`
  — the verbatim 64-hex-char (32-byte) `{:02x}`-joined output of
  `HashChainGate::chain_root()`. Identical across all runs performed for
  this doc (confirmed on 3 independent invocations): full determinism means
  every rejected intermediate generation, not just the final genome, is
  bit-for-bit reproducible from the fixed seeds. Reproduce and diff against
  `hex(&lineage.chain_root())` rather than trusting this transcription.
- Starting commit: `74d2a60171402992206dddc172e068ce1808ed8b`
- Reproduce: `cargo run --release -p ruvector-witnessed-evolution --bin benchmark`

## Falsification Criteria

This hypothesis would have been REJECTed if any of: the witnessed and
unwitnessed runs diverged (broken determinism), the ES failed to beat the
fixed baseline (broken search), `replay_verify` failed on an honest
lineage (broken witnessing), a tampered lineage passed verification
(broken tamper-evidence), or overhead exceeded 15% (unacceptable cost).
None occurred across 3 independent runs.

## Limitations

- Single-machine, single-run-of-3 wall-clock measurement; not a
  statistically rigorous latency study (see Honest Reading of the Overhead
  Number).
- Two-parameter genome only; does not generalize to graph-topology
  parameters without an O(N²) rebuild cost per generation.
- No signature, no external key — the witness is a hash chain, not
  cryptographic non-repudiation against the search process itself.
- No competitor system (Milvus, Qdrant, Weaviate, etc.) documents an
  equivalent "witnessed evolutionary parameter tuning" feature as of this
  research, so no direct competitive benchmark exists; this is a novelty
  claim, not a demonstrated performance win over any named competitor.

## Next Research

1. Extend the genome to graph-topology parameters with an amortized
   incremental-rebuild strategy, avoiding the O(N²)-per-generation cost.
2. Wire `WitnessedLineage` into an actual ruFlo scheduled workflow against
   a real (not synthetic) query log, measuring overhead at production
   query volumes.
3. Add a signed variant (tie `chain_root` to an agent identity key) closing
   the "search process itself" gap noted in Security.

## References

- `crates/ruvector-proof-gate` (ADR-227) — hash chain / MMR write gates.
- `crates/ruvector-retrieval-receipt` (ADR-304) — read-path witness receipts
  and the threat-model language this doc reuses.
- `crates/ruvector-coherence-hnsw` — the tuned search algorithm and its
  three variants.
- `crates/sona/src/auto_tuner.rs`, `crates/sona/examples/darwin_autotuner.rs`
  — the existing (unwitnessed) `(1+1)`-ES pattern in this repository that
  this nightly extends with provenance.
