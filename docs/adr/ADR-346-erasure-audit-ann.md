# ADR-346: Erasure-Leak Auditing for ANN Indexes — Is a Deleted Vector Actually Erased?

## Status

Inconclusive (primary hypothesis) / Accepted (audit crate, non-default).

Concretely:

- The pre-registered hypothesis — *tombstoned deletions leak the deleted vector
  to a query-only adversary, and a rebuild-based erasure closes that leak* — is
  **INCONCLUSIVE**: the baseline did not leak measurably, so the candidate had
  nothing to close (gate G0 failed).
- A **separate, unregistered finding is CONFIRMED** on three independent
  replicate seeds: `ruvector-hnsw-repair`'s `EagerRepair` — the strategy ADR-259
  recommends — *does* leak, at a held-out distinguishing accuracy of
  0.6017–0.6183 across four corpora.
- The new crate `ruvector-erasure-audit` is accepted in-tree as an audit
  instrument and as retained evidence. Its `LocalRebuild` erasure mode is
  **not promoted** to a default deletion path (gate G3 failed at 63×), and is
  recorded as the recommended procedure only for compliance-grade erasure,
  where per-operation cost is irrelevant.
- No existing crate's default behaviour changes. The sole edit outside the new
  crate is `#[derive(Clone)]` on `ruvector_hnsw_repair::HnswGraph`.

## Context

The 2026-06-18 nightly (`docs/research/nightly/2026-06-18-hnsw-delete-repair`,
ADR-259) built `ruvector-hnsw-repair`: a self-contained HNSW with three
pluggable deletion strategies — `TombstoneOnly`, `BatchRepair`, `EagerRepair` —
and evaluated them on recall@10 and latency after 20% deletion. It concluded
that repair recovers recall that tombstoning loses, and recommended repair for
workloads where deletions accumulate.

That evaluation, like the published work it surveys (hnswlib, Qdrant, Milvus,
pgvector, usearch, Vespa), measures deletion on a *utility* axis only. None of
them ask what deletion fails to remove.

Separately, ADR-345 (2026-09-05) added eviction witnesses to
`ruvector-agent-memory`, closing a gap where admission and retrieval were
witnessed but deletion was not. That work certifies *that* an eviction
happened. It does not ask whether the eviction erased anything.

This ADR's question: **when a vector is deleted from an HNSW index, can an
adversary with ordinary query access still determine that it was once there?**
That is the GDPR Art. 17 verification question, and inverted, it is a
membership-inference attack against a deleted record.

## Hypothesis

```text
Given a seeded synthetic corpus of 2,000 x 32-dim clustered vectors indexed by
ruvector-hnsw-repair's HnswGraph (m=8, m0=16, ef_construction=40), and a
black-box adversary limited to ordinary queries (choose ef, read back
(id, distance)) probing with a held target vector,

when leakage is measured as a paired distinguishing advantage over 600 trials
per configuration — arm A inserts the target and erases it, arm B inserts an
unrelated decoy at the same sequence position and erases it identically, both
take the same post-erasure churn, and the adversary's feature and sign are
selected on a split-half and scored on the untouched 300-pair half —

then (G0) the Tombstone baseline shows a held-out distinguishing accuracy of at
least 0.60, and (G1) the candidate LocalRebuild erasure holds it at or below
0.55,

subject to (G2) LocalRebuild's recall@10 after 20% deletion being within 2.0
percentage points of EagerRepair's, (G3) its mean erasure latency being at most
5x EagerRepair's, and (G4) zero bytes of the erased vector remaining resident.

If G0 fails the run is INCONCLUSIVE regardless of the other gates: there was no
leak to close.
```

Full methodology, raw output, the feasibility probe, the declared setup
correction and the confirmatory replication are in
`docs/research/nightly/2026-09-22-erasure-audit-ann/README.md`.

## Decision

1. Add `crates/ruvector-erasure-audit`: a black-box erasure-leak audit for ANN
   deletion procedures. It comprises a seeded clustered-corpus generator with
   no `rand` dependency (`data`), three erasure modes (`erasure`), a
   six-feature black-box adversary and a split-half paired distinguisher
   (`audit`), a chained erasure-audit certificate (`certificate`), and a
   harness shared verbatim by the headline and replication binaries
   (`harness`). Three binaries: `erasure-audit-bench`, `erasure-replicate`,
   `erasure-probe`.
2. Add `ErasureMode::LocalRebuild`, which removes dangling edges, re-derives
   each affected neighbour list by an ordinary ef-search over the *surviving*
   graph, and zeroizes the victim's stored coordinates. **Do not promote it to
   a default deletion strategy** and do not alter `ruvector-hnsw-repair`'s
   `DeletionStrategy` trait or its existing implementations.
3. Add `#[derive(Clone)]` to `ruvector_hnsw_repair::HnswGraph`. This is the
   only change outside the new crate; it is additive, changes no behaviour, and
   exists so a prepared base index can be forked per trial rather than rebuilt
   2,400 times per configuration.
4. **Record the `EagerRepair` leak as a confirmed finding** and amend the
   practical guidance that ADR-259 gives, without changing ADR-259's code or
   its recall conclusions. The two results are on orthogonal axes and both
   stand.
5. Keep the crate in-tree as an audit instrument and as retained evidence, per
   the nightly process's evidence-retention rule. It is not wired into CI by
   this ADR.

## Evidence

Raw output is in the nightly README; summarised here.

### Acceptance gates

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| G0 baseline (`Tombstone`, churn=0) leaks | >= 0.60 | 0.5250, CI [0.469, 0.581] | **FAIL** |
| G1 candidate (`LocalRebuild`) indistinguishable | <= 0.55 | 0.5167, CI [0.460, 0.573] | PASS |
| G2 recall@10 loss vs `EagerRepair` | <= 2.0 pp | 0.47 pp | PASS |
| G3 mean erase latency vs `EagerRepair` | <= 5.0x | 63.26x | **FAIL** |
| G4 retained vector bytes | == 0 | 0 | PASS |

G0's failure alone forces INCONCLUSIVE under the rule fixed in advance. G3
fails independently; even had G0 passed, the verdict for a general deletion
path would have been REJECT.

### The confirmed secondary finding

`EagerRepair` held-out distinguishing accuracy at churn=0, four independent
corpora:

| Run | Accuracy | 95% CI | Leak |
|---|---|---|---|
| headline | 0.6150 | [0.559, 0.668] | yes |
| replicate 1 | 0.6017 | [0.545, 0.655] | yes |
| replicate 2 | 0.6100 | [0.554, 0.663] | yes |
| replicate 3 | 0.6183 | [0.562, 0.671] | yes |

`Tombstone` over the same four runs: 0.5250 / 0.5100 / 0.5133 / 0.5150, none
significant. `LocalRebuild`: 0.5167 / 0.4767 / 0.5133 / 0.5250, none
significant.

The replication used a confirmation hypothesis fixed *before* those three seed
offsets were drawn, because the finding was discovered by inspection rather
than tested for. Discovery and confirmation are kept separate deliberately.

The advantage decays with post-erasure churn but had not reached chance at 200
inserts on n=2,000: 0.6150 (churn=0) -> 0.5933 (50) -> 0.5567 (200).

### Mechanism, measured directly

`erasure-probe` counts neighbour lists differing between arms A and B versus
how often the adversary's observation differs at all:

| Mode | lists differing | observation differs |
|---|---|---|
| `Tombstone` | 12.20 | 7.5% |
| `EagerRepair` | 11.45 | 35.0% |
| `LocalRebuild` | 9.35 | 27.5% |

All three perturb a comparable number of lists; only the repair modes'
perturbation reaches the query API. `EagerRepair` fills the hole left by the
victim using *the victim's own neighbour list* as the candidate pool, so it
removes one pointer to the deleted vector and writes a new edge chosen by that
vector's coordinates. A tombstone's dangling edge is skipped before any
distance is computed, so it is structurally present and observationally inert.

`LocalRebuild` perturbs the graph too and still does not leak: its edges depend
only on surviving vectors, so they differ from the control in a direction
uncorrelated with the target.

### Utility and cost (20% deletion, identical victim ids)

| Mode | recall@10 before | after | delta | erase mean | erase p95 | retained payload |
|---|---|---|---|---|---|---|
| `Tombstone` | 0.9173 | 0.8933 | -2.40 pp | 0.0 us | 0.1 us | 50.0 KiB |
| `EagerRepair` | 0.9173 | 0.9097 | -0.77 pp | 15.4 us | 36.9 us | 50.0 KiB |
| `LocalRebuild` | 0.9173 | 0.9050 | -1.23 pp | 972.6 us | 3217.3 us | 0 |

The recall picture reproduces ADR-259's direction. Post-erasure query p95 is
within noise across modes (71.6 / 78.5 / 76.3 us).

### Certificate chain

9 records appended, `verify()` clean, exhaustive per-record tamper sweep
detected 9/9; the unit-test sweep detects 25/25.

### Declared setup correction

The first full run returned paired accuracy of *exactly* 0.5000 everywhere —
every pair tied bit-for-bit. The feasibility probe diagnosed the cause: the
originally chosen corpus (16 clusters, sigma=0.18) had recall@10 = 0.219 *at
ef=256*, i.e. ground-truth neighbours were effectively ties and the index was
non-functional. Two changes followed, both declared in the nightly README:
the corpus moved to 64 clusters / sigma=0.60 (recall@10 = 0.927, matching
ADR-259's 0.914 regime), and the trial count rose from 150 to 600 because the
probe measured a 60-95% tie rate. **No acceptance threshold was changed at any
point.**

## Consequences

- ADR-259's recall conclusions stand unchanged. Its *practical recommendation*
  gains a caveat: `EagerRepair` is the only measured mode that leaks, so it is
  the wrong choice when the index holds erasable data and its query API is
  reachable by untrusted callers.
- The workspace gains a reusable instrument for asking "did this deletion
  erase anything?" of any mutation-on-delete index. The method is not
  HNSW-specific.
- `ruvector-erasure-audit` is also, by construction, a membership-inference
  harness. It requires paired ground truth to produce a number, so it is a
  measurement tool rather than a turnkey attack, but its feature set is exactly
  what an attacker would compute. This is intentional and is stated in the
  crate's module documentation.
- No default behaviour changes anywhere. `ruvector-hnsw-repair`'s three
  strategies, trait and tests are untouched beyond the added `Clone`.
- The tombstone null is a bounded claim, not a safety proof, and is recorded as
  such. If a stronger adversary overturns it, the headline picture inverts.

## Alternatives Considered

- **A white-box leakage measure** ("does any live node still reference the
  victim?"). Rejected as the headline metric: it is trivially 100% for
  `Tombstone` and 0% for both repair modes, and it measures structural residue,
  which the probe shows is not the same as adversary-observable residue — the
  mode with the *most* structural residue leaks the least.
- **Training a classifier over all six features** rather than ranking on one.
  Rejected for tonight: it needs a train/test split inside each configuration
  and an ML dependency, and the paired-ranking statistic is interpretable
  without either. It would make the adversary stronger, and is the first
  next-research item precisely because it could falsify the tombstone null.
- **Full index rebuild as the candidate erasure.** Rejected: it is the
  known-correct answer and needs no experiment.
- **Cover traffic** (rewiring random unaffected nodes to mask the repair
  region). Not implemented: this adversary queries at the victim's own
  location, so hiding which region was touched does not help it. Worth
  revisiting against an adversary that must first locate the erasure.
- **Promoting `LocalRebuild` as the default delete.** Rejected on measured
  evidence: 63x `EagerRepair`'s cost with a 6.1x tail ratio, against a leak
  that the default baseline does not exhibit.
- **Moving G3's denominator after seeing 63x.** Rejected outright. The gate was
  mis-specified — it is fair for "should this be the default delete" and
  meaningless for "is this a viable offline erasure procedure" — and that
  mis-specification is recorded in the nightly README's "What I would
  pre-register differently" rather than fixed retroactively.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-erasure-audit/src/{lib,data,erasure,audit,certificate,harness}.rs`
- `crates/ruvector-erasure-audit/src/bin/{erasure_audit_bench,erasure_replicate,erasure_probe}.rs`
- `crates/ruvector-hnsw-repair/src/graph.rs`: `#[derive(Clone)]` on `HnswGraph`
- `Cargo.toml`: workspace member entry
- 25 unit tests across the five library modules

No further implementation is planned under this ADR. Follow-up scope is in the
nightly README's "Next research" and is explicitly out of this ADR.

## API Shape

```rust
pub enum ErasureMode {
    Tombstone,
    EagerRepair,
    LocalRebuild { ef_rebuild: usize },
}
pub fn erase(graph: &mut HnswGraph, id: usize, mode: ErasureMode) -> ErasureStats;

pub struct ProbeConfig { pub k: usize, pub ef_lo: usize, pub ef_hi: usize,
                         pub jitter_trials: usize, pub jitter_sigma: f32,
                         pub jitter_seed: u64 }
pub fn observe(graph: &HnswGraph, q: &[f32], cfg: &ProbeConfig) -> Features;

pub struct PairedDistinguisher { /* .. */ }
impl PairedDistinguisher {
    pub fn push(&mut self, a: Features, b: Features);
    pub fn evaluate(&self) -> DistinguisherResult; // split-half selection
}

pub struct CertificateChain { /* .. */ }
impl CertificateChain {
    pub fn append(&mut self, subject_id: &str, mode: &'static str, referrers: u32,
                  rebuilt_lists: u32, retained_vector_bytes: u32,
                  audit_advantage_milli: u32, timestamp_ns: u64) -> &ErasureCertificate;
    pub fn verify(&self) -> Result<(), ChainError>;
    pub fn head(&self) -> u64;
}
```

## Feature Flags

None. The crate has a single dependency (`ruvector-hnsw-repair`) and no
optional features. `ruvector-hnsw-repair` gains no feature flag either — the
added `Clone` is unconditional.

## Benchmark Evidence

```bash
cargo run --release -p ruvector-erasure-audit --bin erasure-audit-bench   # ~65s
cargo run --release -p ruvector-erasure-audit --bin erasure-replicate     # ~8s
cargo run --release -p ruvector-erasure-audit --bin erasure-probe         # ~10s
cargo test -p ruvector-erasure-audit --release                            # 25 tests
```

Acceptance gates are compiled into `erasure-audit-bench` as `GATE_*` constants
and the binary prints PASS/FAIL and the verdict itself, so the call cannot
drift from the numbers. See "Evidence" above and the nightly README for the
full raw output.

## Security

No new cryptographic primitive. `CertificateChain` uses keyless FNV-1a,
matching `ruvector-agent-memory::ops` (ADR-134), with the same documented
scope: it detects naive edits to a stored log and is **not**
adversary-resistant, since anyone who can edit a record can recompute the
hashes. Tail truncation is not detectable internally — a unit test asserts this
limitation explicitly — which is why `head()` exists to be anchored externally
per ADR-342. Subject identifiers are stored only as `fnv1a(subject_id)`,
because a log of who was erased is itself personal data. Upgrading to the
signed scheme of `ruvector-retrieval-receipt` (ADR-340) requires no change to
the record layout.

The crate performs no I/O, opens no sockets, spawns no threads, and contains no
`unsafe`.

## Governance

None introduced. An erasure path built on this would inherit the
"no witness, no mutation" invariant of ADR-345 / the proof-gated-writes line;
that integration is not implemented here.

## Failure Modes

See the nightly README's "Failure modes" for the full account. In brief: the
first corpus was degenerate and was caught only because its result was
implausibly clean; `LocalRebuild` can thin a neighbour list on sparse upper
layers and is guarded by keeping the larger of the two candidate lists; its p95
is 6.1x its median, so any synchronous erasure path needs a queue; zeroization
removes the payload from the live index but says nothing about allocator,
page-cache, snapshot or swap copies; and the tombstone null bounds *this*
adversary at *this* power rather than proving absence.

## Migration

None. The new crate is additive; `ruvector-hnsw-repair` gains a derive that
cannot break an existing caller.

## Rollback

Remove `crates/ruvector-erasure-audit` and its workspace member entry. The
`#[derive(Clone)]` on `HnswGraph` may be left or reverted independently; no
other crate depends on either change.

## Rejection Criteria

The primary hypothesis is treated as INCONCLUSIVE rather than rejected because
its precondition failed: G0 measured 0.5250 against a >= 0.60 threshold, so the
leak the candidate was built to close was never demonstrated and G1 could not
be a meaningful test. `LocalRebuild` is separately not promoted because G3
measured 63.26x against a <= 5.0x threshold.

## Open Questions

1. Does the `Tombstone` null survive a stronger adversary — a trained
   classifier over all six features, a multi-target correlation attack, or a
   timing channel? This is the claim most likely to fall.
2. Is the `EagerRepair` leak caused specifically by drawing replacement
   candidates from the victim's neighbour list? A variant that removes dangling
   edges but selects replacements from the referrer's own ef-search would
   isolate this.
3. How much post-erasure churn is needed to reach chance, as a function of `n`?
   Measured only to 200 inserts on n=2,000, where the advantage was still
   0.5567.
4. Do these results hold on real embeddings rather than synthetic
   Gaussian-cluster data? Nothing here should be treated as guidance until they
   do.
5. Are other index families leakier? A PQ codebook retrained on a corpus that
   included the victim, or an IVF list rewritten on delete, are plausibly much
   larger leak surfaces than a single graph edge, and are unmeasured.
