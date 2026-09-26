# Repairing an HNSW Graph After a Delete Is What Leaks the Deleted Vector

Deletion in a vector index is usually a tombstone: flip a bit, skip the node at
query time, move on. The graph is never repaired. Everyone knows this costs
recall, and there is a body of work on fixing it — including a deletion-repair
crate in this repository that implements tombstone-only, batched and eager
repair, and concludes that repair is worth its cost.

That evaluation, like every one I could find, measures deletion on a utility
axis: recall and latency. None of them ask the other question. When you delete
a vector, is it *gone* — or can someone who can only send queries still tell it
was there?

I built an audit to measure that, and the answer inverted my expectation.

## The measurement

Leakage is a distinguishing advantage. For each trial I build two indexes that
differ in exactly one event:

- **A**: base corpus, then target `v` is inserted, then deleted.
- **B**: base corpus, then an unrelated decoy `w` is inserted at the same
  position in the insert sequence, then deleted the same way.

Both then take the same post-deletion churn. An adversary who holds `v` and has
nothing but query access — pick `ef`, read back `(id, distance)` — probes both
with `q = v` and computes six scalar statistics: the sum of top-k distances at
low `ef`, the nearest distance at high `ef`, the gap between them, the Jaccard
instability between the low- and high-effort result sets, the churn under
jittered repeats, and the result-count deficit.

Accuracy is the fraction of pairs where a chosen statistic ranks A above B,
ties counting half. 0.5 is indistinguishable; 1.0 is a perfect oracle. The
feature *and its sign* are picked on the first half of the trials and scored on
the untouched second half, so the headline number is not a best-of-six artefact.
600 trials per configuration, 300 held out, Wilson 95% intervals. A leak is
declared only when the interval's lower bound clears 0.50.

Three erasure modes, on a 2,000 × 32 seeded clustered corpus with `m=8`,
`m0=16`, `ef_construction=40`:

| Mode | What it does |
|---|---|
| `Tombstone` | mark deleted; graph untouched |
| `EagerRepair` | remove dangling edges, splice in the victim's own neighbours as replacements |
| `LocalRebuild` | remove dangling edges, re-derive each affected list by ef-search over the surviving graph, zeroize the payload |

## The result

```text
mode           churn            feature   hold_acc            95% CI  leak?
Tombstone          0            dsum_lo     0.5250  [0.469, 0.581]     no
EagerRepair        0         effort_gap     0.6150  [0.559, 0.668]    YES
LocalRebuild       0         effort_gap     0.5167  [0.460, 0.573]     no
Tombstone         50            dsum_lo     0.5200  [0.464, 0.576]     no
EagerRepair       50         effort_gap     0.5933  [0.537, 0.647]    YES
LocalRebuild      50   topk_instability     0.4917  [0.436, 0.548]     no
Tombstone        200            dsum_lo     0.4950  [0.439, 0.551]     no
EagerRepair      200            dsum_lo     0.5567  [0.500, 0.612]    YES
LocalRebuild     200         effort_gap     0.4883  [0.432, 0.545]     no
```

Tombstoning does not leak. Repairing does.

Because I found that by reading the rest of the table rather than by testing
for it, I fixed a confirmation hypothesis and re-ran on three fresh seeds — new
corpus, new targets, new decoys:

```text
 replicate mode                  hold_acc            95% CI  leak?
         1 EagerRepair             0.6017  [0.545, 0.655]    YES
         2 EagerRepair             0.6100  [0.554, 0.663]    YES
         3 EagerRepair             0.6183  [0.562, 0.671]    YES

EagerRepair leaked in   3/3 replicates
LocalRebuild clean in   3/3 replicates
Tombstone leaked in     0/3 replicates
```

Four independent corpora, `EagerRepair` at 0.6150 / 0.6017 / 0.6100 / 0.6183 —
a 1.7-point spread.

## Why

The repair heuristic fills the hole left by the victim using the victim's own
neighbour list as the candidate pool. It removes one pointer to the deleted
vector and, in the same motion, writes a new edge whose endpoint was *chosen by
the deleted vector's coordinates*. The victim is erased from the result set and
re-encoded into the topology.

A tombstone writes nothing. Its dangling edge is real, but search skips a
deleted neighbour *before computing a distance to it*, so it never influences
traversal order, candidate admission, or any returned value. It is structurally
present and observationally inert.

A direct probe confirms this. Counting how many neighbour lists differ between
arms A and B, versus how often the adversary's feature vector differs at all:

```text
 clusters     sigma           mode     lists_diff    obs_diff_rate
       64      0.60      Tombstone          12.20            0.075
       64      0.60    EagerRepair          11.45            0.350
       64      0.60   LocalRebuild           9.35            0.275
```

All three modes perturb about the same number of lists. Only the repair modes'
perturbation reaches the API. Structural residue and observable residue are
different things, and the field has been reasoning about the first.

`LocalRebuild` perturbs the graph too — 27.5% observable difference rate — and
still does not leak, because its edges are derived only from vectors that are
still present, so they differ from the control in a direction uncorrelated with
the target. That is what indistinguishability looks like when it is not the
same as "changes nothing".

## The cost

```text
mode           recall_b  recall_a  delta_pp  del_us_mean  del_us_p95   rtn_KiB
Tombstone        0.9173    0.8933     -2.40          0.0         0.1      50.0
EagerRepair      0.9173    0.9097     -0.77         15.4        36.9      50.0
LocalRebuild     0.9173    0.9050     -1.23        972.6      3217.3       0.0
```

20% deletion, all three modes erasing the identical victim id set. The
recall picture matches the prior work: tombstoning costs 2.4 points,
repair costs 0.8. `LocalRebuild` lands between them at 1.2 points, and is the
only mode that leaves zero bytes of the deleted payload resident — but it costs
973 µs per erasure against `EagerRepair`'s 15.4, with a 6.1× tail ratio.

## What I had pre-registered, and what it says

The hypothesis was: tombstones leak, and a rebuild-based erasure closes the
leak. Five gates, fixed before the first run and compiled into the benchmark
binary:

| Gate | Threshold | Measured | |
|---|---|---|---|
| G0 baseline (`Tombstone`) leaks | ≥ 0.60 | 0.5250 | **FAIL** |
| G1 candidate indistinguishable | ≤ 0.55 | 0.5167 | PASS |
| G2 recall loss vs `EagerRepair` | ≤ 2.0 pp | 0.47 pp | PASS |
| G3 erase latency vs `EagerRepair` | ≤ 5.0× | 63.26× | **FAIL** |
| G4 retained payload bytes | == 0 | 0 | PASS |

G0 was a precondition: if the baseline does not leak there is nothing to close,
and the run is INCONCLUSIVE whatever else happens. That is the verdict. G1's
pass is unearned — the candidate closed a leak that was not there. G3 fails
independently, so `LocalRebuild` is not a default delete path.

G3 was also, in hindsight, the wrong denominator. A compliance-erasure path
runs once per subject request, offline, where 973 µs is free. Normalising
against `EagerRepair` made it a gate for "should this be the default delete",
not for "is this a viable erasure procedure". I left the threshold alone and
recorded the mis-specification instead.

## A correction worth stating

The first full run returned paired accuracy of *exactly* 0.5000 on every
feature and every mode. Every A/B pair tied bit-for-bit. That is a diagnosis,
not a null result, so I wrote a probe:

```text
 clusters     sigma    recall@10 recall_ef256
       16      0.18       0.2190       0.2200
       64      0.60       0.9270       0.9780
```

My original corpus (16 tight clusters) had recall@10 of 0.22 *at ef=256* — the
ground-truth top-10 were effectively ties and the index was non-functional. No
deletion is observable in a cloud of 125 near-identical vectors. I changed the
corpus to the regime that reproduces the prior work's 0.914 recall, and raised
the trial count from 150 to 600 because the probe showed a 60–95% tie rate
makes 75 held-out pairs too few to resolve a 0.60 gate from a 0.55 one.

The corpus and the sample size changed. No acceptance threshold did.

## What this does and does not establish

It establishes, on four independent synthetic corpora, that the repair
heuristic from the DPG line of work — reconnect a deleted node's neighbours to
each other — creates a query-observable trace of the deleted vector, and that
deriving the replacement edges from surviving vectors instead removes that
trace at a 63× cost.

It does not establish that tombstoning is safe. The null means "not detectable
by this adversary, with these six features, at n=300 held out". A trained
classifier over all six features, a multi-target correlation attack, or a
timing channel could all overturn it, and the first of those is cheap enough
that it is the obvious next experiment. It also says nothing about an adversary
with memory access, for whom a tombstoned vector is simply sitting there — 50
KiB of it, in this run.

Everything is synthetic Gaussian-cluster data. Before any of this becomes
guidance it needs to run on real embeddings.

## Practical upshot

If your index is a hot store, deletions are frequent, and you have no erasure
obligation, the prior guidance is unchanged: repair.

If your index holds data you may be required to erase, and its query API is
reachable by the people whose data it is, then the repair strategy is the one
mode measured to leak — and the intuition that "repair is strictly better than
tombstoning" is wrong on an axis the recall benchmark cannot see.

---

Code: `crates/ruvector-erasure-audit` (new crate, 3 binaries, 25 tests).
Full methodology, raw output and failure modes:
`docs/research/nightly/2026-09-22-erasure-audit-ann/README.md`. Decision
record: `docs/adr/ADR-346-erasure-audit-ann.md`.

```bash
cargo run --release -p ruvector-erasure-audit --bin erasure-audit-bench
cargo run --release -p ruvector-erasure-audit --bin erasure-replicate
cargo run --release -p ruvector-erasure-audit --bin erasure-probe
```
