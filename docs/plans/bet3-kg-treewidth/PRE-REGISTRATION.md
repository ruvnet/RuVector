# BET 3 — Treewidth probe of curated knowledge graphs (PRE-REGISTRATION, FROZEN)

**Status:** frozen before any data touched the probe. Issue #534. Branch
`docs/seprag-bet3-kg-treewidth` (cut off PR #535, where `ruvector-seprag` lives).
Follows the mandatory "prove-not-hype" protocol
(`docs/plans/seprag-cch-retrieval/FUTURE-DIRECTIONS.md`).

## The bet (one sentence)

The salvaged separator-tree branch-and-bound k-NN kernel is **validated and
treewidth-independent at query time**, but CCH *contraction/build* blew up on
embedding and citation backbones because they are **high-treewidth** (ADR-199:
elim-tree height ≈ 0.5·n). The bet: a **curated, bounded-degree knowledge graph**
(Wikidata-style — the backbone underlying HotpotQA / 2WikiMultiHop / MuSiQue) may
finally be **low-treewidth enough** for contraction/build to stay cheap, making the
kernel usable on a real multi-hop retrieval backbone.

## Why this is a probe-first go/no-go (not a benchmark yet)

ADR-199 established that small separators are an **intrinsic structural prerequisite**:
better separator heuristics or lower degree *cannot* manufacture small separators that
do not exist. So the only thing worth measuring first — cheaply — is whether the KG's
treewidth is road-like. A high-treewidth result is a **clean, reportable NO-GO** (ADR-199
/ ADR-201 precedent) and the bet stops there, with no QA-harness investment.

**The QA benchmark is NOT built unless this gate returns GO.**

## Primary metric (one number)

The **fitted elimination-tree-height scaling exponent `p`** in `elim_h ∝ n^p`, where
`elim_h = max over vertices of elim_depth` from `SepRag::build_with(Balanced)`, measured
across a scale sweep and fit by ordinary least squares on `log(elim_h)` vs `log(n)`.

- Road-like / low-treewidth: `elim_h ≈ 3.5·√n` ⟹ **p ≈ 0.5**.
- Expander / high-treewidth: `elim_h ≈ 0.5·n` ⟹ **p ≈ 1.0**.

`p` cleanly separates the two regimes (ADR-199 anchors: roadNet-PA p≈0.5, ogbn-arxiv
citation p≈1.0). Reported **alongside** the absolute blowup ratio `|G+|/|G_nav|` at the
largest n — judged absolute, never the ratio alone (ADR-199 hub-dampening backfire).

## Scale sweep & subgraph extraction

- n ∈ **{2 000, 5 000, 10 000, 20 000}** (capped at the backbone's size if smaller).
- Subgraph = connected **BFS-ball** from a fixed seed over the largest connected
  component, unit edge weights (hop distance) — identical to `m1_arxiv.rs`. Same seed
  policy across backbones for comparability.

## Probe set (bracketed by calibrated controls)

| Backbone | Role | Prior / expectation |
|---|---|---|
| **roadNet-PA** (on disk) | 🟢 calibration control | known GOOD: 7.6×, p≈0.5. **Must reproduce or run is VOID.** |
| **ogbn-arxiv citation** (on disk) | 🔴 reference | known BAD: 23.8×, p≈1.0 |
| **WN18RR** (WordNet) | KG under test — sparse, hierarchical | bet's best case |
| **FB15k-237** (Freebase subset) | KG under test — hub-heavy | adversarial stress |
| **2WikiMultiHopQA Wikidata evidence graph** | KG under test — faithful to eventual benchmark | most decision-relevant |

All loaded as undirected unit-weight **entity graphs** (relation labels dropped for the
structural backbone; the bet is about graph structure, not edge types).

## Frozen gate

Decided on the **largest n reached** for each KG, with the exponent `p` fit across the
sweep, and **only valid if the road control reproduces ~√n / ~7.6× in the same run**.

- **GO** → design the multi-hop QA benchmark (separate, agreed step):
  `p ≤ 0.6` **AND** blowup ratio ≤ ~10× at largest n.
- **NO-GO / KILL** → write ADR-203 finding, stop:
  `p ≥ 0.8` **OR** blowup ratio ≥ 23× (citation-like).
- **INCONCLUSIVE** (`0.6 < p < 0.8`): extend the sweep or try a sparser KG variant;
  do **not** proceed to the QA harness on an inconclusive result.
- **VOID**: road control fails to reproduce ~√n / ~7.6× → probe miscalibrated, fix
  harness and rerun.

## Adversarial check (protocol rule #5) — the methodological upgrade over ADR-199

ADR-199's NO-GO rested only on the **upper** bound (elim_h from one separator heuristic).
A NO-GO could in principle be a weak-heuristic artifact, and a GO could be an
over-optimistic separator. So for every backbone we additionally:

1. **Treewidth LOWER bound** via **minor-min-width (MMD/MMW)**: repeatedly remove a
   minimum-degree vertex, contract it into its minimum-degree neighbour, track the max
   min-degree seen. This is a standard, cheap treewidth lower bound. A NO-GO is
   **structurally certain** only if even this lower bound grows ≈linearly (not merely the
   upper bound). A GO requires the lower bound to also be small.
2. **Both separators** — run `Balanced` and `BfsLayer` so the heuristic's contribution to
   elim_h is visible (ADR-199 showed Balanced ≫ BfsLayer on low-diameter graphs).

If the upper bound is linear but the lower bound is small, the verdict is **INCONCLUSIVE
(heuristic-limited)**, not KILL.

## Secondary diagnostics (reported, not gated)

- Build time; separator-size distribution (top-level separator size); degree
  distribution / max degree (to characterise the "bounded degree" claim); sampled
  Dijkstra-oracle recall@10 sanity (confirm pipeline validity on the new backbone, as
  `m1_arxiv.rs` does — correctness is expected to hold everywhere, as in ADR-199).

## Honesty commitments

- Report the KG result **bracketed between** the two controls in the **same run** — never
  in isolation.
- Judge absolute `|G+|` and elim_h, never the blowup ratio alone.
- A NO-GO is a legitimate, reportable third finding (1 WIN / 2 KILLS → possibly 3).
- If 2WikiMultiHop KG acquisition proves impractical to obtain cleanly, WN18RR (sparse)
  + FB15k-237 (hub-heavy) already bracket the KG question; 2Wiki is reported as a
  follow-up rather than silently dropped.
