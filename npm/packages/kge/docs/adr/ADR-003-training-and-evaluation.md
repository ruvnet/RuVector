# ADR-003: Training and evaluation

## Status
Proposed

## Date
2026-09-21

## Context

The scorer (ADR-002) is only as good as how it is trained and how it is
measured. The literature is unusually clear on both:

- **Training strategy dominates model choice** (Ruffinelli et al., ICLR 2020,
  LibKGE): under a common well-tuned protocol, a 2011-era model matches later
  architectures trained naively. So the loss / negative sampling / regularizer /
  optimizer choices matter more than picking HolE vs RotatE (this drives
  ADR-004's HPO weighting).
- **Evaluation has a silent trap**: filtered MRR/Hits@k depend on how tied
  scores are broken. Sun et al. 2020 (arXiv:1911.03903) showed many older papers
  used TOP (optimistic) tie-breaking, inflating scores — worst on WN18RR, whose
  symmetric relations produce large tied blocks. RANDOM is now the LibKGE/PyKEEN
  default.

`ruvector-gnn` already provides the training primitives: `Optimizer`/
`OptimizerType::Adam` and `Loss`/`LossType` (training.rs), `hadamard_product`
and tensor ops (tensor.rs), and `ElasticWeightConsolidation` (ewc.rs).

## Decision

### Losses and negative sampling

1. **Two supported regimes, both from the LibKGE-preferred set:**
   - **Self-adversarial negative sampling** (RotatE, arXiv:1902.10197):
     negatives weighted by the model's own current score via softmax — the de
     facto standard for margin-based scorers.
   - **1-vs-all cross-entropy** (ComplEx-N3, Lacroix et al. 2018,
     arXiv:1806.07297): score every entity as a candidate tail, no sampling
     variance — the preferred setup for ComplEx/HolE.
   The choice is a bandit arm (ADR-004), not hardcoded.

2. **N3 (nuclear 3-norm) regularization** (Lacroix et al. 2018): elementwise, no
   extra matrix ops, cheap to port, with consistent small gains on
   WN18RR/FB15k-237/YAGO3-10. Combined with reciprocal-relation reformulation it
   lets the base scorer reach ComplEx-N3 numbers. Plain L2 is the fallback.

3. **Optimizers: Adam and Adagrad**, reusing `ruvector-gnn`'s `Optimizer`
   (`OptimizerType::Adam`, training.rs). LibKGE supports both; the schedule is an
   HPO arm.

### What runs where

4. **Training is native-Rust only** in v1. Batching is structured for CPU: the
   entity/relation matrices are dense tensors updated per mini-batch, scored as
   matmuls (ADR-002 §4). **WASM ships inference/scoring only.** A toy-scale
   (<10k entities) WASM training path for interactive demos is a stretch goal
   behind the same FFT-crate spike as ADR-002, not a v1 commitment — the
   research brief found no evidence either way on WASM training (a stated gap).

### Evaluation protocol

5. **Filtered MRR and Hits@1/3/10 with RANDOM tie-breaking, CI-asserted.** The
   evaluator must never default to argmax or stable-sort, which silently
   reproduces TOP-biased inflation. The CI assertion is concrete: a scorer that
   returns **identical** scores for all candidates must yield a mean rank
   ≈ `(|E|+1)/2` across seeds (not 1 = TOP, not `|E|` = BOTTOM), seeded for
   reproducibility. This is the Sun et al. 2020 fix as a test, not a comment.

6. **Frozen, instance-disjoint splits.** One content-hashed split per dataset
   (train / validation / transfer / test), never regenerated between arms;
   `assert_train_eval_disjoint` (`crates/sona/src/darwin_guard.rs`, reused from
   typesafe ADR-004/006) runs on every proposal. Test is scored only for the
   baseline and the final champion of a campaign.

7. **The HolE gate number is ComplEx's LibKGE number, by the equivalence.**
   LibKGE has no HolE entry; because HolE≡ComplEx (asserted by the ADR-002 unit
   test), the reproduced ComplEx numbers are the correct target for HolE. This
   dependency is why the equivalence test is load-bearing.

### Datasets

| Dataset | ~size | Licence / source | Used for |
|---|---|---|---|
| FB15k-237 | 310k triples, 14,541 ent, 237 rel | Freebase-derived; follows source, fetched at bench time, nothing redistributed | dense-graph link prediction |
| WN18RR | 93k triples, ~41k ent, 11 rel | WordNet-derived; same stance | sparse/symmetric tie-break stress |
| CoDEx-M | 185k triples, 17,050 ent, 51 rel | Wikidata-derived, CC-BY (verify exact text at fetch); ships hard negatives | mid-scale gate + adversarial set |
| YAGO3-10 | ~1M triples, 123k ent, 37 rel | YAGO-derived; same stance | scale/throughput |

Licences are named only where the brief cites them (CoDEx CC-BY, to be
confirmed against the source repo); the rest follow their source and nothing is
redistributed, mirroring typesafe ADR-006.

## Consequences

- Reaching literature-competitive MRR needs a proper training regime, not a
  one-shot fit; `kge train` and ADR-004's loop are first-class, and zero-tuning
  parity is never implied.
- RANDOM tie-breaking makes this crate's numbers directly comparable to
  LibKGE/PyKEEN and *lower* than TOP-inflated papers — that is correct, and the
  README says so.
- Training native-only means the WASM package is a scoring client of a table
  trained elsewhere; the table is a versioned artifact with a lineage link
  (ADR-004, `rvf-manifest`).

## Alternatives considered

- **Margin-ranking loss only.** Simpler, but LibKGE shows 1-vs-all
  cross-entropy is stronger for ComplEx/HolE; both are kept as arms.
- **L2 regularization only.** Works, but N3 is nearly free and consistently
  better; N3 is the default, L2 the fallback.
- **Ship HolE-specific published numbers as gates.** Impossible: HolE's original
  numbers were an optimization artifact (arXiv:1707.01475) and LibKGE dropped
  the model. ComplEx-by-equivalence is the honest target.

## Evidence

- Losses/negatives/regularizer: RotatE arXiv:1902.10197; ComplEx-N3
  arXiv:1806.07297; LibKGE (Ruffinelli et al., ICLR 2020). Reproduced ComplEx
  MRR: FB15k-237 0.348, WN18RR 0.475, YAGO3-10 0.551 (LibKGE README).
- Tie-breaking: Sun et al. 2020, arXiv:1911.03903.
- Datasets: FB15k-237, WN18RR (standard splits); CoDEx arXiv:2009.07810.
- Repo reuse: `crates/ruvector-gnn/src/{training.rs,tensor.rs,ewc.rs}`;
  `crates/sona/src/darwin_guard.rs`.
- Unmeasured: CoDEx-M has no baseline MRR in the brief — read from
  arXiv:2009.07810's table at harness build (ADR-006); WASM training viability
  is a stated gap.
