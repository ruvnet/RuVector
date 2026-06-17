---
adr: 203
title: "Treewidth Probe of Curated Knowledge Graphs (SepRAG BET 3)"
status: accepted
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-197, ADR-199, ADR-200, ADR-201, ADR-202]
tags: [ruvector, seprag, cch, treewidth, knowledge-graph, wikidata, wordnet, freebase, graph-rag, no-go]
---

# ADR-203 — Treewidth Probe of Curated Knowledge Graphs (SepRAG BET 3)

## Status

**Accepted — go/no-go gate executed; verdict NO-GO across all three curated KGs
(2026-06-04).** This is the last untested backbone for the salvaged separator-tree
branch-and-bound k-NN kernel (`ruvector-seprag`), which is validated and
*treewidth-independent at query time* but whose CCH **contraction/build** blew up on
high-treewidth embedding and citation graphs ([ADR-199]). BET 3 tested whether a
**curated, bounded-degree knowledge graph** is low-treewidth enough for contraction to
stay cheap. It is not. The thread scoreboard is now **1 WIN, 3 KILLS**.

Pre-registration (frozen before any data touched the probe):
`docs/plans/bet3-kg-treewidth/PRE-REGISTRATION.md`. Harness:
`crates/ruvector-seprag/examples/kg_treewidth_probe.rs`.

## Context

ADR-199 killed CCH full contraction on embedding/citation backbones: they are
high-treewidth (elimination-tree height ≈ 0.5·n), so contraction fill-in explodes even
though the separator-tree *query* prunes ~100 % of search space and stays exact. The one
backbone left untested was the design's true intended scope ([ADR-196]): a **sparse,
structured, relational** knowledge graph — Wikidata-style, the backbone underlying
multi-hop QA (HotpotQA / 2WikiMultiHop / MuSiQue). The hope: bounded relation degree
might give road-network-like small separators where dense similarity graphs cannot.

This was sequenced **probe-first** under the "prove-not-hype" protocol: measure treewidth
cheaply as a go/no-go gate *before* building any multi-hop QA benchmark. A high-treewidth
result is a clean, reportable NO-GO ([ADR-199]/[ADR-201] precedent) and the bet stops.

## Decision

Probe real curated KGs with the validated `ruvector-seprag` kernel and the existing
road-control harness, bracketing every KG between two calibrated anchors **in the same
run**, and apply a frozen gate.

### Method

- **Primary metric:** the fitted **elimination-tree-height scaling exponent `p`**
  (`elim_h ∝ n^p`), OLS on a `{2k,5k,10k,20k}` scale sweep of connected BFS-ball
  subgraphs. Road-like ⟹ p≈0.5; expander ⟹ p≈1.0. Reported **alongside** the absolute
  blowup ratio `|G+|/|G_nav|` read at the ADR-199-matched reference n (~2k), judged
  absolute (never the ratio alone — the ADR-199 hub-dampening lesson).
- **Calibration anchors (same run):** roadNet-PA (known GOOD; must reproduce ~√n / ~7.6×
  or the run is VOID) and ogbn-arxiv citation (known BAD).
- **Curated KGs under test:** WN18RR (WordNet — sparse, hierarchical), FB15k-237
  (Freebase — hub-heavy), CoDEx-L (genuine Wikidata — the faithful representative of the
  multi-hop-QA backbone).
- **Adversarial upgrade over ADR-199 (protocol rule #5):** a **minor-min-width treewidth
  LOWER bound** per backbone, plus both separators (Balanced + BfsLayer). A lower bound
  proves treewidth *cannot* be small, so a NO-GO is structurally certain rather than a
  weak-separator-heuristic artifact.
- **Adaptive build budget:** a high-treewidth contraction explodes super-linearly
  (fill-in ≈ tw²·n), so a slow build at moderate n is itself the NO-GO signal; the sweep
  caps once a build exceeds the budget instead of running away.

### Frozen gate

GO: `p ≤ 0.6` **AND** blowup ≤ 10× at reference n. KILL: `p ≥ 0.8` **OR** blowup ≥ 23×.
INCONCLUSIVE between. VOID if the road control fails. The QA benchmark is **not** built
unless GO.

## Empirical Outcome (2026-06-04)

Recall is exact (30/30 vs the Dijkstra oracle) on **every** backbone — the query kernel
is correct everywhere, as in ADR-199. The cost is contraction/build, governed by
treewidth.

| Backbone | role | avg deg | max deg | p (exp) | blowup @ref(2k) | elim_h / n | tw **lower** bound | verdict |
|---|---|---|---|---|---|---|---|---|
| **roadNet-PA** | control | 2.6 | **9** | 0.613 | **7.2×** | 0.04 | **tw ≥ 4** | calibrated ✓ |
| ogbn-arxiv citation | reference | 6.9 | 4999 | 1.259 | 17.4× | 0.28 | tw ≥ 44 | NO-GO (ADR-199) |
| **WN18RR** (WordNet) | KG | 3.7 | 520 | 0.508 | **59.9×** | 0.21 | tw ≥ 28 | **NO-GO** |
| **FB15k-237** (Freebase) | KG | 9.2 | 1999 | — | **42.3×** | **0.46** | tw ≥ 42 | **NO-GO** |
| **CoDEx-L** (Wikidata) | KG | 3.4 | 4999 | **1.826** | 5.1× | 0.17 | tw ≥ 28 | **NO-GO** |

(Control VALID: road reproduced p=0.613, 7.2×@2k ≈ √n, tw ≥ 4 — textbook. Build times:
road 2.2 s @ n=20k; arxiv 43 s @ n=5k; KG builds explode at n≤5–10k — the super-linear
contraction signal. FB15k-237 capped at a single point because its n=2k build already
exceeded budget; its verdict rests on absolute blowup + the tw lower bound, not the
exponent.)

### Findings

1. **All three curated KGs are high-treewidth — NO-GO, structurally certain.** The
   minor-min-width **lower** bounds (28–44) are **7–11× the road control's (4)** and on a
   par with the citation reference (44). A lower bound is a *guarantee* treewidth is at
   least that large, so the verdict does not depend on the separator heuristic — it closes
   the one gap ADR-199's upper-bound-only argument left open.

2. **Three distinct failure signatures; the conjunction gate caught each.** No single
   metric would have sufficed:
   - **WN18RR** — deceptively *low* exponent (0.508, below the GO line) but enormous
     absolute blowup (59.9×) and elim_h ≈ 0.2–0.33·n. Caught by the **blowup** criterion.
     A textbook case of ADR-199's "judge absolute height, not the scaling slope."
   - **FB15k-237** — elim_h ≈ 0.46·n (near-linear) already at n=2k; the densest, most
     hub-heavy KG. Caught by absolute blowup + tw lower bound.
   - **CoDEx-L (Wikidata)** — a tree-like *periphery* (5.1× at n=2k, road-like!) whose
     treewidth **collapses** as the sweep reaches the hub-dense core (elim_h 158→842,
     blowup 5.1→37.4×, fitted p=1.826). Caught by the **exponent** criterion — a
     single-n blowup snapshot would have been a **false GO**. This is the precise reason
     the pre-registration made the scaling exponent the primary metric.

3. **"Curated / bounded-degree" was the wrong property.** Average degree is low (2.6–9.2)
   for every KG, yet all carry heavy **hubs** (max degree 520 / 1999 / 4999 vs road's 9):
   high-frequency relations and top-level categories (`instance-of`, `country`,
   hypernym roots). Hubs are separator-killers — they force large balanced separators
   regardless of average degree. Knowledge graphs are **small-world with hubs**, the same
   structural class as the citation/embedding graphs ADR-199 killed. Road-like low
   treewidth comes from near-planar **geometric locality**, which no *semantic* graph has.

**Verdict: NO-GO for CCH contraction on curated knowledge graphs (WordNet, Freebase,
Wikidata).** Combined with ADR-199, this closes the CCH-full-contraction line: there is
no retrieval backbone of practical interest with road-network-like treewidth. The bet's
structural prerequisite — *uniformly* small separators — does not hold for semantic
graphs, full stop.

## Deviations from the frozen pre-registration (honesty record)

Three operational corrections were made; none touched the KG pass/fail bar or changed any
verdict (every KG misses on multiple criteria with wide margins):

1. **Blowup read at the ADR-199-matched reference n (~2k), not "largest n."** The frozen
   gate said "blowup ≤ 10× at largest n," but the road control revealed blowup grows with
   n *even for road networks* (7.2×@2k → 14.5×@20k), because `|G+|` grows ~n·polylog while
   separators stay O(√n). "≤10× at largest n" is internally inconsistent with "road
   reproduces 7.6×" (an N≈1.5k anchor). Reading absolute blowup at the matched reference n
   restores consistency; the **primary metric (exponent p) is unchanged**, and the road
   control still validates the run.
2. **Adaptive build-time budget** added so a high-treewidth contraction caps the sweep at
   moderate n instead of running away (it can otherwise consume minutes / exhaust memory).
   Side effect: FB15k-237 reached only one sweep point (exponent undefined); its verdict
   rests on absolute blowup + tw lower bound, which is sufficient and reported as such.
3. **CoDEx-L (genuine curated Wikidata) used as the faithful KG** in place of the literal
   2WikiMultiHop evidence graph — anticipated by the pre-registration's honesty clause;
   serves the identical structural role with far less data engineering.

## Consequences

**Positive.**
- The CCH-contraction direction is now decisively settled across **five** backbones
  (road, citation, two embedding manifolds in ADR-199, three KGs here) — no further
  speculative treewidth bets are warranted.
- The probe methodology is reusable: any future backbone can be screened in minutes,
  bracketed by the road control, with a treewidth lower bound that makes a NO-GO certain.

**Negative.**
- BET 3's intended multi-hop Graph-RAG benchmark is **not** built — correctly gated off by
  a cheap probe before the expensive data engineering (the protocol working as designed).

**Neutral / preserved value (unchanged from ADR-199).**
- The separator-tree **branch-and-bound pruning query** remains exact (recall 30/30 on
  every backbone here) and treewidth-independent — only *contraction/build* fails. Its
  reusable home is a treewidth-immune hierarchy (IVF/clustering, ADR-193/[ADR-201] BET 4),
  not separators over a semantic graph.
- The proven WIN of the thread is **BET 1** (re-weight vs rebuild under metric drift,
  [ADR-200]/[ADR-202]) — a different mechanism, unaffected by this NO-GO.

## Alternatives considered

- **Build the multi-hop QA benchmark first, measure treewidth later.** Rejected — inverts
  the cheap go/no-go gate; the whole point is to kill a high-treewidth backbone before the
  data engineering.
- **A sparser KG variant / k-core peeling to shed hubs.** Rejected as a rescue: ADR-199
  already showed hub-dampening *backfires* (it shrinks the denominator faster than it
  shrinks fill-in and destroys good cuts), and removing hubs removes exactly the
  multi-hop bridges retrieval needs. Not pursued without new evidence.
- **The literal 2WikiMultiHop evidence graph instead of CoDEx.** CoDEx-L is genuine
  curated Wikidata and serves the identical structural role with far less data-wrangling;
  given three independent KG NO-GOs spanning WordNet/Freebase/Wikidata, the literal 2Wiki
  edge list is unlikely to differ and is left as an unneeded follow-up.

## Open questions / next directions

- **Capitalize on BET 1** (the proven WIN, [ADR-202]) — live serving hook into the
  `ruvector-gnn` embedding-flush path; collapse-aware quality metric. Highest remaining EV.
- The pruning kernel on a **treewidth-immune** substrate (IVF) is the only place the
  salvaged separator-tree idea can still live; BET 4's standalone "beats plain IVF probe"
  head-to-head remains technically open ([ADR-201]).
