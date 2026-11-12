# ADR-332: RF Sensing Modality Router (Stretch)

- **Status**: Proposed (stretch — ADR-only this wave, implementation deferred pending RuView coordination)
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: `ruvnet/RuView` (sibling repo — Wi-Fi CSI sensing; sensing-only boundary per Wave-3's novelty mapping); `crates/ruvector-mmwave` (sensing-adjacent crate in this repo); ADR-320 (SourceKind already enumerates `RuViewRf` as an observation source); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, rf-sensing, ruview, modality-routing, stretch

## Context

Wave-4 evidence review grades this paper **B+** —
[arXiv:2608.20322](https://arxiv.org/abs/2608.20322), "A comparison
between ceiling-mounted FMCW, IR-UWB and Wi-Fi radar for in-bedroom human
activity monitoring and sleep interruption detection," submitted
2026-08-20, **under review at IEEE Access — not yet peer-reviewed**.
Twenty subjects, six room layouts, same classifier across simultaneously
recorded modalities.

**The program brief's technology-to-number mapping was wrong and is
corrected here — cite only these attributions** (verified against the
paper's HTML full text):

| | IR-UWB | FMCW | Wi-Fi |
|---|---|---|---|
| Cross-subject activity macro F1 | **89.0%** (best) | 83.4% | 79.0% |
| Unseen-room macro F1 | 78.5% (−10.5 pp) | **83.8%** (best; +0.4) | 68.8% (−10.2 pp) |
| Sleep-interruption (hardest layout) | 94.2% | 93.4% | **92.6%** (floor — all exceed 92%) |
| Approx. component cost | **€14** | €20 | €320 |

The brief listed the numbers in a fixed FMCW/IR-UWB/Wi-Fi order while the
figures actually run best-to-worst — swapping the FMCW and IR-UWB
attributions on both F1 and cost, and presenting Wi-Fi's 92.6% sleep
score without noting it is the floor of a three-way near-tie. The Wi-Fi
cost caveat is the paper's own: €320 is a specialized SDR configuration
for monostatic sensing, development hardware, not commodity CSI gear.

**Artifact availability — checked, not assumed**: an open synchronized
dataset at gitlab.ilabt.imec.be/datasets/Activity-recognition-datasets
(fetched, live). **No model code** — the CNN is described, not shipped.
Dataset-available, code-absent posture.

**Implication accepted from the brief (with corrected numbers)**: the
finding does not say abandon Wi-Fi — Wi-Fi remains excellent for coarse
sleep/presence monitoring (≥92.6%) and is the ubiquitous-infrastructure
play. It says **sensing modality should stop being a fixed architectural
decision**: IR-UWB wins fine-grained activity at the lowest cost; FMCW
wins unseen-environment robustness; Wi-Fi wins ubiquity and coarse
monitoring.

## Decision

Adopt the modality-router posture, ADR-only this wave:

1. **Modality becomes a routed decision, not an architecture.** RuView's
   strategy target is a general RF intelligence layer: Wi-Fi for
   ubiquitous presence and coarse behavior, IR-UWB where fine activity
   recognition and low cost matter, FMCW where geometry and unseen-
   environment robustness matter — selected per deployment/task, ideally
   via the ADR-331 VoI primitive (modality choice is a routing decision
   with measurable quality/cost tradeoffs).
2. **A common representation above all three.** RuVector learns the
   shared embedding layer above modality-specific frontends; ADR-320's
   `SourceKind::RuViewRf` observations already give the provenance
   vocabulary for multi-modality fusion, and per-modality source tagging
   is required so ADR-330's arbitration can treat co-located sensors
   observing one event as correlated evidence, not independent votes.
3. **Deferred implementation.** No code lands in this wave. The
   implementation home spans `ruvnet/RuView` (sibling repo,
   sensing-side) and possibly `crates/ruvector-mmwave`; sequencing is
   after the acceptance-test items, per ruv's stated priority order
   (fifth of six). The released imec dataset is the designated
   evaluation corpus when implementation begins — an external,
   independently published benchmark, satisfying ADR-324's
   external-grounding invariant by construction.

## Consequences

### Positive

- Converts a challenge to the Wi-Fi-first strategy into a routing
  opportunity without abandoning the Wi-Fi installed-base advantage.
- The corrected table gives the program a citable, verified basis for
  per-modality claims before any hardware decision.
- An open external dataset exists for grounded evaluation from day one.

### Negative

- Not yet peer-reviewed; single study, 20 subjects, bedroom setting —
  conclusions may narrow under review. Re-verify before implementation
  hardens.
- Cross-repo scope (RuView) means this ADR alone decides posture, not
  placement details; a RuView-side ADR is required when work begins.
- No model code shipped — classifier reproduction is from-description.

## Security / Validation Gates

- **Citation discipline (blocking on docs)**: any figure from this paper
  must carry the per-technology attribution from the corrected table
  above; the brief's swapped ordering must never propagate.
- **Correlated-sensor rule**: multi-modality deployments must tag
  per-modality provenance so ADR-330 arbitration applies.
- Standard gates apply when implementation lands (future wave).

## Affected Repos

- Docs-only in `ruvnet/ruvector` this wave. Future implementation:
  `ruvnet/RuView` (primary), possibly `crates/ruvector-mmwave` here.

## Dependencies

Soft: ADR-330 (correlated-evidence handling), ADR-331 (VoI as the
modality-selection rule). No hard dependencies; nothing blocks on this
ADR.

## Alternatives Considered

- **Keep Wi-Fi-first as a fixed architecture**: rejected — the verified
  numbers show a 10.2 pp unseen-room drop and a fine-activity gap
  against a €14 alternative; fixing the modality forfeits both.
- **Switch to IR-UWB-first**: rejected — same fixed-architecture error
  in the other direction; Wi-Fi's ubiquity and coarse-monitoring parity
  (≥92.6%) remain decisive for presence/sleep workloads.
- **Implement this wave**: rejected — fifth in ruv's priority order,
  cross-repo, dataset-only artifact, and the paper is not yet
  peer-reviewed; posture now, implementation after re-verification.
