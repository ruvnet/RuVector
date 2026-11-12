# ADR-338: Electromagnetic World Model via Privileged-Modality Distillation

- **Status**: Proposed (stretch — ADR-only this wave; implementation deferred pending RuView coordination, consistent with ADR-332)
- **Date**: 2026-08-23
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-332 (PIR — RF sensing modality router; its deferral is honoured, not re-litigated); ADR-320 (AtomicObservation provenance — `SourceKind::RuViewRf` already exists); `crates/ruvector-mmwave`, `crates/ruvector-perception`; `ruvnet/RuView`; see `docs/research/perpetual-intelligence-runtime/12-wave5-evidence-review.md` and `13-wave5-program-plan.md`
- **Tags**: pir, wave-5, rf, csi, world-model, distillation, ruview, stretch

## Context

"Electromagnetic World Model for 6G: A Unified Framework for Joint
Environment Reconstruction and Channel Prediction"
([arXiv:2608.17769](https://arxiv.org/abs/2608.17769), submitted
2026-08-18; Zhao, Yu, Zhang, Zhang, Zhang, Liu) encodes partial CSI and
multi-view RGB into a shared representation processed by "a hierarchical
world-model backbone with local and global aggregation," then reads it
with two heads: "a **mixture-of-experts (MoE)-based CSI prediction head**
reconstructs the complete CSI, while a **depth prediction head** estimates
multi-view depth maps that are further converted into three-dimensional
(3D) point clouds."

Wave-5 evidence review grades it **A** — every figure exact, including
**SGCS 0.9699** for CSI prediction and **zero-shot generalization at
28 GHz**, with the metric name ("squared generalized cosine similarity")
correct in the briefing, which is a place briefings often drift. Primary
category is `eess.SP` only, so it will not surface in ML-venue sweeps.

**Artifact posture is the worst of this wave, and it drives the sizing.**
No code, **and no dataset**. The arXiv Comments field is empty, the
abstract carries no repository URL, and GitHub searches return nothing
attributable to the authors. Critically, the paper's dataset "is
constructed based on a campus digital twin" — the authors' own
construction, unreleased. **Reproduction therefore requires rebuilding a
digital twin from scratch, so the dominant cost is data generation, not
model implementation.** Anything sized as an integration effort would be
wrong by an order of magnitude. (Affiliations are absent from arXiv
metadata; author names are consistent with a BUPT / China Mobile
6G-channel-modelling group, but this was **not** confirmed from the PDF
and is not asserted here.)

**What exists here — checked at source, and it is close to nothing.**

- `crates/ruvector-mmwave/` is a **single 419-line `lib.rs`**: a `no_std`,
  zero-allocation UART frame parser for the Seeed mmWave protocol
  (MR60BHA2), surfacing `Event::{Breathing, HeartRate, Distance, Presence,
  Unknown}`. It is a **byte-level protocol decoder, not a sensing model** —
  no CSI, no learning, no embedding.
- The only actual CSI code is one bridge binary,
  `crates/ruvector-hailo-cluster/src/bin/ruview-csi-bridge.rs`.
- `crates/ruvector-perception/src/modality.rs` mentions CSI.
- `crates/ruvector-agent-memory/src/observation.rs` already enumerates
  `SourceKind::RuViewRf` — the provenance vocabulary ADR-332 relies on.
- Distillation infrastructure **exists but is sensor-unwired**:
  `crates/ruvllm/src/qat/{distillation,reasoning_loss,training_loop}.rs`
  and `crates/ruvllm/src/reasoning_bank/distillation.rs` are real
  teacher→student code, but for **LLM quantization**. There is no RGB or
  LiDAR ingest anywhere, no privileged-modality training loop, and no CSI
  dataset loader.

`ruvnet/RuView` is a live sibling repo (updated 2026-08-23) and the
primary home for this work. It is **not checked out locally**, despite
`ruvector-mmwave`'s docstring citing a `~/projects/RuView/firmware/` path.

**ADR-332, merged in Wave 4, already decided the posture** and says
verbatim: "**Deferred implementation. No code lands in this wave.**" This
ADR does not re-open that. It records a *design thesis* so that whichever
wave does implement has something specific to build against.

## Decision

1. **Do not reproduce the paper's architecture literally.** It requires
   multi-view RGB **at inference**, which defeats a substantial part of
   the RuView deployment thesis — commodity radio observations, no
   cameras. Reproducing it faithfully would produce a system that cannot
   be deployed where RuView is meant to deploy.
2. **Use RGB, LiDAR, or depth as *privileged training evidence* only.**
   The training-time teacher consumes CSI **plus** privileged modalities
   and learns the shared electromagnetic representation; the deployed
   student consumes **CSI alone** and is distilled from that teacher.
   Target capability at deployment: geometry, motion, and channel, from
   commodity radio observations.
3. **State the intended result plainly, so it can be falsified**: a
   **camera-free electromagnetic world model trained using temporary
   visual supervision**. That is a deliberate divergence from the source
   paper, not a reimplementation of it, and it should be evaluated as
   such.
4. **Implementation is deferred**, consistent with ADR-332. No code lands
   in Wave 5.
5. **Size the eventual work package by its data-generation cost.** With
   neither code nor dataset released, and the source dataset being a
   self-built campus digital twin, the dominant cost is constructing
   training data — not implementing the backbone.
6. **Reuse the provenance vocabulary that already exists.** Any CSI
   observation entering agent memory uses `SourceKind::RuViewRf`
   (ADR-320); the privileged modalities must be **recorded as
   training-only provenance** so that no deployed inference path can
   silently acquire a dependency on them.

## Consequences

### Positive

- Preserves the deployment thesis: if the student needs only CSI, the
  system runs on commodity radio hardware, which is the entire economic
  argument RuView rests on.
- Privileged-modality distillation is a well-understood pattern
  (learning-using-privileged-information), so the risk is concentrated in
  data generation rather than in an unproven training scheme.
- A camera-free student is a **stronger and more falsifiable** claim than
  reproducing a camera-dependent model — it either works without RGB at
  inference or it does not.

### Negative

- **Distillation from privileged modalities loses information**, and the
  paper's 0.9699 SGCS was achieved *with* RGB at inference. The student
  should be expected to underperform that, and the acceptance bar must be
  set against a CSI-only baseline rather than against the paper's number.
- **The dominant cost is building a dataset that does not exist.**
- The primary implementation home (`ruvnet/RuView`) is a separate
  repository, so this inherits the same cross-repo coordination cost as
  ADR-332 and ADR-336.
- ADR-332's corrected RF attributions remain load-bearing context: IR-UWB
  89.0% cross-subject F1 / 78.5% unseen-room / €14; FMCW 83.4% / **83.8%
  best unseen-room** / €20; Wi-Fi 79.0% / 68.8% / €320 (SDR development
  hardware, not commodity CSI gear). Wi-Fi's 92.6% sleep-interruption
  score is the **floor** of a three-way near-tie, not a headline. Any
  modality decision made under this ADR uses those, not the swapped
  version from an earlier briefing.

## Security / Validation Gates

- **No implementation this wave** — ADR-only, consistent with ADR-332.
- **Re-verify before implementing**: ADR-332 already requires
  re-verification of its source (arXiv:2608.20322, under review at IEEE
  Access, not peer-reviewed) before any implementation. The same applies
  to arXiv:2608.17769, whose numbers are unreproducible today for want of
  code and data.
- **Training-only provenance is enforced, not assumed** (Decision §6): a
  deployed inference path that acquires a privileged-modality dependency
  is a regression, and the provenance records are what make that
  detectable.
- **External grounding** (ADR-324): any eventual evaluation must trace to
  external evidence. ADR-332's designated open corpus
  (`gitlab.ilabt.imec.be/datasets/Activity-recognition-datasets`, verified
  live) satisfies this by construction and is the natural starting point —
  in contrast to the EMWM paper's own unreleased digital twin.
- **Citation discipline**: cite as "EMWM (arXiv:2608.17769)" in full —
  "world model" is heavily overloaded (JEPA, Genie, Dreamer) and "EMWM" is
  an unregistered acronym. Do not adopt either as a crate or module name.

## Affected Repos

- `ruvnet/ruvector`: **docs only this wave.** No code.
- `ruvnet/RuView`: primary future implementation home. Not checked out
  locally; coordination required before any work begins.
- Possibly `crates/ruvector-mmwave` here for a future CSI ingest path —
  noting that today it is a 419-line UART frame parser, not a foundation
  to build a world model on.

## Dependencies

Soft: ADR-332 (modality router — this refines its deferred implementation
direction), ADR-320 (`SourceKind::RuViewRf` provenance vocabulary),
ADR-324 (external-grounding invariant). No hard dependencies; nothing in
Wave 5 blocks on it.

## Alternatives Considered

- **Reproduce the paper's architecture faithfully**: rejected — it needs
  multi-view RGB at inference, which contradicts the RuView deployment
  thesis. See Decision §1.
- **Skip the teacher and train a CSI-only model directly**: rejected as
  the primary path — the paper's core finding is that joint optical/RF
  training yields a better shared representation, and privileged-modality
  distillation is precisely how to keep that benefit without the
  deployment cost. Worth retaining as the **baseline** the student must
  beat.
- **Implement in Wave 5**: rejected — ADR-332 deferred implementation, no
  dataset exists, and `ruvnet/RuView` is not even checked out locally.
  Implementing would mean building a digital twin first.
- **Size it as an integration of a released model**: rejected — there is
  no released model and no released dataset. This is
  reproduction-from-description with a heavy data-generation component.
