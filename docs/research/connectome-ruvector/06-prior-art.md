# 06 - Prior Art and Differentiation

> Framing reminder: this is a graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation. It is not consciousness upload. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Survey the published landscape this project sits inside, so `./07-positioning.md` can frame the work honestly, `./08-implementation-plan.md` can avoid reinvention, and reviewers can see who did what first. Each section names what is published, what is open source, and where this project overlaps with and differs from the work.

## 2. The 2024 Nature whole-fly-brain paper

**Citation and summary.** Dorkenwald et al., *Nature* 2024, published as part of the FlyWire consortium's capstone set of papers. Parallel companions: Matsliah et al., *Nature* 2024, which published the community-proofread FlyWire v783 connectome; Schlegel, Yin et al., which published the cell-type catalogue; Lin et al., which applied LIF-from-connectome at whole-fly scale and demonstrated that behaviors — feeding response, grooming, some sensorimotor transformations — can be reproduced by a connectome-derived LIF model *without* trained parameters.

**What's published.** Connectome (public), LIF reproduction of behaviors in a non-embodied setting, predicted neurotransmitter maps, cell-type hierarchy.

**What's open.** FlyWire data is CC-BY and community-accessible. The LIF simulation code from the Lin et al. paper is available (Python + JAX) as a companion repository under the FlyWire consortium's research releases. Cell-type catalogue is open.

**Overlap with this project.** We use FlyWire directly (`./02-connectome-layer.md`). We build a Rust LIF kernel that is model-compatible with the Lin et al. regime (`./03-neural-dynamics.md`).

**Differentiation.**
- Rust runtime, not Python/JAX. See `./03` §8.
- Embodied (NeuroMechFly body, `./04-embodiment.md`), not bare LIF-in-Python.
- Side-car analysis layer built on RuVector graph primitives (dynamic mincut, sparsifier, spectral coherence, DiskANN trajectory index), not offline analysis. See `./05-analysis-layer.md`.
- Auditable counterfactual circuit surgery with `ruvector-mincut::certificate` witnesses, not unstructured edge removal.

This is the closest published anchor and the one this project explicitly builds on. No claim that we reproduce the science; we claim we make the substrate better.

## 3. FlyWire (connectome data)

**What's published and open.** The FlyWire v783 release (2024): ~139,255 neurons, >54M synapses with predicted NT, proofread meshes, cell-type catalogue. Access via CAVE/annotation clients, bulk CSV download, neuroglancer visualization. License: CC-BY.

**Overlap.** Our graph schema (`./02-connectome-layer.md`) is tailored to the FlyWire release, including cell-type, hemilineage, and NT prediction.

**Differentiation.** FlyWire is data; we are a runtime. Where FlyWire is consumed by scientists in Python via `fafbseg` / `navis`, we consume it in Rust via a streaming CSV loader into `ruvector-graph` with CSR materialization, AgentDB embeddings, and on-disk rvf snapshot.

## 4. Janelia hemibrain

**Published.** Scheffer et al., *eLife* 2020. A ~25K neuron dense EM connectome of the central brain, with synapse-level annotations and neuPrint API. Very high proofreading density.

**Open.** Yes. Raw data, neuPrint API, morphologies. Apache-2.0-ish.

**Overlap.** We can optionally load hemibrain as a second dataset. Our schema multi-dataset from day one (`./02` §7).

**Differentiation.** Hemibrain is half-brain, FlyWire is whole-brain; we default to FlyWire. Hemibrain is a cross-validation target.

## 5. NeuroMechFly v1 and v2

**Published.** Lobato-Rios et al., 2022 (v1); extended NeuroMechFly v2 releases in 2023-2024 with improved contact, visual perception, and interface with spiking controllers. University of Lausanne / EPFL (Ramdya lab).

**Open.** MJCF body model, MuJoCo-compatible. Example controllers in Python. License: research-friendly; check the specific release note at ingest time.

**Overlap.** Our embodiment layer (`./04-embodiment.md`) uses NeuroMechFly v2 MJCF directly.

**Differentiation.** Their controllers are Python; ours is a Rust bridge into MuJoCo with a deterministic contract to the LIF kernel. Their analysis is behavioral; ours is connectome-level with mincut boundaries, coherence collapse, and counterfactual cuts on the *same* run.

## 6. Eon (2026)

**Context.** Eon is the contemporaneous project that most closely resembles the combination of "LIF-from-connectome + NeuroMechFly + visual system model → embodied simulation." Based on public talks and preprints around the 2024 Nature paper, Eon appears to be the first project to demonstrate closed-loop connectome-embodied simulation at fly scale.

**Published.** Preprints, conference talks, website. Concrete releases (as of writing) are Python/JAX-first.

**Open.** Partial; MJCF body derives from NeuroMechFly and is public, the LIF + visual-system pipeline is in the process of being open-sourced.

**Overlap.** Functionally the same pipeline shape: connectome + LIF + body + vision. This is the project we differentiate against hardest.

**Differentiation.**
- Rust runtime + graph-native substrate. Eon is Python/JAX-based.
- Analysis story: our claim is not "we embody a fly", it is "we give you a graph-native, auditable, counterfactual-capable runtime for a connectome under a body". The RuVector analysis layer (`./05`) is the differentiator. Eon does connectome embodiment; we do connectome embodiment *plus* the structural-coherence / boundary / counterfactual-cut analysis that RuVector's graph crates uniquely enable at scale.
- Dataset neutrality: we treat FlyWire v783 as v1, hemibrain as v2, OpenWorm as sanity. Eon is, to our current understanding, FlyWire-first.

This is not a competitor stance; Eon is an adjacent project and should be cited as the leading publicly visible demonstration. The differentiation is methodological — substrate and analysis surface — not scientific.

## 7. OpenWorm

**Published.** Sibernagel et al., *PLoS One* 2014; ongoing releases to the present. A community-driven simulation of *C. elegans* (302 neurons, ~7,000 synapses) in a deformable body.

**Open.** Heavily so. `c302` (NEURON-based), `Sibernetic` (fluid-body sim), `Geppetto` front-end, NeuroML models.

**Overlap.** Same pipeline shape at smaller scale: connectome + dynamics + body.

**Differentiation.** OpenWorm is *C. elegans* scale; our target is FlyWire scale. OpenWorm uses NeuroML and NEURON; we use Rust LIF. OpenWorm has no equivalent of the RuVector graph analysis layer.

**Why we care.** `C. elegans` is the smallest-possible test case for our pipeline. One end-to-end run fits in a few MB of RAM. Useful for CI and sanity tests; we include it in the roadmap as a 1-day bring-up.

## 8. Blue Brain Project

**Published.** Markram et al., *Cell* 2015 (neocortical microcircuit); many follow-ups through 2024; the recent "digital twin" of mouse somatosensory cortex.

**Open.** Partial — NEURON-based detailed biophysical models, BluePy tooling, some circuit releases under open licenses.

**Overlap.** Philosophy: data-driven circuit reconstruction + biophysical simulation.

**Differentiation.**
- Mammalian (rat / mouse cortex) vs. invertebrate (Drosophila). Different scientific regime.
- Biophysical Hodgkin-Huxley multi-compartment models vs. connectome-constrained LIF. Ours is much cheaper to run; theirs is more biophysically committed.
- They do not use dynamic mincut / sparsifier / DiskANN-indexed trajectories. We do.
- Their simulator (NEURON + Python glue) is a different stack.

We do not compete with Blue Brain. We are a lighter, graph-first, embodied variant at insect scale.

## 9. Flood-Filling Networks (FFN) and EM segmentation

**Published.** Januszewski et al., *Nature Methods* 2018 (FFN); widely used in Google-Janelia-FlyWire segmentation pipelines.

**Open.** FFN reference implementation open; pipelines (CloudVolume, Neuroglancer, CAVE) open.

**Overlap.** FFN is upstream of FlyWire; we consume its output.

**Differentiation.** Not in our scope to do segmentation. We consume the proofread connectome. If we want to extend to a new dataset, FFN-class pipelines are how the dataset gets produced.

## 10. Other named projects

### 10.1 MICrONS

1 mm³ of mouse visual cortex reconstruction (IARPA). Open data. Not at whole-brain scale, not fly. Out of v1 scope; a possible v3+ dataset target.

### 10.2 Zheng et al. FAFB

The EM volume FlyWire reconstructs from. Relevant only as the substrate for FlyWire; no direct use here.

### 10.3 Virtual Fly Brain

A browser visualization and ontology layer over fly connectome data. Useful for cross-referencing cell-type IDs; not a runtime.

### 10.4 Nengo / Spaun

Chris Eliasmith's work: population-coded spiking neural simulations, Spaun as a cognitive demonstration. Not connectome-derived in our sense; population-level abstraction.

### 10.5 Brian2 / NEST / GeNN

Simulation engines compared in `./03-neural-dynamics.md` §8. Prior art in the *tool* dimension, not in the *substrate* dimension.

### 10.6 CATMAID / neuPrint / CAVE

Connectome viewing/annotation stacks. Input tools, not runtime.

### 10.7 Dendrify

Reduced-compartment dendritic models. Rust analog already in `crates/ruvector-nervous-system::dendrite`. Used as an optional enhancement in `./03` §7.

## 11. The RuVector-native precedents we lean on

Our substrate story rests on RuVector primitives that have their own ADR histories:

- ADR-014 / ADR-015 — coherence engine and coherence-gated transformer. Establishes `ruvector-coherence` as a first-class metric surface.
- ADR-017 — temporal tensor compression. Relevant for spike-train storage at scale.
- ADR-144 / ADR-146 — DiskANN Vamana implementation. Scales trajectory indexing.
- ADR-148 / ADR-149 — brain hypothesis engine and brain performance optimizations. Demonstrates the pi-brain scale (13K memories, 1.2M graph edges) relevant to our episode index.
- ADR-150 — pi-brain on ruvltra / Tailscale. Deployment precedent.

Nothing in our stack is unproven at its own scale. The contribution is composing these into a connectome-runtime surface and applying them live.

## 12. Differentiation table (compressed)

| Prior work | Primary scale | Runtime language | Graph analysis surface | Embodied | Counterfactual cuts | Auditable witnesses |
|---|---|---|---|---|---|---|
| Nature 2024 (Lin et al.) | 139K (fly) | Python/JAX | offline | no | ad hoc | no |
| Eon | 139K (fly) | Python/JAX | limited | yes | limited | no |
| Blue Brain | mammalian cortex | C/Python | custom | partial | no | no |
| OpenWorm | 302 (worm) | Python/NEURON | small | yes | no | no |
| FlyWire tools | 139K (fly) | Python | viewer + stats | n/a | n/a | n/a |
| Brian2/NEST/GeNN | variable | Python/C++ | n/a | n/a | n/a | n/a |
| **This project** | **139K (fly)** | **Rust** | **dynamic mincut + sparsifier + spectral + DiskANN** | **yes** | **yes** | **yes (mincut certificates)** |

## 13. What is genuinely new here

1. Rust-native runtime for a FlyWire-scale connectome with a deterministic event-driven LIF engine and a body bridge.
2. A graph-analysis surface (`./05-analysis-layer.md`) — dynamic mincut boundaries, coherence collapse, DiskANN-indexed trajectories, auditable counterfactual surgery — that no peer project exposes together.
3. The combination of pi-brain-scale infrastructure (AgentDB, DiskANN, ONNX embeddings) with a connectome object as the primary store.

None of the above is a scientific claim about the brain. They are substrate claims about what you can do if the connectome lives in RuVector. The science (does this reveal something biologically interesting?) is the next stage, and the prior art that governs it is the 2024 Nature paper regime.

## 14. Citations to copy into downstream writeups

- Dorkenwald, Matsliah, et al., *Nature* 2024 — FlyWire whole-brain connectome release set.
- Lin et al. (FlyWire consortium), *Nature* 2024 — LIF-from-connectome behavioral reproduction.
- Scheffer et al., *eLife* 2020 — hemibrain.
- Lobato-Rios et al., *eLife* 2022 — NeuroMechFly v1.
- Eon project, preprint — connectome-embodiment pipeline.
- Markram et al., *Cell* 2015 — Blue Brain neocortical microcircuit.
- Januszewski et al., *Nature Methods* 2018 — Flood-Filling Networks.
- Stimberg et al., *eLife* 2019 — Brian2.

These are the citations every downstream doc should reference when making a substrate claim. `./07-positioning.md` uses them as the reference frame for every external-facing statement.
