# Connectome-Driven Embodied Brain on RuVector

**Branch:** `research/connectome-ruvector`
**Started:** 2026-04-21
**Status:** Research + Design (pre-ADR)
**Coordinator:** goal-planner agent (initial master plan)

## Thesis

RuVector can be the substrate for a connectome-driven embodied brain system, but the substrate alone is not enough: a neural dynamics engine and a body simulator must sit around it.

## Positioning

**Not** "mind upload" or "consciousness upload." This is a **graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation**.

## Scientific grounding

- 2024 Nature paper: whole-fly-brain LIF model derived from the connectome alone reproduced feeding, grooming, and other sensorimotor transformations.
- FlyWire + Janelia: full adult fly connectome (~139,000 neurons, 50M+ synapses).
- Eon (2026): combined this LIF-from-connectome approach with NeuroMechFly and a visual system model to embody the brain in a virtual body.

## 4-layer architecture (to be deeply specified)

1. **Connectome & state graph** (RuVector-native) — typed nodes/edges, fast motif/adjacency queries.
2. **Neural dynamics engine** (new Rust crate) — event-driven leaky integrate-and-fire, delays, conductances.
3. **Embodied simulator** (external integration) — NeuroMechFly / MuJoCo / Brax body + sensory loop.
4. **RuVector analysis & adaptation loop** — mincut boundaries, motif discovery, coherence-collapse detection, counterfactual perturbation.

## Novel angles RuVector enables

- Subgraph boundary crossings preceding behavioral state changes (grooming, feeding, freezing).
- Coherence-collapse detection as a real-time "neural fragility" signal.
- Motif compression via vectorized trajectory embeddings.
- Counterfactual circuit perturbation by cutting / reweighting graph boundaries.

## Documents in this directory

Index is maintained by `00-master-plan.md` once the goal-planner agent publishes it. Each sub-document below will be written by specialist agents the planner dispatches.

- `00-master-plan.md` — goal-planner: decomposed goals, dependency DAG, phased roadmap
- `01-architecture.md` — system design of the 4-layer stack
- `02-connectome-layer.md` — RuVector graph schema for FlyWire; import pipeline
- `03-neural-dynamics.md` — Rust LIF kernel design (event-driven)
- `04-embodiment.md` — body simulator selection + sensory/motor coupling
- `05-analysis-layer.md` — mincut, sparsifier, motif, coherence primitives applied to connectome
- `06-prior-art.md` — literature survey + related-work differentiation
- `07-positioning.md` — product framing, scientific integrity, hype management
- `08-implementation-plan.md` — phased milestones, dependencies, risks

## Explicit non-goals

- No claims of consciousness upload or digital minds.
- No synthetic training of behavior — behavior must emerge from connectome + dynamics + body.
- No proprietary connectome data — FlyWire public release only.
- No new root-level files. All work lives in this directory.
