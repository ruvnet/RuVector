# 07 - Positioning, Scientific Integrity, and Hype Management

> Framing reminder (also the thesis of this document): this is a graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation. It is not a model of consciousness, a mind upload, a digital person, a substrate-independent intelligence, or a proxy for any of those. See `./00-master-plan.md` §1.

## 1. Purpose

Fix how this project is talked about, written about, and published. The positioning is load-bearing: if the first external sentence drifts, the science and the engineering both get dismissed. This doc is the rubric every README, paper abstract, tweet, and demo voiceover has to pass.

## 2. What this project is

Exactly and only the following:

1. **A Rust runtime** that ingests the FlyWire whole-fly-brain connectome into a typed graph store (`./02-connectome-layer.md`).
2. **An event-driven LIF kernel** that advances that graph under connectome-derived dynamics (`./03-neural-dynamics.md`).
3. **A body bridge** to NeuroMechFly in MuJoCo for closed-loop sensorimotor operation (`./04-embodiment.md`).
4. **A graph-analysis surface** — dynamic mincut, sparsifier, spectral coherence, DiskANN-indexed trajectory search, auditable counterfactual circuit surgery — applied live to the running simulation (`./05-analysis-layer.md`).

The scientific anchor is the 2024 Nature whole-fly-brain result showing that behavior can emerge from a connectome-derived LIF model without trained parameters. Everything this project does is aligned to that regime; nothing tries to reach past it.

## 3. What this project is not

A non-exhaustive, binding list:

1. **Not consciousness upload.** We do not claim the simulated fly has subjective experience, qualia, sentience, phenomenology, awareness, or any equivalent property. We make no claim about whether connectome simulation could in principle have such properties.
2. **Not mind upload.** We do not transfer, preserve, reconstruct, or otherwise instantiate an individual's mental state. There is no individual fly whose mind is being "uploaded"; there is a connectome average across proofread individuals, run under model dynamics.
3. **Not a digital person / digital being / digital creature.** The closed-loop simulation is a scientific artifact that produces spike trains and torques. Anthropomorphic framing ("the fly decides", "the fly is hungry") is metaphor only, to be quarantined.
4. **Not artificial general intelligence.** Connectome-scale simulation of an invertebrate is not AGI and not on a path to it. We do not gesture at AGI even indirectly.
5. **Not proof of biological or ethical personhood of any simulation.** No moral patiency claims.
6. **Not a pharmaceutical or clinical tool.** We do not market this for drug discovery, disease modeling, or anything clinical.
7. **Not a product of proprietary connectome data.** FlyWire public release only.

## 4. The positioning one-liner

> **A graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation.**

Every external document has this as its anchor. The one-liner is the first sentence of the README, the abstract, and every keynote slide deck. Paraphrases are allowed; substitutions are not.

## 5. Long-form positioning (for the README and the preprint)

> This project is a Rust runtime for running a whole-fly-brain (*Drosophila melanogaster*, FlyWire v783) connectome under event-driven leaky integrate-and-fire dynamics, coupled to a NeuroMechFly body in MuJoCo, with a dynamic graph-analysis layer that exposes structural coherence metrics, subgraph motif discovery, and auditable counterfactual circuit surgery. The scientific anchor is the 2024 Nature whole-fly-brain result showing that behaviors can emerge from a connectome-derived LIF model without trained parameters; this project's contribution is substrate, not new biology: it demonstrates that a graph-first runtime (RuVector) can host a connectome-scale embodied simulation with built-in analyses that prior pipelines implement offline or not at all. We make no claim about consciousness, upload, sentience, or digital personhood.

The last sentence is mandatory in the README. It is the canary the community reads to decide whether to take the work seriously.

## 6. Hype-avoidance rubric (binding on all prose)

Before any external text is published, each of these must be true:

- [H1] The sentence does not anthropomorphize the simulated fly beyond the precision used by the 2024 Nature paper. "The fly exhibits a grooming-like motor pattern" is allowed; "the fly wants to groom itself" is not.
- [H2] The sentence does not use "consciousness," "mind," "self," "sentience," "awareness," "subjective," "qualia," "experience" (in the phenomenal sense), or synonyms. "Behavior," "dynamics," "state," "motor pattern," "circuit," "coherence" are fine.
- [H3] The sentence does not imply upload, transfer, preservation of an individual organism. Connectomes are reference templates, not individual minds.
- [H4] Any claim about "novel discovery" names the null control used to establish significance (shuffled spike times, rewired connectome, etc.).
- [H5] Any causal claim from counterfactual surgery cites the mincut witness receipt.
- [H6] Any performance claim names hardware, dataset version, configuration hash.
- [H7] The work's differentiation is expressed against prior art from `./06-prior-art.md` — not vaguely against "prior approaches."

Failing any of these requires a rewrite before publication.

## 7. Prose examples

### Acceptable

- "Running a FlyWire-scale LIF over a NeuroMechFly body at 25 Hz control rate on a single laptop CPU, we observe grooming-like motor patterns when descending command neurons are stimulated, consistent with the 2024 Nature whole-fly-brain regime."
- "The coherence score drops below its baseline 180 ms before 72% of observed walking-to-grooming transitions in 10 replay episodes (shuffled-spike null p < 0.01)."
- "Cutting the α/β lobe connections of the mushroom body (500 edges, witness 0x7f3a…) reduces the feeding-like motor output by a factor of 4 in replay, compared to the unmasked run."

### Unacceptable

- "The digital fly decides when to eat." (anthropomorphic)
- "We uploaded a fly brain to silicon." (upload framing)
- "A first step toward preserving minds." (upload framing)
- "The simulated fly is conscious of its environment." (consciousness claim)
- "The fly has experiences in its virtual world." (phenomenal framing)

## 8. Audience targeting

Three primary audiences, in priority order:

### 8.1 Neuroscience labs (fly circuits, connectomics, systems neuroscience)

**Hook.** Dynamic graph analysis you cannot do offline: mincut boundaries tracked live, motif libraries updated per second, counterfactual circuit surgery with audit trails. All on the FlyWire release they already use.

**Venue.** *Nature Methods*, *eLife*, *Current Biology*, *Neuron*. Workshop at Cosyne, society for neuroscience, FlyWire community meetings, Janelia methods workshop.

**Deliverable.** Reproducible one-command demo (`cargo run --example walking_grooming`) plus a methods preprint pointing at the Rust crates and the replay bundle format.

### 8.2 Embodied-AI and neuromorphic-ML researchers

**Hook.** A Rust-native, deterministic, event-driven LIF engine hooked to a high-fidelity insect body. No trained parameters; behavior emerges from the connectome. This is a clean substrate for hybrid connectome-ML experiments.

**Venue.** NeurIPS, ICLR, Neuromorphic Computing and Engineering, CCNeuro.

**Deliverable.** Papers that reuse `ruvector-lif` + `ruvector-embodiment` and extend the analysis layer with their own learning rules, with our engine as an unchanged dependency.

### 8.3 Safety-oriented ML researchers

**Hook.** Auditable counterfactual analysis of a complex recurrent system whose "parameters" (the connectome) are structural, not learned. A sandbox for interpretability and intervention that has ground-truth structural knowledge absent from LLMs.

**Venue.** ICML interpretability workshop, FAccT if framed carefully, Anthropic / OpenAI safety teams as direct reach.

**Deliverable.** Case studies: "the mincut boundary between region A and region B is load-bearing for behavior X; cutting it with a certified witness produces the expected deficit." Parallel to mech-interp in LLMs, but on a system with a known graph.

**What we do not target.** Transhumanist communities, longevity/mind-preservation communities, consumer "brain tech" audiences. Even if they show up with interest, we point them at `./06-prior-art.md` and the 2024 Nature paper and decline to frame our work for them.

## 9. Venue strategy (recommended sequence)

1. **Workshop preprint + demo (M4 end):** bioRxiv preprint describing the runtime and one analysis result (coherence collapse predicts transition, or motif library is maintained live). Cosyne 2027 poster if timing works.
2. **Rust tooling release (M5 mid):** crates published (`ruvector-lif`, `ruvector-embodiment`, thin wrappers under `ruvector-connectome*`), with Apache-2.0 license, explicit CITATION.cff pointing to the 2024 Nature paper.
3. **Methods paper (M5 end):** *Nature Methods* or *eLife*. Title variant: "A graph-native runtime for embodied connectome simulation with live coherence analysis and counterfactual circuit testing."
4. **Follow-up scientific paper (v2):** if the coherence-collapse / boundary-crossing signals are robust, submit to *Nature Neuroscience* or *eLife* as a methods-and-findings piece, co-authored with a fly-circuits lab as domain partner.
5. **Safety / interpretability track (ongoing):** use the counterfactual surgery harness as a case study in ML safety venues when invited.

## 10. Open-source governance

- Repository stays under `ruvnet/ruvector`, branch policy normal (`research/connectome-ruvector` → `main` via PR after review).
- Apache-2.0 for all new crates. FlyWire data is not distributed; we pin a specific version in the manifest and download at build time via a standard loader.
- Third-party contributions welcome; CLA optional (Apache-2.0 contribution model is enough).
- ADRs get written only *after* research locks in; this branch does not touch ADR numbering.
- Each preprint and release carries the hype-avoidance rubric (§6) as a visible CONTRIBUTING note.

## 11. Red-flag phrases to pre-emptively strike

Automated scanning should flag any of the following in PR descriptions, commit messages, READMEs, or docs:

- "consciousness", "conscious"
- "upload", "uploading a mind"
- "substrate-independent", "brain-in-silico" (the latter is weak; keep out of titles even if technically defensible)
- "digital person", "digital being", "digital fly" (as an entity, not as an artifact)
- "first step toward AGI"
- "emergent sentience"
- "wakes up", "comes alive"
- "simulation of experience", "phenomenal"

A CI rule on docs checks for these in our `docs/` directory. Matches fail the lint.

## 12. Where this project could drift (so we watch)

1. A demo video cut for social media uses anthropomorphic voiceover. Mitigation: maintainer approves all public media; red-flag scan on transcripts.
2. A press release misquotes. Mitigation: no press releases without the long-form positioning (§5) attached.
3. An internal code comment or variable name implies consciousness. Mitigation: lint also covers `src/` for the red-flag phrases.
4. A collaborator's paper that uses our runtime reframes it as upload. Mitigation: license is permissive, but the CITATION.cff requests an accurate description of the runtime.

## 13. What success looks like

1. A neuroscientist reads `./07-positioning.md` and says "yes, this is an honest methods paper, not hype."
2. A safety researcher reads `./05-analysis-layer.md` and says "this is an auditable intervention harness I can reuse."
3. A journalist reads the README and writes about "graph-native connectome runtime," not "digital mind."
4. The project's public artifacts (repo, preprint, demos) pass the §6 rubric at every checkpoint.
5. The 2024 Nature paper is cited in every one of our external documents; no external doc exists that misrepresents the connection.

Anything less is a positioning failure and a scientific failure at the same time, which is why this document exists before `./08-implementation-plan.md`.
