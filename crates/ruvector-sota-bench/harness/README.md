# RuVector MetaHarness adapter

This package connects the RuVector SOTA benchmark to the current MetaHarness
stack while keeping ADR-282's research gate as the only path to a nightly PR.
The dependency lock pins:

- `metaharness` 0.4.2
- Darwin 0.8.0
- Flywheel 0.1.7
- Algorithmic Harness 0.1.0
- Router 0.3.2
- Workspace Probe 0.1.1
- Red/Blue 0.1.4
- Weight-EFT 0.1.1

Requires Node 20 or newer. Node 22 is used in CI.

```bash
cd crates/ruvector-sota-bench/harness
npm ci --ignore-scripts
npm test
npm run doctor
```

## Commands

```bash
# Verify every capability and its pinned version
npm run doctor

# Run the signed Flywheel loop. Builds target/release/sota-all first if needed.
cargo build --release -p ruvector-sota-bench --bin sota-all
node dist/src/cli.js flywheel --repo "$(git rev-parse --show-toplevel)" \
  --generations 4

# Run Darwin against a human-curated, hash-pinned benchmark suite
node dist/src/cli.js darwin --repo "$(git rev-parse --show-toplevel)" \
  --bench /absolute/path/to/suite.json --generations 3 --children 4 --seed 1
```

`darwin-ann` uses Darwin's injected GEPA evaluator directly over the RuVector
ANN genome (`ef_search`, `M`, construction effort, and runner topology), keeps
per-instance scores, and preserves the Pareto frontier:

```bash
node dist/src/cli.js darwin-ann --repo "$(git rev-parse --show-toplevel)" \
  --candidates 12
```

The generic `darwin` command evolves the seven MetaHarness operating-policy
surfaces with Pareto selection, crossover with epistatic linkage, bootstrap
promotion, a cumulative risk budget, a 5% cost ceiling, FDR, and curriculum. It
refuses to run without a hash-verified benchmark suite.

Flywheel evolves the constrained ANN genome and evaluates five paired items per
arm. Confirmation and frozen-anchor suites must have disjoint dataset/seed
identities. The paired bootstrap must clear the preregistered 0.005 minimum
effect; inconclusive results do not promote. Flywheel checkpoints each
generation and signs its lineage. The final replay
bundle is independently re-verified against the frozen RuVector promotion gate.
Generated files live under `.metaharness/ruvector/`:

- `checkpoint-N.json` — resumable state plus partial replay bundle
- `replay-bundle.json` — signed, externally verifiable promotion lineage
- `research-gate-input.json` — evidence handoff with
  `pr_creation_authorized: false`

Runs are cached by commit, normalized policy, dataset hash/seed, embedding-space
identity, native-binary fingerprint, environment, and extra arguments. Native
`ef_search` sweeps can use `runBenchmarkBatch()` to evaluate several values in
one process. RSS is sampled every 10 ms after an idle baseline window and stored
both raw and baseline-subtracted.

That last invariant is deliberate: exploration evidence must enter the
confirmation workflow defined by ADR-282. This adapter never owns a repository
write token, opens a PR, or treats a smoke benchmark as publication evidence.

## Scoring and safety

Benchmark processes are spawned without a shell, receive only allowlisted,
bounded ANN parameters, inherit only `PATH`, have a wall-clock timeout, and
write to an isolated temporary directory. Invalid or missing reports fail closed.
Fitness uses the median over all observed runs and peak memory; it never selects
the best row from a report.

For promotable evidence, build with real dataset support and point the ADR-282
manifest dataset `source` at the absolute, pre-downloaded HDF5 file:

```bash
cargo build --release -p ruvector-sota-bench \
  --features real-datasets --bin sota-all
node dist/src/cli.js flywheel --repo "$(git rev-parse --show-toplevel)" \
  --manifest /absolute/path/research-manifest.json
```

The build requires the platform HDF5 development library and headers. Dataset
downloads are intentionally outside the candidate job; the harness consumes a
pre-positioned file and verifies its manifest hash before running.

The adapter verifies the dataset bytes against the manifest SHA-256 before
execution, verifies the canonical embedding-space identity, refuses synthetic
research results, and emits schema-shaped `research-manifest.json` and
`raw-results.json`. Run the trusted validator afterward:

```bash
python3 scripts/research-gate/research_gate.py evaluate \
  .metaharness/ruvector/research-manifest.json \
  .metaharness/ruvector/raw-results.json \
  --output .metaharness/ruvector/evaluation.json
```

The capability doctor executes probes for cost-aware routing, workspace
evidence, the deterministic control plane, capability-contained Red/Blue, and
Weight-EFT reward-hack detection. `capabilityVetoProvider()` can attach real
Workspace Lens receipts and archived trajectories; critical workspace drift,
live credentials, verification tampering, gold reads, and sandbox escapes are
hard promotion vetoes. These signals can only reject and never rescue weak
benchmark evidence.
