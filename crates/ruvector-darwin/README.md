# ruvector-darwin

Bounded, deterministic evolutionary parameter search for RuVector research
candidates.

See the crate-level docs in `src/lib.rs` for the design rationale and hard
constraints, and `docs/research/nightly/2026-10-06_darwin-evolved-gnn-rerank-hyperparameters/`
for the nightly research run that introduced it and its first application.

```rust
use ruvector_darwin::{run_evolution, Bound, EvolutionConfig, Evaluation, Genome};

let parent = Genome::new(vec![0.60, 0.50]); // e.g. (alpha, coherence_threshold)
let bounds = vec![Bound::new(0.05, 0.95), Bound::new(0.0, 0.95)];
let report = run_evolution(parent, bounds, EvolutionConfig::default(), |g| {
    // Evaluate `g.genes` against your own held-out-safe fitness function.
    Evaluation::Fitness(g.genes[0] - g.genes[1])
});
if report.beats_parent() {
    println!("{}", report.to_json_pretty());
}
```
