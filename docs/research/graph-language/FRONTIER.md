# RuGraph research and acceptance frontier

Reviewed 2026-10-16. Goal: auditable agent memory decisions with reproducible
failure analysis. This is engineering informed by current research, not a claim
that RuGraph is state of the art across graph programming or AI reasoning.

## Research to implementation

| Primary source | Established direction | RuGraph action | Remaining gap |
|---|---|---|---|
| [Souffle provenance](https://souffle-lang.github.io/provenance) | Proof explanation is mature prior art | Preserve explicit witness lineage; distinguish witness replacement from changed truth status | No minimal proof tree or all proof enumeration |
| [Why Neighborhoods Matter, May 2026 preprint](https://arxiv.org/abs/2605.15109) | Ablations can expose dependence on graph context beyond final citations | Add edge, observation and source ablation with full recomputation | No full traversal trace or graph node masking |
| [GraphRAG-Bench](https://graphrag-bench.github.io/) | Evaluate graph retrieval and reasoning across task types | Independent application oracle, generated variations, raw samples | No public corpus QA benchmark or comparison against external systems |

## Runnable agent memory application

From the RuVector root, with Rust 1.90:

```bash
cargo run --release --locked --manifest-path crates/ruvector-language/Cargo.toml --example agent_memory
cargo run --release --locked --manifest-path crates/ruvector-language/Cargo.toml --example agent_memory -- request > memory-request.json
cargo run --release --locked --manifest-path crates/ruvector-language/Cargo.toml --example agent_memory_bench
```

The application ranks synthetic memory vectors, traverses related memories,
checks recent signed reliability observations, derives usability using positive
and negative rules, and audits the effect of removing either source or one edge.
It asserts decision equivalence with a direct evaluator before emitting results.
There is no LLM, API charge, model download, learned embedding quality claim or
personal information in the fixture. The vectors and source identities are synthetic.

## Counterfactual audit API

Rust: `ruvector_language::audit::run(&request, &options)`.
JavaScript: `await client.audit(request, options)`.
Python: `client.audit(request, options)`.
Both core and native CLIs accept:

```bash
rvg-core audit --request request.json --audit-options options.json
```

Example options for the bundled incident fixture:

```json
{
  "interventions": [
    {"kind": "observation", "id": "db-error-1"},
    {"kind": "source", "id": "telemetry-a"},
    {"kind": "edge", "id": "api-db"}
  ],
  "max_execution_work": 400000
}
```

`changes` lists only status/domain changes; `changed_decisions` also counts
witness changes. A null before/after means the subject/predicate is not in that
execution's decision domain. Unknown means it was assessed with neither sign.
A no-change result does not prove a source is irrelevant in other contexts or
joint interventions. Compare full audit reports to reproduce an audit; the
ordinary receipt `verify` command does not accept audit reports.

## Benchmark contract

The status-only handwritten baseline and RuGraph see identical graph, vectors,
observations and time. Their statuses must match. The baseline omits validation,
capability checks, resource budgets, provenance and hashing, so timing ratios
are feature-cost comparisons, not an equivalent-feature speed contest. Neither
path includes parsing, SDK startup or storage capture. The audit timing includes
one baseline and three full executions. Raw sorted samples and environment are
in [agent-memory-benchmark.json](agent-memory-benchmark.json). Reproduce on the
deployment host; shared container timings are noisy. RSS is not measured by this
harness. No code-reduction percentage is claimed.

## Acceptance trace

| Requirement | Executable evidence |
|---|---|
| Source sensitivity and redundant evidence | `tests/audit.rs` necessary/redundant/conflict cases |
| Graph context effects | edge deletion distinguishes absent domain from unknown |
| Independent decision semantics | 256 generated cases plus 64 oracle ablations |
| Canonical receipts | input reversal and repeated audit equality |
| Resource and authority bounds | invalid targets, trial caps, reservation limits, capability denial, prediction rejection |
| Process integration | CLI audit and JS/Python real binary tests |
| Private temporary files | SDK permission and cleanup tests on both paths |
| Performance transparency | three sizes, 5 warmups, 30 samples, raw data |

Next research gate: evaluate a real application dataset and compare an equivalent
handwritten implementation with the same governance features. Until then, retain
preview positioning. External benchmark leadership and research novelty remain
unestablished.

## Reuse without stale decisions

The new Rust `PreparedSnapshot` API reuses validation, indexes and the canonical
snapshot hash under an immutable borrow. It recomputes decisions with a fresh
context and preserves complete receipts and logical budgets. See
[ADR-GL-007](../../adr/ADR-GL-007-immutable-prepared-snapshots.md) and the
[paired measurements](prepared-specialized-benchmark.json). This addresses repeated
host-query cost; it is not an incremental engine or a cache of authority/model
outputs. The cold path and process SDK behavior remain semantically unchanged.

A rejected hash buffer experiment and first prepared candidate are retained with
the final results to make selection visible. Warm improvements do not establish
cold-path improvements, production service latency or superiority over other
languages. A dedicated-host performance run and real dataset remain the next
external validation gates.

## Exact selection and allocation reduction

Exact cosine ranking now partitions candidates and sorts only the top K. Runtime
node sets borrow immutable node IDs; owned data remains at public output/provider
boundaries. The logical work and retained-byte charges remain unchanged to preserve
receipts and limits. The paired two-dimensional fixture measured an additional
10.73% to 42.57% reduction in prepared latency. This does not predict performance
for high-dimensional embeddings or large K. See [raw paired evidence](selection-borrowing-benchmark.json)
and [ADR-GL-008](../../adr/ADR-GL-008-exact-selection-and-borrowed-node-sets.md).
