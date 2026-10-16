# ADR-GL-006: Bounded counterfactual evidence audits

Date: 2026-10-16
Status: Accepted

## Context and research

RuGraph receipts reproduce decisions and identify selected witnesses. A selected
witness is not necessarily indispensable: another observation can support the
same fact. Removing an edge can change the retrieval domain even when that edge
is absent from observation lineage. Operational users need to test these effects
before relying on agent memory decisions.

Souffle already provides proof explanations, including minimal height trees:
https://souffle-lang.github.io/provenance . Provenance is not a novel invention.
The 2026 preprint *Why Neighborhoods Matter* tests cited and uncited graph
ablations and argues that final citations alone miss retrieval dependencies:
https://arxiv.org/abs/2605.15109 . We adopt the experimental question, not its
models, dataset or reported performance. GraphRAG-Bench distinguishes retrieval
and reasoning tasks: https://graphrag-bench.github.io/ . Our deterministic
synthetic workload is not that benchmark or evidence of superiority over it.

## Decision

Add a pure host audit API and matching CLI and process SDK methods. A caller
supplies a normal RunRequest plus 1 to 64 explicit interventions. Each deletes
one observation, all observations with one source identifier, or one edge from
an independent copy of the original validated snapshot. No intervention is
cumulative. No original graph, artifact or observation is modified.

Run a baseline, validate all target identifiers, then recompute every intervention.
Compare decisions by subject and predicate. Report status deltas and the number
of complete decisions whose status or witness lineage changed. Null denotes a
missing decision domain entry, which is different from the status unknown.
Retain baseline and trial receipt hashes, snapshot hashes, context/program hashes,
record counts, actual work and reserved work. Repeating the audit is the report
verification procedure. Hashes are not authenticated signatures.

Reject any program containing prediction steps. A provider can have effects or
nondeterminism outside the interpreter; supporting it would weaken the contract.
Capabilities remain host grants. Runtime program and receipt wire version stays
1; the separate audit report schema starts at 1.

## Bounds and trust

At most 65 executions. Before execution, reserve `(trials + 1) * min(program
budget, host max_steps)` against max_execution_work, itself at most one billion.
This is a conservative execution reservation, not a wall clock or total CPU
budget. Snapshot copying, target lookup, comparison and report serialization
are outside the runtime meter but bounded by 100,000 snapshot records, 16 MiB
serialized input and 64 trials. Reports are bounded to 16 MiB with incremental
accounting. SDK process timeouts and output caps remain active.

Reject missing targets rather than treating a typo as robustness. Reject duplicate
interventions, unknown fields and duplicate JSON keys. Inputs are validated
before cloning. Private SDK temporary files have owner permissions and cleanup
on success and failure. No shells, network, model loading, or new dependencies.

Source identifiers are caller assertions, not authenticated independent sources.
A source ablation is conditional sensitivity under the supplied graph and rules,
not proof of real world causality or evidence completeness. Single interventions
do not enumerate minimal removal sets, Shapley values or interacting failures.

## Alternatives

1. Full minimal proof enumeration: potentially exponential; rejected for this
bounded version. Existing canonical witnesses remain unchanged.
2. LLM explanation: adds cost and nondeterministic unsupported claims; rejected
for the audit engine.
3. Incremental dependency invalidation: potentially faster, but correctness across
alternative proofs and graph retrieval requires a larger semantic change.
Full recomputation supplies a reference oracle for future incremental work.

## Acceptance and rollback

Tests cover necessary and redundant evidence, grouped source removal, conflicts,
expiry, missing versus unknown, deterministic report replay, bad targets, budgets,
permissions and prediction rejection. Generated reference cases compare 128 seeds
at two times against an independently implemented evaluator and reverse input
order to verify canonical receipts. Another 32 seeds test source and edge deltas.
CLI and both SDKs execute audits against real binaries. The synthetic application
benchmark records raw timing samples at 128, 1024 and 4096 nodes.

Rollback removes the additive audit module, command and SDK methods. Existing
programs, package names, CLI commands and receipt semantics are unchanged.
