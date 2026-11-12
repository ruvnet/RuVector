# ADR-282: Pre-PR Quality Gate for Nightly “Dream” Research

- **Status**: Proposed
- **Date**: 2026-07-29
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-265, ADR-267, ADR-271
- **Tags**: research, nightly, dream, benchmarks, reproducibility, ci

## Context

Nightly research and automated “dream” cycles are valuable discovery
mechanisms. They explore algorithms quickly, produce runnable proof-of-concept
crates, and connect ideas across the RuVector ecosystem.

The current process optimizes for generation throughput rather than review
quality. A candidate can open a draft pull request before:

- equal-budget controls have been established;
- results have been repeated across seeds;
- a real embedding dataset has been exercised;
- the implementation uses the production topology named in the claim;
- memory includes container, allocator, index, metadata, and scratch costs;
- Clippy and all relevant tests are green; or
- duplicate crates, ADR numbers, and prior work have been reconciled.

This creates a large queue of red or conflicting draft PRs and transfers
basic experimental validation to reviewers. More importantly, a benchmark
can be internally reproducible while still failing to support its headline
claim. Examples include comparing methods at different memory budgets,
calling an exact k-NN graph “HNSW,” or reporting encoded payload bytes as
total allocated memory.

ADR-267 defines publication-grade validation tiers. This ADR governs the
earlier boundary: whether an automated research candidate is ready to open a
pull request at all.

## Decision

Nightly research automation must pass a pre-PR research-quality gate. Failed
candidates retain local or workflow artifacts but do not open a pull request.

### 1. Candidate lifecycle

```text
idea
  -> prior-art and collision scan
  -> implementation
  -> experimental manifest
  -> methodology gate
  -> scoped CI on candidate branch
  -> PR opened
  -> normal repository review and CI
```

The automation may push a `research/candidate/**` or
`research/nightly/**` branch to run a branch-triggered workflow. It may not
call the PR creation API until all required preflight checks succeed.

On failure, the workflow publishes logs, raw measurements, and a concise
failure report as workflow artifacts. It does not create a placeholder draft
PR. A human may explicitly override the gate only by recording the failed
criteria and rationale in a manually opened RFC; the automation itself has
no bypass.

### 2. Treat candidate code as untrusted

The research candidate workflow executes generated code in a containment
boundary with:

- `contents: read` and no repository, issue, pull-request, package, or
  deployment write permission;
- no repository, organization, environment, cloud, package-publishing, or PR
  creation secrets;
- an ephemeral GitHub-hosted runner or a disposable isolated worker destroyed
  after one job, never a persistent privileged runner;
- no Docker socket, host mounts, device access, or privileged containers;
- CPU, memory, process-count, disk, output-size, and wall-clock limits;
- dependency hydration performed by a trusted fetch-only step from validated
  lockfiles with lifecycle/install scripts disabled;
- network disabled before any candidate-controlled build script, test, or
  binary runs where practical, otherwise an explicit destination allowlist
  and connection/byte ceilings;
- read-only dependency caches and no cache-save permission; and
- the workflow definition loaded from trusted default-branch automation, not
  from candidate-controlled YAML.

Secret scanning and changed-file inspection occur before any candidate build
script, test, benchmark, or binary executes. They are defense in depth, not a
substitute for containment.

PR creation occurs in a separate trusted promotion workflow. The promotion
workflow:

1. never executes or imports candidate code;
2. receives the minimal pull-request write token only after candidate jobs
   finish;
3. verifies a provenance attestation covering the exact candidate commit,
   trusted workflow revision, manifest hash, evaluator version, result
   artifact hashes, and complete required-check outcome;
4. checks that the attested commit still equals the branch head; and
5. opens exactly one PR for that immutable commit.

Attestation is produced by a separate trusted job that does not expose its
OIDC/signing capability to candidate processes. The candidate job never has
access to the promotion token, signing identity, or attestation credentials.

### 3. Required experimental manifest

Every candidate includes a machine-readable
`research-manifest.json` beside its report:

```json
{
  "schema_version": 1,
  "commit": "<full sha>",
  "revision": 1,
  "phase": "exploration|confirmation",
  "claim": "<falsifiable primary claim>",
  "independent_variable": "<one intentional difference>",
  "decision_rule": {
    "primary_metric": "recall_at_10",
    "minimum_meaningful_effect": 0.01,
    "expected_direction": "greater",
    "alpha": 0.05,
    "comparison": "paired",
    "outcomes": ["pass", "fail", "inconclusive"]
  },
  "datasets": [
    {
      "name": "<public dataset>",
      "source": "<stable source>",
      "sha256": "<content hash>",
      "sampling": "<deterministic rule>"
    }
  ],
  "embedding_space": {
    "identity": "<complete ADR-281 EmbeddingSpaceIdentity>",
    "embedding_space_id": "<sha256>",
    "query_api": "embed_query",
    "passage_api": "embed_passage"
  },
  "exploration_seeds": [1, 2, 3, 4, 5],
  "confirmation_seeds": [101, 102, 103, 104, 105],
  "topology": {
    "claimed": "hnsw",
    "implemented": "hnsw",
    "configuration": {}
  },
  "budget": {
    "primary_resource": "resident_memory_bytes",
    "tolerance": 0.005,
    "secondary_resources": []
  },
  "metrics": {},
  "memory_accounting": {},
  "environment": {},
  "commands": [],
  "evaluator_version": "<immutable revision>",
  "artifact_retention_class": "candidate|accepted"
}
```

Raw per-run results are retained. The report is generated from those results
or validated against them; headline numbers may not be manually transcribed
without an automated consistency check.

### 4. Primary-resource budgets and Pareto controls

Every experiment preregisters one primary constrained resource appropriate to
its claim, such as resident memory, on-disk bytes, query latency, search
effort, build time, or CPU-seconds. Treatment and control match that resource
within the declared tolerance.

All other material resources are measured and reported as secondary outcomes.
They are not required to be equal simultaneously. When algorithms have
structurally different cost profiles, the candidate reports a Pareto frontier
over the primary outcome and relevant secondary resources instead of forcing
an artificial all-resource match.

For a memory-constrained claim, the constrained quantity includes payload,
codebooks, graph/index data, metadata, container/per-entry bookkeeping, heap
capacity, and allocator-visible allocations. The default tolerance is the
larger of 0.5% or one indivisible allocation unit. If the algorithm cannot hit
the target, it reports bracketing configurations or a Pareto curve and does
not use “same memory budget.”

Selection logic must choose exactly the configured count. Percentile
thresholds with ties require deterministic tie-breaking rather than silently
allocating extra high-precision entries.

### 5. Preregistered decisions, confirmation, and uncertainty

Five seeds are a minimum execution floor, not evidence of adequate power.
Before execution, the manifest preregisters:

- one primary metric;
- the minimum practically meaningful effect;
- expected direction;
- confidence/error threshold;
- paired or otherwise justified comparison method;
- sample-size or power rationale; and
- exact `pass`, `fail`, and `inconclusive` rules.

Stochastic comparisons use paired seeds and paired observations wherever the
same dataset/query workload permits it. If the observed interval overlaps
both zero and the minimum meaningful effect, the result is inconclusive, not
positive.

Reports include:

- every per-seed result;
- mean and standard deviation;
- a 95% confidence interval or a justified non-parametric interval;
- the worst seed; and
- effect size against the control.

Nightly runs are exploratory. A promising exploration cannot open a PR until
it passes a separate confirmation run using a held-out dataset or fresh,
predetermined confirmation seeds that were not used for tuning. Confirmation
is one-shot against a frozen commit, configuration, manifest, evaluator, and
decision rule. Changing the claim or tuning after confirmation begins creates
a new manifest revision and requires another held-out confirmation.

When one nightly cycle explores multiple candidates, the manifest records the
candidate family and selection rule. Confirmation uses held-out evidence and
either controls the family-wise/FDR error rate or labels the outcome
exploratory. Only confirmatory outcomes may support automatic PR creation.

Deterministic algorithms still run enough repetitions to characterize timing
noise when making latency or throughput claims.

### 6. Real embeddings and representative data

Synthetic data may explain mechanisms and exercise edge cases, but it cannot
be the sole evidence for a production retrieval claim.

Before a PR opens, the candidate must run at least one pinned, redistributable
or reproducibly downloadable real dataset appropriate to the claim. Text
retrieval experiments use a real embedding model with:

- exact weights and tokenizer identity;
- the complete ADR-281 `EmbeddingSpaceIdentity` and
  `embedding_space_id`, including prompt/prefix policy and version, pooling,
  normalization, truncation, dtype, runtime revision, and distance metric;
- correct query/passage role APIs under ADR-281;
- cached/offline reproducibility instructions.

If licensing prevents redistribution, the manifest records the stable source,
hashes, and deterministic preparation procedure.

### 7. Real production topology

The experiment must use the topology named in its claim.

- An HNSW claim uses the repository's HNSW construction and traversal.
- A DiskANN claim uses the DiskANN/Vamana path.
- A COW or RVF claim uses the real persistence and reopen path.
- A WASM claim compiles and runs under a WASM target/runtime.
- A GPU, NPU, or SIMD claim exercises the named backend or is explicitly
  labeled a simulation.

Simplified exact k-NN, brute-force, or toy graphs are allowed only when the
claim and title name them accurately. They cannot be used as evidence about
production topology without a separate production-topology experiment.

The control and treatment share the same production path except for the
independent variable named in the manifest.

### 8. Full memory accounting

“Memory” without a qualifier means peak resident memory for the measured
operation plus a component breakdown. At minimum the report distinguishes:

```text
encoded payload
index/graph
codebooks and quantizer metadata
container and per-entry bookkeeping
heap capacity/allocator overhead
temporary build memory
temporary query memory
process peak RSS
```

Payload-only measurements are allowed but must be labeled
`encoded_payload_bytes`; they cannot support a total-memory saving claim.

Rust in-memory structures use allocator instrumentation, heap profiling, or a
validated structural calculation that includes `Vec`, `Option`, capacity,
and per-allocation overhead assumptions. File-format claims measure actual
file size after flush and reopen. WASM claims measure linear-memory pages and
peak growth.

Peak-RSS measurements follow a versioned operational protocol recorded in the
manifest:

1. run treatment and control in fresh, single-purpose processes inside
   equivalent cgroups or job containers;
2. record OS/kernel, allocator and version, cgroup memory limit, page size,
   process/thread count, and child-process policy;
3. measure an empty-harness baseline under the same runtime and report both
   raw peak and baseline-subtracted peak;
4. declare warmup operations and begin measurement only after warmup;
5. sample RSS at no more than 10 ms intervals and also record kernel/cgroup
   high-water marks when available;
6. state whether filesystem page cache is included, and keep cold/warm cache
   state identical across paired runs;
7. terminate and account for all child processes before accepting a run; and
8. repeat paired runs in alternating order to reduce thermal and ordering
   bias.

The protocol version is part of result identity. Results produced under
different memory-protocol versions are not directly pooled without an
explicit comparability analysis.

### 9. Pre-PR CI must be green

The branch-triggered `research-candidate` workflow includes:

1. ADR and crate/package name collision detection;
2. changed-file and secret scanning;
3. `cargo fmt --check`;
4. `cargo clippy --all-targets -- -D warnings` for affected crates;
5. unit, integration, reopen/persistence, and target-specific tests;
6. the research benchmark and its acceptance evaluator;
7. manifest/schema validation and report/result consistency;
8. dependency, license, and advisory checks for new dependencies; and
9. the repository's required regression guards for touched subsystems.

Warnings are failures. Cancelled jobs are not green. A known-red `main`
blocks automated PR creation until it is repaired or the candidate proves,
through a recorded base-versus-head differential run, that the failure is
pre-existing and an authorized human approves proceeding.

An override requires approval through the protected
`research-gate-override` GitHub Environment by a repository member with
`maintain` or `admin` permission who is also listed in the research-gate
CODEOWNERS group. The workflow records an immutable `override.json` containing
the approver, exact base and head SHAs, failed check identities, rationale,
approval timestamp, and expiry. Approval expires after 72 hours, applies to
one candidate revision, and becomes invalid if either SHA or check result
changes. Issue comments, PR comments, or free-form labels alone are not
authorization.

Normal PR CI still runs after creation. Preflight success is necessary, not a
replacement for merge checks.

### 10. Prior-art, collision, and scope gate

Before implementation, automation searches:

- open and merged pull requests;
- issues and existing ADRs;
- workspace package and crate names;
- existing modules with equivalent behavior; and
- current nightly research branches.

The candidate extends an existing crate when practical. A new crate requires
a short justification in the manifest. ADR numbers are allocated uniquely
from current `main`; duplicate ADR numbers fail preflight.

One PR carries one primary falsifiable claim. Broad ecosystem connections and
future applications belong in discussion, not as untested claims.

### 11. Claim language is evidence-bounded

The gate rejects or requires qualification for:

- “production-ready” without production-path tests;
- “only,” “first,” or “SOTA” without a documented comparison search;
- extrapolated 10–20 year predictions presented as measured results;
- performance numbers without hardware and command provenance;
- memory savings based only on encoded payload;
- recall improvements from unequal budgets; and
- target support inferred from portable source without compiling/running the
  target.

Reports separate measured results, inferences, hypotheses, and roadmap items.

### 12. Reproducibility artifact integrity and retention

Every manifest, raw result, report, evaluator binary/package, environment
description, and attestation has a SHA-256 digest in an immutable artifact
index. The index records schema version, evaluator version, workflow
revision, candidate commit, creation time, and retention class.

- Failed/exploratory candidates are retained for at least one year.
- Candidates that open a PR are retained for at least seven years.
- Publication/SOTA artifacts follow ADR-267 and are retained permanently or
  under its archival policy.

Artifacts are stored in versioned, write-once or object-lock storage. A
workflow URL is navigation only; it is not the durable evidence record.
Deletion before expiry requires a repository-security incident record and
leaves a signed tombstone in the artifact index.

## Pull Request Contract

A passing candidate opens a PR containing:

1. the falsifiable claim and independent variable;
2. immutable manifest, raw-result, evaluator, and attestation hashes;
3. the primary constrained-resource comparison and secondary-resource/Pareto
   report;
4. preregistered decision rule, confirmation outcome, per-seed results, and
   uncertainty;
5. real-dataset and production-topology evidence;
6. full memory breakdown and memory-protocol version;
7. exact reproduction commands;
8. attested preflight workflow revision and commit SHA; and
9. known limitations, exploratory selection history, and failed experiments.

The PR may be draft for architectural discussion, but it must already be
green and scientifically reviewable.

## Acceptance Criteria

1. Candidate code runs without secrets, write permissions, persistent
   privileged runners, promotion tokens, or signing credentials; containment
   limits terminate resource abuse.
2. The trusted promoter refuses a missing, invalid, stale, or SHA-mismatched
   attestation and never executes candidate code.
3. A candidate outside its primary-resource tolerance cannot claim an
   equal-primary-budget result; structurally different costs require
   secondary-resource reporting or a Pareto frontier.
4. A percentile tie that changes allocation count fails deterministic-budget
   validation.
5. A stochastic claim without a preregistered meaningful effect, direction,
   error threshold, paired method, power rationale, and
   pass/fail/inconclusive rule cannot enter confirmation.
6. An exploratory winner without held-out data or fresh predetermined
   confirmation seeds cannot open a PR; multiplicity/selection history is
   recorded.
7. Synthetic-only evidence fails a production retrieval claim.
8. A toy graph fails an HNSW/DiskANN topology claim.
9. Payload-only accounting or an incomplete peak-RSS protocol fails a
   total-memory claim.
10. Missing or mismatched `EmbeddingSpaceIdentity`,
    `embedding_space_id`, role APIs, prefix/prompt policy, normalization, or
    distance metric fails an embedding benchmark.
11. Any warning, failed, timed-out, or cancelled required preflight check
    prevents automated PR creation unless an exact-SHA, non-expired protected
    environment override satisfies this ADR.
12. Duplicate ADR or crate/package names fail before implementation is
    proposed for review.
13. The report's headline values are automatically verified against hashed
    raw artifacts using the recorded schema and evaluator versions.
14. Artifact retention tests verify immutable hashes and the one-year,
    seven-year, or permanent retention class.
15. A passing fixture demonstrates that the trusted promoter opens exactly
    one PR for the attested commit SHA.
16. A failing fixture demonstrates that artifacts are retained but no PR is
    opened.
17. Given one corpus, changing only the query template while retaining the
    model ID produces a new `embedding_space_id`, invalidates cache reuse and
    corpus mutation, preserves vector-only reads, and forces a new
    experimental revision and confirmation run.

## Consequences

### Positive

- Reviewers receive evidence-ready research instead of unfinished generation
  output.
- Nightly throughput is converted into a smaller, higher-signal PR queue.
- Headline claims remain tied to budgets, topology, and actual memory.
- Failed ideas still produce useful artifacts without consuming review
  bandwidth.
- Publication-tier validation in ADR-267 starts from cleaner inputs.

### Negative

- Fewer nightly candidates will become pull requests.
- Real datasets, repeated seeds, and production topology increase compute
  cost and cycle time.
- Branch-triggered preflight requires artifact storage and workflow
  orchestration.
- Untrusted execution containment, confirmation runs, attestations, and
  immutable retention add infrastructure and compute cost.
- Some exploratory ideas will remain local until a human invests in the
  missing evidence.

## Alternatives Considered

- **Open every result as a draft and let reviewers decide**: rejected because
  the open queue becomes the validation system and red drafts obscure strong
  work.
- **Require only green unit tests**: rejected because code correctness does
  not establish experimental validity.
- **Use synthetic data for speed**: retained for smoke/mechanism tests but
  rejected as sole evidence for production claims.
- **Defer real topology and memory accounting to a later PR**: rejected when
  the initial PR's headline claim depends on them.
- **Run the gate only after PR creation**: rejected because the purpose is to
  prevent low-signal PR creation, not merely label it afterward.
- **Give the candidate workflow a PR token after its tests pass**: rejected
  because generated code could exfiltrate or misuse a credential from the
  same execution boundary. Promotion is a separate trusted workflow.
- **Treat five exploratory seeds as confirmation**: rejected because low
  power and winner selection across many nightly candidates create
  multiplicity and false-discovery risk.

## Implementation Surfaces

- `.github/workflows/research-candidate.yml`
- `.github/workflows/research-promote.yml`
- `scripts/research-gate/`
- `docs/research/nightly/<date>-<topic>/research-manifest.json`
- JSON Schema for manifests, raw results, overrides, artifact indexes, and
  attestations
- `schemas/embedding-space-identity-v1.json` from ADR-281
- immutable artifact storage and retention policy
- PR-generation automation used by nightly/dream workflows
- ADR/crate/package collision checks against current `main`
