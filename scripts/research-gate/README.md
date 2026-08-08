# ADR-282 research gate

This directory is the trusted implementation of the pre-pull-request gate.
Candidate code is evaluated in a separate contained process; these scripts
validate its data and never import candidate modules.

Third-party dependencies are declared in `requirements.in` (`jsonschema` for
validation, `referencing` for the offline schema registry) and compiled into
`requirements.txt`, a fully hashed lock covering the whole transitive closure.
The trusted jobs install it with `--require-hashes` on a pinned interpreter, so
a substituted artifact fails the install instead of quietly changing what the
gate accepts. Regenerate the lock after editing `requirements.in`:

```bash
uv pip compile scripts/research-gate/requirements.in \
  --generate-hashes --universal --python-version 3.12 \
  -o scripts/research-gate/requirements.txt
```

`--universal` is required: it records hashes for every platform's wheels, so the
same lock installs on the Linux runners and on a developer machine.

```bash
python3 -m pip install --require-hashes -r scripts/research-gate/requirements.txt
python3 scripts/research-gate/research_gate.py validate-manifest \
  research-manifest.json --expect-sha "$CANDIDATE_SHA"
python3 scripts/research-gate/research_gate.py evaluate \
  research-manifest.json raw-results.json --output evaluation.json
python3 scripts/research-gate/research_gate.py validate-report \
  report.json --manifest research-manifest.json --evaluation evaluation.json
python3 -m unittest discover -s scripts/research-gate/tests -v
```

Every document the gate reads or writes is validated against its schema in
`schemas/` before any semantic check runs. Schemas are loaded from disk and
registered under their own `$id`, so `$ref` resolution is fully offline and a
candidate cannot point validation at a network-hosted schema. The hand-rolled
checks that follow enforce the cross-field invariants a schema cannot express.

The evaluator requires paired confirmation runs over the exact preregistered
seed list. It enforces one primary resource budget, full memory accounting,
deterministic selection counts, real-data/production-topology declarations,
and a canonical ADR-281 embedding-space identity.

A candidate's `report.json` is hashed into the attested artifact index, so it
is validated rather than merely required to exist. `validate-report` checks it
against `schemas/research-report-v1.json` and then rebinds every headline value
to the trusted evaluation of the hashed raw results (ADR-282 acceptance
criterion 13). Report headline values must be the evaluator's values, not a
re-rounded restatement of them; round only in prose.

The artifact index must account for the entire evidence tree. Validation walks
the evidence root and rejects any file, nested payload, or symlink that has no
recorded digest, so nothing rides into the attested bundle unhashed. Only
`artifact-index.json` (which cannot contain its own digest) and, during
promotion, the signed `attestation-subject.json` are exempt. A candidate that
legitimately needs to retain another artifact gets it added to the trusted
index in this workflow — not added to the exemption list.

Base health is read with `gh api --paginate`. This repository routinely has
more than one page of check runs per commit, and `base_health.py` cross-checks
the entries it sees against the reported `total_count`: a truncated or
inconsistent response fails closed instead of certifying a red base as green.

The preflight scan diffs two independent depth-1 checkouts, so the workflow
first copies the base commit into the candidate clone from the sibling `base`
checkout — no network, no credentials, and no full-history clone. Deletions are
in scope for the scan, because removing a lockfile is as much a dependency
change as editing one.

The candidate workflow has read-only repository permission and no secrets.
Generated code runs in a networkless, resource-limited, unprivileged
container. A separate trusted job creates the GitHub artifact attestation,
and a separate default-branch promotion workflow owns the pull-request token.
The attestation job is dependency-gated on both preflight and candidate jobs;
it records their actual `needs.<job>.result` values. The candidate job can
only succeed after its sequential scoped CI, methodology validator, raw-result
consistency check, and confirmation evaluator all succeed, so the attested
contained-gate outcome is mechanically tied to those steps rather than a
candidate assertion.

Trusted nightly orchestration should call `.github/workflows/research-candidate.yml`
through `workflow_call` with an immutable branch head SHA. Manual execution
uses the identical `workflow_dispatch` input contract. A candidate-controlled
workflow must never be the caller.

Configure the `research-gate-override` protected Environment with required
reviewers from `@ruvnet/research-gate`, enable “Prevent self-review,” and set
the repository variable `RESEARCH_GATE_CODEOWNERS` to the comma-separated
GitHub logins of that team’s current members. The override workflow reads the
GitHub environment review history, records the actual approving reviewer,
requires that reviewer to have `maintain` or `admin`, and checks the reviewer
against this mirrored CODEOWNERS membership list. Comments and labels are not
accepted as overrides.

The exception is deliberately narrow. A trusted preflight queries check runs
and commit statuses for the exact current `main` SHA. A separately attested
override may convert only those `base/...` failures to `authorized-red`; it
cannot alter candidate containment, scoped CI, methodology, confirmation,
artifact, or attestation outcomes. The override expires within 72 hours and
is revalidated during promotion against the exact base SHA, candidate SHA,
failure set, and branch head.

The nightly (03:17 UTC) and manually runnable, default-branch
`research-nightly-dispatch` workflow discovers immutable heads under
`research/candidate/**` and `research/nightly/**`, de-duplicates them by
ref/SHA, resolves the current base SHA, and dispatches the named
default-branch `research-candidate` workflow. Candidate branches never supply
the dispatch workflow definition or its Actions token. The named run can
then trigger `research-promote`. If the base is red,
the first run fails closed, a reviewer creates a protected override, and a
trusted operator reruns `research-candidate` with that override workflow run
ID.

GitHub artifact retention is a delivery cache, not the durable evidence
store. Production deployment must copy indexed artifacts to versioned
write-once/object-lock storage for the schema's 365-day, 2555-day, or
permanent retention class.
