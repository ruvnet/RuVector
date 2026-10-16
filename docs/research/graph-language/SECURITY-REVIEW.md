# RuGraph scoped security review

Date: 2026-10-16. Scope: the two new Rust crates, SDKs, CLI, their boundaries,
and the additive GraphDB visitor API. Authorized work: implementation, local
read only scanning and correction of defects in the new implementation. No
production systems, external provider credentials or real user data were used.

## Trust model

| Boundary | Threat | Mitigation and residual responsibility |
|---|---|---|
| Untrusted text/JSON → program | Type confusion, duplicate fields, excessive input | Strict parsing, version checks, one representation, hard size/depth bounds |
| Program → host authority | Self granted permissions or arbitrary effects | Host grants checked separately; only three effects; no general I/O operations |
| Snapshot → runtime | Dangling references, oversized properties, excessive derivation | Validation, work and result limits, bounded provenance, fixed point over admitted facts |
| GraphDB → snapshot | Allocation before checking; inconsistent concurrent reads | Borrowed bounded visitors; caller must exclude every writer |
| Predictor → language | Mislabelled scores, malformed output, excessive results | Typed isolation, finite/identity/count checks, native preflight; custom predictors are trusted code |
| Model file → native scorer | Invalid shape, nonfinite weights, forged claimed metrics | Strict artifact bounds and native validation; host trusts producer or reruns evaluation |
| SDK → local executable | Shell injection, output flood, indefinite process | Argument arrays, explicit paths, timeout and bounded pipe draining |
| Receipt → consumer | Tampering or treating a checksum as authenticity | Checksum verification and full replay; authentication is explicitly outside scope |

## Confirmed fixes and validation

The independent review confirmed oversized graph capture and native prediction
allocation paths. Both were fixed at the boundary before copying or accumulating
payloads. Regression tests and the repeated native memory reproducer pass.
JSON roundtrip, request validation, nested canonical hashing, Python output
draining, JavaScript numeric precision and CLI preflight defects were also fixed
and covered by executed tests. No unresolved high or critical finding remains
within this focused code review scope.

The SDK executable and model paths are trusted host configuration. The clients
are not OS sandboxes for arbitrary hostile binaries. A trusted custom Rust
Predictor can perform work outside the interpreter's accounting; only the
provided native implementation preflights its internal scoring work.

## Tools and coverage

Read only scans used the validated `@claude-flow/cli@3.25.6` runtime:

```bash
npx --yes @claude-flow/cli@3.25.6 security scan \
  --target crates/ruvector-language --depth deep --type all
npx --yes @claude-flow/cli@3.25.6 security scan \
  --target crates/ruvector-language-native --depth deep --type all
npx --yes @claude-flow/cli@3.25.6 security scan \
  --target python/ruvector-language --depth deep --type all
```

The npm package was copied into an isolated temporary directory, its empty
dependency lockfile generated with scripts disabled, and both `npm audit --json`
and the same Ruflo scan run from that directory. This avoids auditing the
monorepo parent's unrelated npm dependency tree. The SDK declares no runtime or
development dependencies; the isolated audit reported zero dependencies and zero
advisories. Scoped Ruflo scans reported no findings. Scanner output is supporting
evidence, not a proof of safety.

Scan date and npm advisory retrieval date: 2026-10-16. The service did not expose
an advisory database snapshot timestamp, so an exact feed revision is unknown.
The initial feature review did not include Cargo transitive advisory auditing.
A subsequent CI review identified an existing root dependency advisory gate
failure. The root rustls lock entry is updated to the patched 0.23.46 release,
with unchanged registry dependency requirements, features and minimum Rust
version. The downloaded archive checksum and compilation were verified.
The Graph language workflow now audits both standalone lockfiles with
`cargo-audit` 0.22.2; existing root jobs audit the root lockfile. Passing the
functional acceptance suite alone does not establish advisory clearance.
Python build tools and the host runtime are outside the zero runtime dependency
claim. Other root repository vulnerabilities are outside this feature review.

Rust lockfile hashes, compiler versions, full acceptance commands and results
are recorded in [ACCEPTANCE.md](ACCEPTANCE.md). No secret values are included in
this report or the committed fixtures.

## Counterfactual audit extension, 2026-10-16

Threat: a caller requests repeated evaluations or oversized reports, disguises
missing targets as robustness, supplies ambiguous JSON, or triggers model/provider
effects during an audit. Mitigations: 64 trial cap, aggregate worst-case execution
reservation capped at one billion, validated snapshots before copying, incremental
16 MiB report accounting, missing/duplicate target rejection, strict CLI JSON,
normal capability checks, and rejection of every prediction step. Audit overhead
outside the runtime meter is bounded structurally, not billed as runtime work.

SDK options are written into private temporary files and deleted on both success
and error. Tests check file permissions and cleanup, real process execution and
aggregate budget errors. Audit reports contain caller-supplied evidence identities
and must be handled with the same access policy as ordinary receipts.

Read-only Ruflo scan command: `npx -y @claude-flow/cli@3.25.6 security scan --target
crates/ruvector-language --depth deep --type all`. Completed with zero findings.
Manual boundary review and regression tests supplement scanner coverage. No new
dependencies or lockfile changes; this scan is not a refreshed advisory-feed audit
and does not prove absence of vulnerabilities. Source identity authenticity,
source independence and real-world causal interpretation remain outside scope.

## Prepared snapshots

The additive Rust API borrows an immutable snapshot; private cached fields cannot
be constructed or replaced by the caller. Safe Rust prevents mutation and use
after free while it exists. It retains no caller capabilities, clock, decisions
or model responses. Every execution rechecks program/context and provider calls.
Cached validation/index work is charged again, preserving low-budget rejection
and receipt accounting. Tests cover malformed snapshots, capability revocation,
invalid context, exhausted budgets and repeated provider invocation. No unsafe
code or new I/O boundary is introduced. Index memory is retained until the host
drops preparation; avoid retaining unbounded numbers of prepared snapshots.

## Exact selection and borrowed IDs

The optimizer changes only internal node ownership and physical sorting. It
introduces no unsafe code, dependency or external boundary. Every vector is still
validated before selection; malformed tail candidates cannot escape checks. Public
evidence, receipts and predictor requests remain owned. Borrowed IDs refer to the
immutable snapshot, not transient program strings. Historical logical work and
byte accounting are preserved. Full-sort oracle tests cover ties and extreme
finite values; real native and SDK boundaries remain in the acceptance gate.
