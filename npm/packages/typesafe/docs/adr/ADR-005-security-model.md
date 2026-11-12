# ADR-005: Security model — a bounded classifier, not an LLM host

## Status
Implemented and measured (2026-09-21)

## Date
2026-09-21

## Context

The engine takes untrusted text (`state`, criteria descriptions, examples) and
returns typed decisions. It is not an instruction-following model, so the
prompt-injection class that dominates LLM security does not apply in its usual
form — there is no completion loop to hijack. What remains is smaller but
real: an embedding classifier can be *steered* by adversarial phrasing, the
model weights are a supply-chain artifact, the native binding is an FFI
boundary, and the CLI/server are process boundaries. ruvector already runs a
five-layer supply-chain pipeline (`supply-chain.yml`: dependency-review,
cargo-audit, cargo-deny via `deny.toml`, npm-audit, lockfile-integrity), a
default-deny MCP tool policy (ADR-256), and a test that asserts the MCP server
never shells out (`test/mcp-command-security.js`).

## Decision

### Threat model

| Threat | Vector | Severity | Mitigation |
|---|---|---|---|
| Criteria/state steering | phrasing that shifts similarity toward an attacker-favoured option (e.g. stuffing `state` with another option's examples) | medium | abstain bucket + similarity-margin threshold (ADR-003) so a near-tie yields low `confidence`, not a confident flip; adversarial regression set in CI (ADR-006) |
| Tampered or substituted model weights | `.onnx` replaced in transit or on disk | high | content-hash-pinned **model manifest**, verified at load; release artifacts signed with cosign/sigstore keyless; manifest entries carry a re-review date like `deny.toml` ignores |
| FFI memory-safety defect | crafted input reaches `unsafe` at the napi-rs boundary | high | minimise `unsafe`; fuzz `decide()` inputs (cargo-fuzz) and run under ASan in CI; length limits enforced before the boundary |
| Command injection via CLI/serve | `state`/criteria interpolated into a subprocess | high if present | no subprocesses in the decision path at all; any helper uses `execFileSync` with argument arrays, never `shell: true` — enforced by a test cloned from `mcp-command-security.js` |
| Over-broad WASM capabilities | WASI filesystem/network imports added "for convenience" | medium | the wasm32 build links **no** WASI fs/net imports; CI inspects the import section (`wasm2wat` / `wasm-objdump`) and fails on any |
| Data egress | telemetry, model download at runtime | high (breaks the product promise) | **no network by default**: models ship bundled or from a local manifest path; CI greps the built native binary and the WASM bundle for `fetch`/`http` symbols in the decision path, mirroring the regex-against-artifact pattern in `mcp-command-security.js` |
| PII in logs | `state` text logged verbatim | medium | `state` is never logged; receipts store hashes and lengths, not text; debug logging redacts unconditionally |
| Dependency CVE / yanked crate | transitive dependency | low–medium | existing `cargo-deny`, `cargo-audit`, `npm audit`, lockfile-integrity gates reused as-is |

### Input limits (rejected with a typed error, never silently truncated)

- `state` ≤ 16 KB (embedding models truncate at 256–512 tokens; a longer state
  would be silently cut — better to say so).
- ≤ 255 options per `choice`; each description ≤ 1 KB; ≤ 10 `examples` per
  option, each ≤ 512 B.
- ≤ 64 questions per request.

### Trust boundaries

- **Tier C (untrusted, never authoritative):** `state`, criteria, examples,
  labels supplied at request time. They influence a *decision*; they never
  influence a *promotion* (ADR-004) except through the tier-A/B gate.
- **Tier B (quarantined):** LLM-judge labels, if a deployment enables them.
- **Tier A (authoritative):** programmatic verifiers — frozen-split accuracy,
  parity tests, the sequential test.

### Supply chain and release

- Model manifest: `{name, sha256, dims, license, source_url, added, review_by}`
  per weight file; load fails closed on mismatch.
- `npm publish --provenance` for `@ruvector/typesafe` and its platform packages;
  cosign signatures on the model manifest and the release tarballs; SBOM
  (CycloneDX) attached to each release.
- `deny.toml` and `supply-chain.yml` apply unchanged; the new crates add no new
  license classes.

## Consequences

- Runtime behaviour is deliberately boring: no downloads, no shell, no network,
  no logging of inputs. Convenience features that need any of those (model
  auto-download, remote judges) are opt-in flags with their own CI assertion
  that the default build does not contain them.
- Fuzzing and sanitizer runs add CI time; they are the price of a native FFI
  on untrusted input.
- Steering resistance is bounded: an adversary who can write the criteria can
  always define a bad classifier. The mitigation is that the *confidence*
  degrades honestly, which is a property the tests can check.

## Alternatives considered

- **An LLM-based injection filter in front of the classifier.** Adds a network
  or heavy local model to a path whose whole point is to avoid one, and the
  literature on LLM-as-detector is weak ("How Not to Detect Prompt Injections
  with an LLM", ACM 2025). Rejected.
- **Runtime model download by default.** Convenient, but it is exactly the
  egress the product promises not to have. Manifest-pinned bundle instead.

## Evidence

- Embedding-classifier injection detection (~97.7 % with tree classifiers on
  MiniLM embeddings): arXiv:2410.22284; CEUR-WS Vol-3920 paper 15;
  Springer doi:10.1007/s10207-026-01264-8. LLM-as-detector weaknesses: ACM
  doi:10.1145/3733799.3762980.
- Sigstore/cosign for blobs and SBOMs: docs.sigstore.dev (cosign signing other
  types); Chainguard "how to sign an SBOM with cosign".
- ruvector: `.github/workflows/supply-chain.yml`, `deny.toml`, `SECURITY.md`,
  `npm/packages/ruvector/test/mcp-command-security.js`, `mcp-policy.js` (ADR-256).
