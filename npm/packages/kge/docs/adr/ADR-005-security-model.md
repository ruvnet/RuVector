# ADR-005: Security model — a bounded scorer over untrusted triples

## Status
Implemented and measured (2026-09-21); optimize campaign pending

## Date
2026-09-21

## Context

The engine takes triples and text (entity/relation ids, descriptions, query
states) and returns plausibility scores. Like typesafe's classifier, it is not
an instruction-following model, so the LLM prompt-injection class does not apply
in its usual form. What remains is the typesafe threat surface (weights as a
supply-chain artifact, an FFI boundary, process boundaries) **plus two threats
specific to knowledge-graph embeddings**: the scores can be poisoned by decoy
triples elsewhere in the graph, and query access leaks training-set membership.

ruvector already runs the five-layer supply-chain pipeline (`supply-chain.yml`:
dependency-review, cargo-audit, cargo-deny via `deny.toml`, npm-audit,
lockfile-integrity) and a default-deny MCP tool policy (ADR-256), reused
unchanged.

## Decision

### Threat model

| Threat | Vector | Severity | Mitigation |
|---|---|---|---|
| **Targeted-fact poisoning via decoy triples** | symmetric / inverse / compositional decoy triples added elsewhere indirectly flip confidence on a target fact (Bhardwaj 2021) | high | trust-tier gate on triple admission (A/B/C, below); anomaly detection on symmetry-pattern concentration at ingest; **adversarial regression set in CI scored on confidence-drop, not flip** (ADR-006) |
| **Membership inference via query access** | repeated scoring queries let a query-only attacker infer training-set membership (Wang & Sun) | medium (high for PII/medical graphs) | rate-limit + log query patterns; never expose raw per-triple confidence deltas; opt-in sparsity-preserving DP-SGD for sensitive deployments |
| Tampered/substituted embedding table | entity/relation table replaced in transit or on disk | high | content-hash-pinned model manifest, verified at load, fails closed; cosign/sigstore-signed artifacts — identical to typesafe ADR-005 |
| Malicious triple injection at ingest | untrusted upstream (scraped/federated/user-submitted) source | medium–high | same Tier A/B/C gate; neighbourhood-agreement filtering of noisy triples, quarantine on disagreement |
| Entity-linking spoofing | attacker registers a colliding entity id to hijack its embedding neighbourhood | medium | entity-id namespace validation at `ruvector-graph`'s existing `EdgeSchema`/schema layer — **no KGE-specific literature mitigation found, flagged as a local design gap** |
| FFI memory-safety defect | crafted scoring input reaches `unsafe` at the napi-rs boundary | high | minimise `unsafe`; fuzz the scoring entry point (cargo-fuzz), ASan in CI; input limits enforced before the boundary — identical to typesafe ADR-005 |
| Over-broad WASM capabilities | WASI fs/net imports added "for convenience" | medium | wasm32 build links **no** WASI fs/net imports; CI inspects the import section (`wasm2wat`/`wasm-objdump`) and fails on any |
| Data egress | telemetry, model download at runtime | high (breaks the promise) | **no network by default**; tables ship bundled or from a local manifest path; CI greps the built artifacts for `fetch`/`http` symbols in the scoring path |
| Triple/text in logs | entity descriptions or query state logged verbatim | medium | triples' text is never logged; receipts store hashes and lengths; debug logging redacts unconditionally |
| Dependency CVE / yanked crate | transitive dependency, incl. the **new FFT crate** | low–medium | existing `cargo-deny`/`cargo-audit`/`npm audit`/lockfile-integrity gates; **the FFT crate's licence is checked at add time** against `deny.toml:178-186` (confidence-threshold 0.8), not assumed |

### Input limits (rejected with a typed error, never silently truncated)

- ≤ 1 M entities and ≤ 100 k relations per loaded table (index build budget).
- ≤ 255 candidate options per `PREDICT`; `k` ≤ 1000 per query.
- entity/relation description text ≤ 1 KB; query `state` ≤ 16 KB.
- ≤ 64 queries per request batch.

### Trust boundaries

- **Tier A (authoritative):** programmatic verifiers — frozen-split MRR, the
  RANDOM-tie-break assertion, ANN recall@k, the sequential test.
- **Tier B (quarantined):** LLM-judge or heuristic triple labels, held until
  confirmed by A.
- **Tier C (untrusted, never authoritative):** ingested triples, descriptions,
  query text. They influence a *score*; they never influence a *promotion*
  (ADR-004) except through the A/B gate.

### Supply chain and release

- Model manifest `{name, sha256, dims, scorer, license, source_url, added,
  review_by}` per table file; load fails closed on mismatch.
- `npm publish --provenance` for `@ruvector/kge` and platform packages;
  cosign signatures on the manifest and release tarballs; CycloneDX SBOM per
  release. `deny.toml` and `supply-chain.yml` apply unchanged; the only new
  licence class to review is the FFT crate's (expected MIT OR Apache-2.0).

## Consequences

- Runtime behaviour is deliberately boring: no downloads, no shell, no network,
  no logging of triples. Convenience features that need any (auto-download,
  remote labelers) are opt-in flags with their own CI assertion that the default
  build lacks them.
- Poisoning resistance is bounded — an adversary who controls enough of the
  graph can shift any score. The mitigation is that the symmetry-pattern attack
  (the highest-generalizing vector) has a dedicated CI regression set gated on
  confidence degrading honestly, plus ingest-time anomaly detection.
- Fuzzing, ASan, and the FFT-crate licence review add CI time — the price of a
  native FFI over untrusted structured input.

## Alternatives considered

- **Blanket DP-SGD on the whole table.** Dense DP noise destroys utility on
  large sparse embedding tables (arXiv:2311.08357); selective/ sparsity-
  preserving DP for confidential statements only is the opt-in path instead.
- **An LLM triple-plausibility filter at ingest.** Adds a heavy model and
  network to a path that must avoid both; the poisoning defense is architectural
  (provenance/trust-tiers), not a model.
- **Trust upstream graph sources.** Rejected: the poisoning literature is
  explicitly about indirect, upstream decoy triples.

## Evidence

- Poisoning: Zhang et al., IJCAI 2019 (arXiv:1904.12052); **Bhardwaj et al., ACL-
  IJCNLP 2021 (arXiv:2111.06345, code github.com/PeruBhardwaj/InferenceAttack) —
  symmetry-pattern attacks generalize across every model/dataset tested.**
  Membership inference: Wang & Sun (arXiv:2104.08273). DP: ScienceDirect
  S1570826821000640; sparsity-preserving DP-SGD arXiv:2311.08357.
- Reused unchanged from typesafe ADR-005: manifest hashing, cosign/SBOM, FFI
  fuzzing, no-net/no-shell CI assertions, WASI import inspection.
- Repo: `.github/workflows/supply-chain.yml`; `deny.toml:178-186,195`;
  `crates/ruvector-graph/src/schema.rs` (entity-id namespace validation point).
- Local gap (no literature mitigation): entity-linking spoofing.
