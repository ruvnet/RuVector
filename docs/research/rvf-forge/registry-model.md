# RVForge Registry — Data Model v0.1

> Content-addressed, immutable, predecessor-linked release registry per
> ADR-294 §3/§9 and requirements P3/P12. This is the wire-format contract
> that the publisher CLI (`@ruvector/rvforge`), the Reader, and the hosted
> registry all implement against. Schema changes bump `schemaVersion` and
> require a migration note here.

## Identity rules

- Every object is canonical JSON (RFC 8785 style: sorted keys, UTF-8, no
  insignificant whitespace, no floats where integers suffice).
- An object's id is `sha256:<hex>` of its canonical JSON **excluding** the
  `signatures` array. Content addressing makes releases immutable: any
  change is a new object with a new id.
- `predecessor` links releases into a lineage chain; the first release of
  a package has `predecessor: null`.
- Signatures are Ed25519 over the object id; `role` is one of
  `publisher` | `registry` (countersignature per ADR-294 §9).

## Objects

### PublisherRecord

```json
{
  "schemaVersion": 1,
  "type": "publisher",
  "publisherId": "sha256:…",
  "displayName": "Cognitum",
  "publicKeys": [
    { "keyId": "ed25519:…", "alg": "ed25519", "publicKey": "base64", "validFrom": "2026-08-03T00:00:00Z", "revokedAt": null }
  ],
  "identityEvidence": { "method": "dns|github|manual-review", "reference": "…" },
  "contact": { "support": "…", "privacyPolicy": "…" },
  "signatures": [ { "keyId": "ed25519:…", "role": "registry", "sig": "base64" } ]
}
```

### Release (immutable)

```json
{
  "schemaVersion": 1,
  "type": "release",
  "releaseId": "sha256:…",
  "package": { "name": "…", "publisherId": "sha256:…" },
  "version": "1.3.0",
  "predecessor": "sha256:… | null",
  "rvfIdentity": "sha256:…",
  "rvfSize": 123456789,
  "capabilityManifest": "sha256:…",
  "runtimeProfiles": ["wasm", "rvm"],
  "compatibility": { "rvmVersionMin": "0.x", "stateSchemaVersion": 1, "witnessSchemaVersion": 1 },
  "modelManifest": { "location": "embedded|publisher-inference|meta-llm|byo", "digests": ["sha256:…"] },
  "softwareInventory": "sha256:…",
  "evaluationReport": "sha256:… | null",
  "securityReport": "sha256:… | null",
  "provenance": "sha256:…",
  "trustLevel": "published|tested|reviewed|enterprise-approved",
  "rollbackSafeUntilStateSchema": 2,
  "listing": { "category": "…", "description": "…", "priceModel": "free|open-source|one-time|subscription|…" },
  "publishedAt": "2026-08-03T00:00:00Z",
  "signatures": [
    { "keyId": "ed25519:…", "role": "publisher", "sig": "base64" },
    { "keyId": "ed25519:…", "role": "registry", "sig": "base64" }
  ]
}
```

Rules: a release is rejected unless (a) the publisher signature verifies
against a non-revoked key in the PublisherRecord, (b) `rvfIdentity`
matches the uploaded artifact, (c) `capabilityManifest` resolves, and
(d) `predecessor` is null or an existing release of the same package.
`trustLevel` records evidence achieved, never implied safety (ADR-294 §7).

### CapabilityManifest

```json
{
  "schemaVersion": 1,
  "type": "capability-manifest",
  "manifestId": "sha256:…",
  "defaultPolicy": "deny",
  "requests": [
    { "class": "filesystem", "scope": "user-selected", "rationale": "reads documents you choose" },
    { "class": "memory", "scope": "512MiB", "rationale": "model working set" },
    { "class": "persistent-state", "scope": "encrypted-local", "rationale": "agent memory" }
  ],
  "denials": ["network", "microphone", "process", "background", "external-model-providers"],
  "manualReviewTriggers": [],
  "signatures": [ { "keyId": "ed25519:…", "role": "publisher", "sig": "base64" } ]
}
```

`class` values are the 15 ADR-286 capability classes. `scope` must be
specific — broad scopes like `all-files` set `manualReviewTriggers`
(ADR-294 §8) and are prohibited from rendering as vague prose in the
install UX ("access your computer" is banned; requirements P6).

### WitnessReceipt

```json
{
  "schemaVersion": 1,
  "type": "witness-receipt",
  "receiptId": "sha256:…",
  "subject": "sha256:…",
  "event": "build|verify|install|update|capability-grant|capability-denial|revocation-check",
  "outcome": "pass|fail|denied",
  "actor": { "kind": "builder|reader|registry|rvm", "id": "…" },
  "evidence": { "details": "…" },
  "timestamp": "2026-08-03T00:00:00Z",
  "prevReceipt": "sha256:… | null",
  "signatures": [ { "keyId": "ed25519:…", "role": "registry", "sig": "base64" } ]
}
```

Receipts hash-chain via `prevReceipt` per subject, forming the verifiable
witness chain required by the §15/P16 acceptance tests.

### Revocation

```json
{
  "schemaVersion": 1,
  "type": "revocation",
  "revocationId": "sha256:…",
  "subject": "sha256:…",
  "subjectType": "release|publisher-key|publisher",
  "reason": "compromise|malware|policy|publisher-request",
  "effect": "block-execution",
  "issuedAt": "2026-08-03T00:00:00Z",
  "signatures": [ { "keyId": "ed25519:…", "role": "registry", "sig": "base64" } ]
}
```

`effect` is always `block-execution`: revocation blocks execution by
policy but NEVER deletes or silently removes locally owned RVFs or user
state; export and forensic access are preserved (ADR-294 §9, requirements
P12).

### TransparencyLogEntry

```json
{
  "schemaVersion": 1,
  "type": "log-entry",
  "index": 12345,
  "entryHash": "sha256:…",
  "treeHead": "sha256:…",
  "object": "sha256:…",
  "objectType": "release|revocation|publisher",
  "timestamp": "2026-08-03T00:00:00Z"
}
```

Append-only Merkle log over every published release, revocation, and
publisher record (ADR-294 §9). Readers verify inclusion proofs before
trusting a release; a release absent from the log is treated as
unpublished.

## Storage layout (local registry MVP)

```text
registry/
  objects/sha256/<2-hex-prefix>/<digest>.json   # all objects, content-addressed
  packages/<publisher>/<name>/releases.jsonl     # append-only release index
  log/entries.jsonl                              # transparency log
  log/tree-head.json                             # current signed tree head
```

The local (file-backed) registry is the MVP implementation target; the
hosted registry exposes the same objects over the /v1 API surface.
