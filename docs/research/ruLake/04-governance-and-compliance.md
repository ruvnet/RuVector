# 04 — Governance & Compliance

"Enterprise-grade" is a specific list. If ruLake cannot answer each of
these questions **without** a six-month consulting engagement, the
warehouse team vetoes it and the spike fails. This document walks the
list, says what RVF already carries, what is missing, and what ruLake
needs to build.

The list (deal-breakers in the order data-governance leads ask about them):

1. Row-level access control
2. Column-level access control / masking
3. Lineage
4. GDPR / CCPA right-to-erasure
5. PII masking on ingest
6. Audit logs
7. Region pinning
8. Encryption at rest + in transit + key management
9. Retention / time-travel policy compatibility
10. Break-glass + incident response

---

## 1. Row-Level Access Control

**What RVF has.** `crates/rvf/rvf-runtime/membership.rs` implements
`MembershipFilter` — a per-branch bitset saying which vector IDs are
visible. COW branching (`cow.rs`) enables creating a per-user or
per-group snapshot cheaply (~3 ms for 1 M-vector parent with 100 edits,
per README).

**What the host already does.** Every Tier-1 / Tier-2 warehouse has a
production row-level security model: BQ row access policies, Snowflake
row access policies, Unity Catalog row filters.

**ruLake choice.** **Delegate to the host.** Row-level security is
applied **after** the UDF returns top-k. This means the UDF can in
principle see more vectors than the caller is allowed to see; the host
filters them out before the analyst ever observes them.

**What we give up.** The UDF observes IDs it will later discard. This
is fine for confidentiality (the IDs are opaque u64 handles), but for
a customer who needs _the UDF itself_ to be unable to read certain
vectors, we have to push the ACL down through `MembershipFilter`. That
is a v2 feature (per-user bundle materialisation).

---

## 2. Column-Level Access Control / Masking

**What RVF has.** Nothing at column granularity — RVF's "column" is
the whole vector.

**What the host does.** BQ column-level security, Snowflake dynamic
masking policies, Unity Catalog column masks.

**ruLake choice.** **Entirely delegated to the host.** The vector
column in Parquet is opaque; if the host decides the user is not
allowed to see it, it returns NULL. The UDF does not see unauthorised
vectors because it does not see the `vec` column — it only sees the
`id`, and it reads vectors directly from GCS. This means the `.rvf`
bundle itself must be ACL-protected at GCS level (object-ACL or
bucket-ACL).

**Named trade-off.** The `.rvf` bundle is protected by object-store
ACLs, not by host warehouse column ACLs. A user with BQ read on the
table but no GCS read on the bundle will get "permission denied" from
the UDF. This is a _support ticket path_, not a data leak, but it must
be documented in the operator runbook.

---

## 3. Lineage

**What RVF has.** Three things:

- **`FileIdentity` + `lineage.rs`** in `rvf-types`. Every `.rvf` file
  records its parent, grandparent, and full derivation chain by hash.
- **`WITNESS_SEG (0x0A)`**. Every insert, query, and deletion is
  hash-linked into a SHAKE-256 witness chain. Changing any byte breaks
  the chain.
- **Cryptographic signatures (Ed25519, ML-DSA-65).** Segments and
  witnesses can be signed.

**What the host does.** Dataplex (GCP), Unity Catalog (Databricks),
Polaris (Snowflake / Iceberg), AWS Glue. Each tracks "job X read table
Y and wrote table Z" via their own ingest of query history.

**ruLake choice.** Emit a lineage edge **per BQ job** via the
catalog's lineage API (Dataplex Data Lineage REST, Unity Catalog
lineage tables, Polaris events). The edge carries:

- BQ job id
- bundle content hash (the `.rvf` manifest hash)
- witness chain head (a 32-byte SHAKE-256 hash)
- rulake UDF version

```
BQ job_id  ─────▶  table: project.ds.embeddings
      │                    │
      │                    ▼
      │            [ruLake bundle hash: 0xAB...]
      │                    │
      ▼                    ▼
  rulake.udf v1.2    [witness_chain_head: 0x7C...]
```

**What the host graph now shows.** An analyst clicking on the job in
Dataplex sees the same information plus "rulake signed this read."
Legal / audit can replay the witness chain offline to verify that the
bundle read was the one the host logged.

**What we give up.** Two lineage systems — ruLake's witness chain and
the host's lineage graph — that have to agree. They can diverge on
timing (witness chain commits async). The operator runbook must
describe how to detect divergence and which system wins.

---

## 4. GDPR / CCPA Right-to-Erasure

**The hardest question in enterprise governance. Read slowly.**

### What "deletion" actually means

A customer makes a data subject request: "delete everything about
user 12345." That decomposes into:

- (a) **Behavioural deletion.** No query ever returns user 12345's
  vector again.
- (b) **Cryptographic deletion.** No file anywhere on disk contains
  user 12345's vector value.
- (c) **Link deletion.** No log, no witness, no lineage record lets
  anyone reconstruct that user 12345's vector existed.

(a) is minutes. (b) is weeks, typically. (c) is philosophically
impossible at the witness-chain level — that is the _point_ of a
witness chain — and requires a different mitigation.

### What RVF has

- `DeletionBitmap` in `rvf-runtime/deletion.rs` — logical deletion.
  Bit set means query returns "no match." Instantaneous.
- `JOURNAL_SEG (0x04)` — records the deletion event, append-only.
- `COW` compaction — rewrites VEC_SEG without the tombstoned rows,
  retires the old file.
- `RedactionLog (0x35)` in rvf-federation — records _what_ was
  redacted (categories, counts) without the raw values.

### ruLake orchestrator (new)

Phase 1 — **Behavioural deletion (minutes):**

1. Host (BQ / Snowflake / DuckDB) runs `DELETE FROM tbl WHERE id = ?`.
2. ruLake catalog adapter receives the event.
3. ruLake writes a new JOURNAL_SEG tombstone to the `.rvf` bundle.
4. UDF refreshes its `DeletionBitmap` on next query (or is forcibly
   warmed via a management API).

_From this point, no query returns the vector._

Phase 2 — **Cryptographic deletion (asynchronous, ≤ 30 days SLA):**

5. Nightly compaction job rebuilds VEC_SEG without the tombstoned rows.
6. A new witness chain entry records: "`compaction_replaced <bundle_hash_old> with <bundle_hash_new>`, redaction_reason: gdpr_dsr_<ticket_id>".
7. The old `.rvf` bundle is deleted from GCS after the host's Time
   Travel retention window (BQ 7 days default, Snowflake up to 90,
   Delta 7) expires. **Time Travel retention is the governing SLA**,
   not ruLake.
8. A `RedactionLog (0x35)` segment is appended to the new bundle with
   the SHAKE-256 hash of the pre-redaction vector payload. This proves
   "the thing we redacted existed" without revealing its value.

Phase 3 — **Link deletion (named trade-off):**

9. The witness chain entry from step 6 **still references the old
   bundle hash**. That is immutable; breaking it would break
   audit integrity. A sophisticated adversary with access to BOTH
   `bundle_hash_old` AND the logs recording the deletion event could in
   principle correlate "user 12345 was at bundle_hash_old." We do not
   claim to solve this. The RedactionLog entry names _categories_, not
   identities.

### What we give up

- **Not sub-minute.** Compaction is nightly. If a customer's legal
  requirement is "gone within an hour," we need a faster compaction
  path. Not in v1.
- **No per-row encryption.** Per-row encryption with per-user keys +
  key deletion (the "crypto-shredding" pattern) would give us
  (b) instantly, but adds a key-management system and a per-row
  overhead. Deferred to v2.
- **Witness chain retains a hash of the redacted payload.** Named.
  Documented. Legal-reviewable. Not hidden.

### Budget

- Phase 1 orchestrator: 1.5 E-wks (week 9)
- Phase 2 compaction automation: 2.0 E-wks (week 9–10)
- Legal-reviewable doc for data protection officers: 0.5 E-wks
- Per-row encryption (v2): ~6 E-wks

---

## 5. PII Masking on Ingest

**What RVF has.** `rvf-federation/pii_strip.rs` — 3-stage pipeline
with 12 built-in detectors (paths, IPs, emails, API keys, env vars,
usernames). Outputs a `RedactionLog` segment recording what fired.

**What the host does.** BQ Data Loss Prevention (DLP), Snowflake
automatic classification, Unity Catalog column tags.

**ruLake choice.** **Belt-and-suspenders.** Run our PII stripper on
ingest (before vectors are embedded or stored), AND let the host DLP
classify downstream. Both produce evidence; the host's evidence wins
for compliance reporting.

---

## 6. Audit Logs

**What RVF has.** `WITNESS_SEG (0x0A)` — hash-linked append-only log
of every insert, query, deletion. Signatures optional.

**What the host does.** BQ audit logs (Cloud Logging), Snowflake
access history, Unity Catalog audit logs, Azure Monitor.

**ruLake choice.** **Both.** Every ruLake UDF call writes a witness
entry AND emits a structured log record to the host audit sink
(Cloud Logging for BQ, Snowflake event tables for SF, etc.). The
witness entry is the tamper-evident ground truth; the host log is
what the analyst's Splunk query will find.

Matching is by `(job_id, bundle_hash, query_hash)`.

---

## 7. Region Pinning

**What RVF has.** None at the format level. An `.rvf` bundle is
portable anywhere it can be mmap'd.

**What ruLake adds.** The bundle manifest (`table.rulake.json`)
records the primary GCS region. The ruLake UDF refuses to serve a
query from a Cloud Run instance in a different region, unless a
cross-region policy flag is set. This enforces data residency at the
application layer (the GCS bucket ACL is the hard enforcement).

**What we give up.** Multi-region active-active. If the business
wants EU and US both reading the same bundle live, v1 says "no, run
two bundles." That is a v2 replication problem.

---

## 8. Encryption

**At rest:** Google-managed keys on GCS (default) or CMEK via
KMS. No new work.

**In transit:** HTTPS (Cloud Run default). `rvf-server` supports TLS
via the standard axum path.

**Key management:** CMEK pass-through. For quantum-safe signatures,
`rvf-crypto` supports ML-DSA-65 and SLH-DSA-128s already (see
ADR-154 context and `rvf-crypto/README.md`). We enable them for
bundle signing in v1 and document the verification cost (see
R7 in `00-master-plan.md`).

---

## 9. Retention / Time-Travel Compatibility

**BigQuery:** Time Travel 7 days default, configurable to 7 days
(short) or up to 7 days (verify; table-level `max_time_travel_hours`
can be adjusted). Our GDPR phase-2 compaction respects this.

**Snowflake:** Time Travel up to 90 days. Phase-2 compaction SLA is
"after Time Travel expires" — so ≤ 90 days worst-case for Snowflake
customers.

**Delta:** 7 days default via `deletedFileRetentionDuration`.

**Iceberg:** Snapshot expiration policy is per-table. Default is
5 days.

**ruLake choice.** Phase-2 compaction SLA is the **max** of the
host's retention window and the data-subject-request legal SLA
(typically 30 days for GDPR). Document which one wins per-host
in the operator runbook.

---

## 10. Break-Glass + Incident Response

**What RVF carries that helps.**

- Witness chain lets an incident responder _prove_ whether a bundle
  has been tampered with since its last signed entry.
- FileIdentity lineage lets them walk back to the exact parent bundle.
- COW branching lets them create a forensic snapshot without locking
  production.

**What ruLake adds.**

- A `ruLake inspect` CLI subcommand that walks the bundle, verifies
  signatures, replays the witness chain, and prints a one-page
  integrity report. Planned in week 10.
- A documented "freeze and forensic" runbook (in `docs/ops/` once
  the spike accepts) that an on-call engineer can follow in under
  15 minutes.

---

## Summary of What ruLake Has to Build (Governance-Only)

| Work item                                         | E-wks | Week |
|---------------------------------------------------|------:|-----:|
| Bundle manifest + region field                    | 0.5   | 2    |
| Dataplex lineage adapter (REST, Rust)             | 2.0   | 9    |
| GDPR orchestrator (phases 1 + 2)                  | 3.5   | 9–10 |
| Audit-log forwarder (Cloud Logging + generic JSON)| 1.0   | 10   |
| `ruLake inspect` forensic CLI                     | 0.5   | 10   |
| Unity Catalog + Polaris lineage sketches          | 1.0   | 11   |
| Operator runbook + data-protection doc            | 1.0   | 11–12 |
| **Total**                                         | **9.5 E-wks** | |

This is ~half the whole spike budget. Governance is where ruLake
earns the word "enterprise"; under-resourcing it is the most common
way this kind of project fails.
