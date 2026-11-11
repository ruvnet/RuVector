# ADR-354: RVF dialect convergence and an effective Strict policy (project Ofer prerequisites)

- **Status**: Proposed
- **Date**: 2026-10-26
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-033, ADR-284, ADR-353; RVM ADR-149, ADR-155, ADR-160
- **Tags**: rvf, interoperability, security-policy, tail-scan, ofer

## Context

Project Ofer's research found that the RVF stack is not interoperable end to end, reproduced on this machine (dossier section 1.7, reproducer `/data/scratch/ofer-research/interop`):

- A file written by `rvf_runtime::RvfStore` (the producer behind `rvf-server`) fails `rvm_rvf::walk` and `verify` with `BadMagic`: the runtime emits unpadded segments and labels a CRC32-rotation hash `checksum_algo = 0`, while the wire registry reads algo 0 as XXH3-128 and `rvm-rvf` walks on 64-byte boundaries. `rvf-runtime/src/hashing.rs` itself documents the incompatibility and says migration needs a version bump plus a dual-accept reader.
- A signed container from `rvf-forge-core` fails the same way because it ends in a 4096-byte Level-0 root page (`RVM0`) that `rvm-rvf::walk` does not skip. With that page stripped it verifies.
- Two tail scanners disagree on the same bytes (`rvf-wire` scans at 64-byte alignment; `rvf-runtime` scans byte-by-byte): the same bug class as ZIP end-of-central-directory parser differentials.
- "Verify the root manifest" (ADR-284 1.1, ADR-155 2.1) is a presence check. `RvfOptions.security_policy` defaults to `Strict` but is never read. `RvfStore::open_with_policy` and `create_signed` (ADR-033 section 4) do not exist. The Level-0 root signature helper has no caller outside authoring.

ADR-353 needs a trustworthy body to stream. These defects must be fixed first; they are worth fixing regardless of streaming.

## Decision

1. **One wire dialect.** The `rvf-wire` registry is authoritative: `checksum_algo` 1 = XXH3-128, 2 = SHAKE-256-128; segments are padded to 64 bytes. New `rvf-runtime` writes use the wire dialect. Anything that will be signed or released to an executor or index consumer uses algo 2 or a descriptor commitment (ADR-353), never algo 0 or 1 as an authenticity claim.
2. **Dual-accept legacy reader.** `rvf-runtime` keeps reading legacy-dialect files (algo 0 with the CRC32-rotation hash, unpadded segments) and marks them `legacy_dialect`. Under `Strict` a legacy-dialect segment cannot feed an executable or an index consumer.
3. **Root page.** `rvm-rvf::walk` recognizes a trailing 4096-byte Level-0 root page, as `rvf-forge-core` already does, and reports it as a root-page region instead of failing.
4. **One tail scanner.** The `rvf-wire` scanner is the single implementation; `rvf-runtime` calls it. A differential test and a fuzz target assert both code paths agree on the same bytes. The stream path never uses tail scanning for authority (ADR-353).
5. **Strict is real.** `security_policy` is read. `RvfStore::open_with_policy` exists. Under `Strict`: an unsigned manifest is refused (`UnsignedManifest = 0x0800`); the Level-0 signature is verified, not just parsed; model and index segments that feed queries must be authenticated (signature or descriptor-bound hash), not only header-hashed.
6. **Interop test in CI in both repos.** The reproducer's cases become tests: a `RvfStore`-written file and a Forge-authored file with root page must both be accepted by `rvm-rvf` verification (RVM side) and by the wire reader (ruvector side). Shared byte fixtures are checked into both repos.

## Compatibility

Existing files stay readable. Writers change; a writer-side dialect switch ships behind a version bump as `hashing.rs` already requires. A legacy-dialect file served to RVM needs the descriptor path or a re-write to the wire dialect.

## Acceptance

- The two reproducer cases pass in CI on both sides.
- A differential fuzz target finds no input on which the two scanners disagree.
- A tampered or unsigned manifest is refused under `Strict` and accepted only under an explicit lower policy; the ADR-033 section 3.5 "Malicious Tail Manifest" test runs against the real `open_with_policy`.
- Dual-accept: every file written before this change still opens and queries identically.

## Consequences and risks

Writer dialect change touches the hot path of `rvf-runtime`; mitigated by dual-accept and by round-trip tests over the existing test corpus. Making `Strict` effective can break callers that relied on it being ignored; mitigated by an explicit lower policy and a release note.

## Implementation notes and residual risks (runtime side, project Ofer)

Implemented in `rvf-runtime` 0.4.0. Outcomes of the adversarial review that shape what "Strict is real" means:

- **Strict is opt-in.** `RvfStore::open` / `open_readonly` remain `Permissive`. No in-tree consumer (`rvf-launch`, `rvf-cli` ingest/delete/compact, `rvf-server`) opts into `Strict`; none was changed silently. `rvf-launch::extract_kernel` now reports `signature_unverified` (and warns on stderr) when it extracts a kernel from a file that carries a signed root page it did not verify. Callers that execute extracted bytes must open with `open_with_policy` first.
- **Non-Strict scans can be steered.** Under `Permissive`/`WarnOnly` an aligned MANIFEST header planted inside the payload of a segment appended after the real manifest wins the tail scan (both scanners agree; scanning cannot authenticate). Strict refuses it: the signed page names the manifest offset, length and payload hash. The legacy byte-granular scan is chosen from the file's layout (the first segment whose end is not 64-aligned decides between a padded and an unpadded successor header), not from the unhashed `checksum_algo` byte, which only breaks ties when the layout is undecidable.
- **Scan cost is bounded.** At most 256 manifest payloads are parsed per open, in place, with a total-size budget of `max(16 MiB, file size)`; windows are read in blocks of at most 8 MiB. With a signed page under Strict/Paranoid the scan stops at the page's manifest offset.
- **Signed stores authenticate whole segments.** The signed directory commits to a SHAKE-256-128 payload hash per segment and pins every header field (id, type and length from the directory; flags, compression, reserved words, timestamp, `uncompressed_len` and padding to the writer's constants).
- **Compaction never launders.** For a signing handle `compact()` carries only directory-listed segments, verifies each against the hash the handle committed to (not recomputed from file bytes), and refuses the whole compaction on any mismatch before changing state. Witness segments appended by `query_audited` and `compact` are listed in a new signed manifest, so a signed store has no unlisted trailing segments.
- **A `Permissive` handle will not cut a signature.** A plain `open()` of a store whose file ends in a valid signed page refuses writes (`ReadOnly`) unless the caller opts in with `allow_unsigning(true)`; `open_signed` keeps the signature.
- **No freshness yet.** An older validly signed file replays cleanly under runtime `Strict`; epoch pinning exists only for RVSD streams. COW children are `Permissive`; `Paranoid` is `Strict` plus reading every directory segment.

### Round 2 changes and known limits

Fixed: layer payloads decode within 2x their wire size (counts are charged before allocation); partition ranges are visited through the vectors actually present (`VectorStore::for_each_in_range`), so a hostile `0..u64::MAX` costs at most one step per visible vector; INDEX/JOURNAL buffering has an ingest-level budget (default 256 MiB); terminal witness records are never dropped (routine ones are capped with a dropped count); a resumed frame log is cut to the consumer's offset and appended; under a signed page the page-named manifest is read directly (newer lookalikes are not consulted) and the manifest segment's own header is pinned (the page records its segment id); only payloads over 4 KiB count toward the manifest parse-attempt cap; a valid-looking header at both candidate successor positions selects the aligned scan; `StreamIngest` frame totals use checked arithmetic (armv7 tested under qemu); stream slots are released when a client stops reading (`idle_timeout`) or a response exceeds `total_deadline`, TCP has its own pool and closes idle sockets; the default `max_table_bytes` is derived from the default `max_body_len` (4 KiB chunks, the smallest), so it cannot be below what the body limit allows.

Known limits (not fixed, by design or out of scope):
- The stream endpoints are unauthenticated: the served body is downloadable by anyone who can reach the port. The body and key paths are operator-supplied; the key is read once, never logged, and its hex copy is not zeroized.
- There is no per-peer slot accounting (HTTP has no peer address without a different `serve` call; behind a proxy it is meaningless); silent HTTP sockets that send no request hold a file descriptor but no slot.
- A body without a root page is served with `file_id = [0; 16]`, so such bodies signed by one key share a pin namespace.
- `StreamIngest` is not `Send` (the policy holds `dyn` references); `query` assumes `k` well below `usize::MAX`; vector slots are `u32`; vectors with the same id from different segments or anchors overwrite each other (last writer wins, signer-controlled); `visible_vectors` counts tombstoned ids; a Layer A whose dimension differs from the VEC dimension is published and its anchors are dropped.
- Pages written before this change record no manifest segment id (zero); that single pin is skipped for them.
- Layer C (the store's own HNSW payload) is decoded by `rvf-index`'s codec and is not under the 2x decode budget.
- Runtime `Strict` has no freshness (no epoch pin); replay of an older validly signed file is not detected.

### Round 3 changes and known limits

Fixed: Layer A partitions must be sorted and disjoint (decode rejects overlap and the stream is poisoned), and every query carries a deterministic work budget in steps (`set_query_work_budget`; `work_truncated` / `work_steps` on the answer) so a valid-but-wide map cannot multiply cost. `RvfOptions::freshness` (see Round 4) adds a freshness pin to signature-checking opens (`StreamStaleEpoch` for older state); a signed page followed by a newer parseable manifest (page swap, planted or interrupted commit) is refused unless `serve_signed_prefix` is set, in which case the ignored manifest is reported by `newer_manifest_ignored`. No page-format change was needed. Pin advance is idempotent for an identical concurrent session. Chunk tables are freed when their segment verifies, so the `max_table_bytes` cap bounds open segments rather than the whole body; the per-segment `chunk_list_hash` is re-derived at completion. `StreamPolicy::conservative()` (16 GiB body) and `StreamPolicy::embedded()` (256 MiB body) give tighter limits than `new()`. The server enforces a minimum read rate per `rate_grace` window (default 32 KiB/s, 10 s), a per-client-address cap (4 HTTP streams, 16 TCP connections; HTTP via `ConnectInfo`), and a 600 s default `total_deadline`.

Known limits: the tombstone set built from journal segments is bounded only by the stream's own journal bytes and the body limits, not by a separate budget (documented, F3-04); per-client caps key on the socket peer, so clients behind one proxy or NAT share a cap; `rvf-server` only serves streams and never instantiates a verifier, so the presets apply to clients and `StreamIngest` users.

### Round 4 changes and known limits

Fixed: **F4-01** vector memory follows received bytes (room is reserved per received chunk, never from `stored_len` or the declared record count); a signed descriptor declaring 512 MiB with 320 KB delivered allocated 1.64 GB before and now stays within a few times the delivered bytes (`f4_01_declared_segment_size_is_not_prepaid`, allocator peak). **F4-04** freshness is a required, explicit choice: `RvfOptions::freshness: Option<Freshness>` (`Unchecked` or `MinEpoch(n)`); a Strict/Paranoid open with `None` fails with `ErrorCode::FreshnessRequired`. ANY bytes between the signed manifest and the page, parseable or not, are an unsigned tail (a damaged newer manifest must not hide itself; tests that inject such bytes opt in with `serve_signed_prefix`) and are refused unless `serve_signed_prefix`, which serves the signed state and reports `unsigned_tail_present()`. Only `MinEpoch` or an external pin gives freshness; `Unchecked` accepts replay of an older validly signed file. **F4-06** per-client accounting fails closed (`ClientCap::PerAddress` refuses with 503 when the peer address is unknown; `ClientCap::Disabled` is the explicit opt-out), keys on address and on IPv4 /24 or IPv6 /64 prefix (IPv4-mapped IPv6 folded into IPv4), holds back `reserve_slots` of the pool for prefixes with no stream yet, and bounds streams below 8x the minimum rate to `max_slow_streams`. Clients behind one NAT or proxy share a prefix and therefore a cap; distributed attackers with many prefixes are not stopped, only bounded to leaving the reserve and the fast share free. **F4-02/03** the Layer A centroid scan is charged to the work budget (a Layer A whose centroids exceed it is skipped and the answer is degraded), and `Auto` never scans more rows than the latency bound allows (oldest rows first, `work_truncated`, honest `coverage_vectors`); `ScanOnly` stays unbounded by design. **F4-05** a late chunk of the right length for a finished multi-chunk segment (table freed) is reported as `Event::Unverified`, counted against the duplicate budget and never as an authentic `Duplicate`; consumers must not log it. Legitimate resends are still tolerated (bounded by `max_duplicate_frames`).

### Residual limits (independent review) and test-suite note

- Strict proves authenticity, not freshness; only `Freshness::MinEpoch` or an external pin detects replay of an older validly signed state.
- A many-prefix attacker can exhaust the stream pool: 8 distinct /24 prefixes reading at about 400 KB/s fill 8 slots. The per-address and per-prefix caps, the reserve and the slow-stream budget bound a single network and keep room for fast and new-prefix clients; they are not a DDoS defence.
- Client memory grows with the bytes streamed (verified vectors and layers are retained).
- Test suites: building the 60,000-vector stream fixture at opt-level 0 took minutes and looked like a hang (`review2_ofer_server f2_05`: CPU-bound in `build_stream_store`, no server wedge). The rvf workspace now optimises the index, runtime, wire and hashing crates in dev/test builds, fixtures are cached under `CARGO_TARGET_TMPDIR` keyed by (n, dim, clusters, seed), and every rvf-server test socket has read and write timeouts.
