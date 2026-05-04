---
adr: 178
title: "ruvector + ruview / hailo cluster integration gap analysis"
status: Proposed
date: 2026-05-04
authors: [ruvnet, claude-flow]
related: [ADR-167, ADR-168, ADR-171, ADR-172, ADR-173, ADR-176, ADR-177]
branch: hailo-backend
---

# ADR-178 — ruvector + ruview / hailo cluster integration gap analysis

## Status

**Closed (iter 257).** All HIGH+MEDIUM gaps remediated; G (Pi 4
measurement) deferred without a Pi 4 in lab; long-form C/D (CSI
pose semantics + downstream cluster consumer) tracked as separate
multi-month ADRs out of this branch's scope.

| Gap | Severity | Status | Closed by |
|-----|----------|--------|-----------|
| A — ruvllm-bridge no deploy artifacts | HIGH | closed | iter 215 |
| B — `EmbeddingProvider` not impl'd | HIGH | closed | iter 218 (path dep + impl) |
| C — CSI bridge dropping I/Q (short) | MEDIUM | closed | iter 217 (doc-only) |
| C — CSI bridge dropping I/Q (long) | MEDIUM | tracked separately | future ADR |
| D — no downstream cluster consumer (short) | MEDIUM | closed | iter 221 (example) |
| D — mcp-brain client (long) | MEDIUM | tracked separately | future ADR |
| E — hailo crates excluded from workspace | MEDIUM | closed | iter 219 |
| F — ADR-167 status stratigraphy | MEDIUM | closed | iter 217 |
| G — Pi 4 throughput unmeasured | LOW | deferred | needs Pi 4 hardware |
| H — `install-bridge.sh` misnamed | LOW | closed | iter 216 |

Original (planning) text below; output is a graded gap inventory
plus a remediation plan sized to the iter cadence (213 iters
across ~5 days at the time the ADR was first written).

## 1. Context

The `hailo-backend` branch shipped ~213 iterations across
`ruvector-hailo` (ADR-167), `ruvector-hailo-cluster` (ADR-167 §8, 168,
169, 170), HEF integration (ADR-176, iter 158-164), and DoS / TLS
hardening (iter 174-213, ADR-172).

Two upstream consumers are repeatedly cited as integrated. ADR-167 §2.5
promised `HailoEmbedder` would `impl
ruvector_core::embeddings::EmbeddingProvider` so any caller holding
`Arc<dyn EmbeddingProvider>` transparently gets NPU throughput; §8.4
made the same promise for `HailoClusterEmbedder`. ADR-171 promised a Pi 5
edge node where the hailo cluster, mcp-brain, and ruview's CSI
WiFi-DensePose pipeline cohabit on cognitum-v0 and feed each other.
Both claims need an audit before operator-facing copy can honestly say
"ruvector has a Hailo-8 backend." This ADR walks the code, grades each
claim, and orders the close.

## 2. Methodology

Static analysis of `hailo-backend` at iter 213 (`a838d9e9e` HEAD).
Read every "contract" ADR (167, 168, 171, 172, 173, 176, 177); inspected
workspace + per-crate `Cargo.toml` for actual path deps (not the ones
the ADRs talk about); inspected the two crate `lib.rs` files and all
seven binaries under `crates/ruvector-hailo-cluster/src/bin/` for trait
impls, public surface, and bridge encoding logic; walked
`crates/ruvector-hailo-cluster/deploy/` for artifact parity (install
script, systemd unit, env example, README mention, TLS flag support)
across the three bridges; cross-referenced `git log` for every iter
that promised work in the listed ADRs to see if it landed. Out of
scope: cluster-bench against cognitum-v0 (static audit, no hardware
runs); dynamic ruview tracing (sibling repo, only the in-tree bridge
is audited).

## 3. Findings

### 3.1 Integration that already exists

The bones are real. The cluster library surface
(`HailoClusterEmbedder`, `HealthChecker`, `P2cPool`,
`EmbeddingTransport`, `GrpcTransport`, `EmbeddingCache`,
`compute_fingerprint`) is consistent and 131 tests green per the
README. The DoS gate stack (README §"Security & DoS hardening") is
genuine defense-in-depth: eight env-tunable layers with floors, rate
limit, mTLS, HEF sha256 pin
(`crates/ruvector-hailo/src/hef_pipeline.rs` via iter-174
`RUVECTOR_HEF_SHA256`), cargo-deny across both crates (iter 177/202).

The HEF EPIC (ADR-176) hit its acceptance criteria. Iter 163 measured
**67.3 embeds/sec/worker, p50 57 ms** on cognitum-v0 (**9.6×**
cpu-fallback); iter 164 verified semantic ordering (NPU
`sim(dog,puppy)=0.50 > sim(dog,kafka)=0.27`); iter 168 hot-cache hit
**15.86 M/sec, p50 <1 µs**; iter 170 saturation (C=100, 60s) held
worker RSS at 91 MB.

The **ruview-csi-bridge** is genuinely deployed. The 350-line binary
parses ADR-018 magics `0xC5110001`/`0xC5110006`, derives an NL
description, posts via the TLS/mTLS-protected embed RPC. Complete
deploy bundle: hardened `ruview-csi-bridge.service`
(`User=ruvector-csi`, `ProtectSystem=strict`),
`ruview-csi-bridge.env.example` (correct for `--tls-domain` post iter
207), idempotent `install-ruview-csi-bridge.sh`, six CLI integration
tests (`tests/ruview_csi_bridge_cli.rs`). The **mmwave-bridge** (iter
116-122) is at the same parity tier: binary, install script
(`install-bridge.sh` — unqualified name, gap H), systemd unit,
`.env.example`, udev rules, TLS roundtrip test (iter 121).

`deploy/cross-build-bridges.sh:36` enumerates all three bridges and
produces aarch64 binaries — the *build* path is parity-clean. The
deploy path is where parity breaks (gap A).

### 3.2 Gaps

#### A. **HIGH** — `ruvllm-bridge` ships without any deploy artifacts

**Evidence.** `ls crates/ruvector-hailo-cluster/deploy/` returns 22
files. mmwave has four
(`install-bridge.sh`, `ruvector-mmwave-bridge.service`,
`ruvector-mmwave-bridge.env.example`, `99-radar-ruvector.rules`).
ruview-csi has three
(`install-ruview-csi-bridge.sh`, `ruview-csi-bridge.service`,
`ruview-csi-bridge.env.example`). ruvllm has **zero** —
`ls deploy/ruvllm-bridge.*` returns ENOENT.

The cross-build script builds it (`BINS` line 36) and the help text
mentions it (line 10), but the install hint at lines 144-145 only
points operators at the mmwave + csi install scripts. The README's
"What it ships" table doesn't list ruvllm-bridge as a binary at all.
ADR-173 (the bridge's home ADR) was last touched at iter 138 and
predates the iter 124 binary landing.

**Impact.** A ruvllm consumer that wants RAG retrieval against the
hailo cluster has no documented deploy path. They get a binary they
can run by hand, no systemd integration, no env file template, no
`User=ruvector-llm` sandbox, no integration with the iter-205
`StartLimitBurst` parking. This is the iter-207 commit message's
specific complaint about ruvllm-bridge ("ships in the repo but has
no install script or systemd unit") and it remains open at iter 213.

**Recommended close.** One iter, ~150 LOC. Mirror the ruview-csi
bundle: `install-ruvllm-bridge.sh` (copy of csi version with UDP
references dropped, fingerprint/TLS env validation kept);
`ruvllm-bridge.service` (`User=ruvector-llm`, `ProtectSystem=strict`,
`NoNewPrivileges`, no `/dev/*` allow — bridge is pure stdin/stdout);
`ruvllm-bridge.env.example` (RUVLLM_BRIDGE_WORKERS, FINGERPRINT, all
four TLS flags including `--tls-domain` baked in); README "What it
ships" row + cross-build hint update at line 145.

#### B. **HIGH** — Neither hailo crate depends on `ruvector-core`; `EmbeddingProvider` is never implemented

**Evidence.** `crates/ruvector-hailo/Cargo.toml:33-49` — dependencies
are `thiserror`, `hailort-sys`, `sha2`, plus optional candle /
tokenizers under `cpu-fallback`. **Zero `ruvector-core` reference.**
`crates/ruvector-hailo-cluster/Cargo.toml:42-67` — dependencies are
`tonic`, `prost`, `tokio`, `dashmap`, `governor`, `ed25519-dalek`,
`ruvector-mmwave`, `ruvector-hailo`. **Zero `ruvector-core` reference.**

The workspace `Cargo.toml:13-26` explicitly excludes both crates and
the comment says verbatim:

> "ruvector-hailo-cluster: multi-Pi coordinator (ADR-167 §8). Standalone
> for now; joins the workspace once iteration 14 lands tonic + the
> ruvector-core path dep."

That iter 14 never landed. Iter 14 in the actual log is
`14f44a3e8 test(hailo): lock in iter-174 HEF sha256 pin behavior`. The
"iter 14" referenced here was the original ADR-167 §5 plan from the
iter 12-15 timeline; the work moved on without it.

`crates/ruvector-hailo/src/lib.rs:396-405` carries an
`embedding_provider_signature_parity` test that, on inspection, is a
no-op — it asserts only `T: Send + Sync`, never that
`HailoEmbedder impl EmbeddingProvider`:

```rust
fn assert_signatures<T>() where T: Send + Sync {}
assert_signatures::<HailoEmbedder>();
```

`crates/ruvector-hailo-cluster/src/lib.rs:140-143` is more honest:

> "Implements `EmbeddingProvider` (once iteration 14 brings the path dep
> on `ruvector-core`) so callers can swap a single-device `HailoEmbedder`
> for a fleet without code changes."

Both `HailoEmbedder` and `HailoClusterEmbedder` expose hand-rolled
methods (`embed`, `embed_one_blocking`, `dimensions`, `name`) that
*shape-match* `EmbeddingProvider`'s trait, but no `impl` block exists.
`ruvector-core::embeddings::EmbeddingProvider` lives at
`crates/ruvector-core/src/embeddings.rs:41-50` exactly as ADR-167 §2.5
described, with `HashEmbedding`, `OnnxEmbedding`, `ApiEmbedding` as
the only impls.

**Impact.** This is the headline integration claim from ADR-167 and it
isn't real. An app that holds `Arc<dyn EmbeddingProvider>` (the
recommended consumer pattern) cannot transparently swap to a Hailo
cluster. Either the caller links `ruvector-hailo-cluster` directly and
uses the inherent-method API (giving up the trait abstraction), or
writes a thin wrapper that implements the trait. Neither is documented.

| Caller | Today's path | What ADR-167 promised |
|---|---|---|
| `ruvector-core::AgenticDb` | Cannot use Hailo backend at all | Drop-in `Arc<dyn EmbeddingProvider>` swap |
| `ruvector-cli` | No `--backend hailo` flag (never landed) | `--backend {cpu,hailo}`, ADR-167 §2.3 |
| `ruvector-server` | No Hailo backend wiring | Worker side per ADR-167 §8.5 |
| App holding `BoxedEmbeddingProvider` | Must rewrite | Zero code changes |

**Recommended close.** Two iters. (1) Land the path dep:
`ruvector-core = { path = "../ruvector-core", default-features = false }`
(default-features off so hnsw / storage / reqwest don't reach the Pi
build); `impl EmbeddingProvider for HailoEmbedder` — three methods,
~20 LOC, delegates to existing inherent methods; same on
`HailoClusterEmbedder`. Replace the no-op signature-parity test with a
real impl-bound static assertion. (2) Rejoin the workspace: move both
crates from `exclude` to `members`; `cargo build --workspace` stays
clean on x86 because `hailo` stays opt-in. Unblocks ADR-167 §2.3's
`ruvector-cli --backend hailo` flag as a follow-up.

#### C. **MEDIUM** — CSI bridge encodes only header summary, drops the I/Q payload

**Evidence.** `crates/ruvector-hailo-cluster/src/bin/ruview-csi-bridge.rs:95-108`,
`summary_to_text`:

```rust
fn summary_to_text(s: &CsiSummary) -> String {
    let kind = if s.magic == CSI_MAGIC_V6 { "feature-state" } else { "raw" };
    format!(
        "wifi csi {} packet from node {} channel {} rssi {} dBm noise {} dBm \
         antennas {} subcarriers {}",
        kind, s.node_id, s.channel, s.rssi, s.noise_floor,
        s.n_antennas, s.n_subcarriers,
    )
}
```

The 20-byte header is parsed (line 70-88); the I/Q payload (`bytes
20..` per the wire-format docstring at line 25) is never inspected.
Each frame becomes a fixed-template NL string with seven
small-cardinality integers interpolated in.

ADR-171 implies the pipeline is `WiFi CSI tensor (N×64×30 complex
floats) → preprocess (magnitude, FFT) → push input vstream (HEF) →
pull output vstream (pose tensor) → postprocess`. The bridge does
**none of that**. It does: parse header → format string → embed
string via the same text-encoder HEF that serves regular sentence
inputs. The cosine embeddings of these strings cluster by
`(channel, rssi-bucket, node-id)` and not by anything related to
actual WiFi-DensePose pose content.

**Impact.** Semantic similarity over the cluster's embedding output
will retrieve "all CSI packets from node 3 on channel 6" together —
useful for telemetry indexing, useless for DensePose-style "which
packets show similar pose?" The bridge is correctly named (it
bridges the *transport*) but ADR-171 implies this is the integration
point for pose semantics, and that part is absent.

**Recommended close.** Two-stage. Short-term (iter ~217, doc-only):
update ADR-171's status block + the bridge file's docstring to
clarify "ruview-csi-bridge ships a *telemetry-indexing* bridge:
header→NL→text-embed. Real CSI→pose semantic embedding requires a
pose-specific HEF and a dedicated `csi-pose-bridge`, **not yet
built**." Long-term (separate ADR): `csi-pose-bridge` needs a hailo8
pose HEF (Hailo Model Zoo today ships only hailo15h/10h, ADR-167
line 184-194), a `HailoPipeline<I, O>` generalization of
`EmbeddingPipeline`, and host-side I/Q preprocessing. Multi-month, not
iter-sized.

#### D. **MEDIUM** — No downstream consumer reads cluster embeddings

**Evidence.** Grep for `HailoClusterEmbedder` outside the hailo
crates returns nothing. The `mcp-brain-server` crate (the ruOS brain
referenced in ADR-171) contains zero references to the hailo cluster.
No consumer of `ruvector-core::EmbeddingProvider` has been rewired to
receive a Hailo-backed implementation. ADR-171's promised
mcp-brain.service is "still unimplemented" per ADR-171's own status
block. `ruvector-cli`, `ruvector-server`, `ruvector-node` — each grep
clean for hailo.

The user-facing entry into the cluster is exclusively via the five
CLI binaries (worker, embed, fakeworker, stats, bench) plus the three
bridges (mmwave, ruview-csi, ruvllm). There is no Rust library caller
in this workspace.

**Impact.** Embeddings produced by the cluster end at the JSONL
output of `ruvector-hailo-embed` or in the response stream of the
three bridges. They don't feed the ruvector vector index (HNSW /
DiskANN / RaBitQ) — those crates can't see this provider through the
trait because of gap B. ruview's CSI summaries flow into the cluster
but the resulting vectors are not stored anywhere durable; they're
returned to the bridge stdout and the bridge logs a count.

**Recommended close.** Sequenced after gap B. Iter ~219:
`examples/hailo-cluster-as-provider.rs` — 5k-doc corpus, embed via
`HailoClusterEmbedder` (now trait impl), insert into
`ruvector-core::AgenticDb` HNSW, measure ingest QPS + retrieval recall
vs `OnnxEmbedding` baseline. Closes ADR-167's "every other subsystem
inherits the speedup" claim. Iter ~220 (separate): mcp-brain client
that calls `embed_one_blocking` on the cluster and posts to
pi.ruv.io — ADR-171's promised design, currently absent.

#### E. **MEDIUM** — Hailo crates excluded from workspace; `cargo build --workspace` doesn't see them

**Evidence.** Root `Cargo.toml:10-26` explicitly excludes
`crates/ruvector-hailo`, `crates/hailort-sys`, and
`crates/ruvector-hailo-cluster`. CI for the cluster crate is implicit
(cross-build per iter 122) but `cargo build --workspace` against the
repo will not even type-check the hailo crates.

**Impact.** Workspace-wide refactors (a `ruvector-core` API change, a
clippy bump, a security advisory rebuild) are easy to ship without
realizing the hailo crates broke. Iter 202 hand-papered this with
per-crate cargo-deny gates, but the type-check side stays disjoint.

**Recommended close.** Folded into gap B. Once the path dep on
ruvector-core is real, the workspace exclude can drop. The `hailo`
feature gate keeps libhailort linkage opt-in, so x86 stock machines
still build clean.

#### F. **MEDIUM** — ADR-167 status stratigraphy is stale and misleading

**Evidence.** ADR-167 line 459 (the worker step-10 deliverable text):

> "Pi runtime smoke: reports `bind=0.0.0.0:50051 model_dir=...`,
> attempts open, exits clean with `NotYetImplemented` (gate is HEF
> compilation only)."

True at iter 12. Iter 163 made the worker actually serve real
embeddings; iter 145 added the startup self-test that prints
`sim_close > sim_far`. The ADR text never got updated. Same problem
in the status table: sections labelled "Earlier (iter 99-116)
snapshot" through "Earlier (iter 134/135) snapshot" stack on top of
the iter-163 NPU-default banner, and an unfamiliar reader has to walk
past three older snapshots to find what's true.

**Impact.** Operator-facing confusion. ADR-167 reads as in-progress;
ADR-176 reads as accepted. ADR-167 is the document a new operator
starts with.

**Recommended close.** One iter, doc-only. Collapse ADR-167's
iter-snapshot stratigraphy. Keep iter 163 (NPU default) as the only
status block. Move iter 99-145 history to a "History" appendix at the
bottom. Replace step-10 worker `NotYetImplemented` text with the
iter-145 self-test reality.

#### G. **LOW** — Pi 4 throughput is documentation only; no measurement

**Evidence.** ADR-177 §"Expected performance" carries a Pi 4 row of
"~3-4 / sec (est) ~1 s (est)". The "Neutral" consequence explicitly
says "Pi 4 throughput is documented but not measured — first operator
with a Pi 4 should run cluster-bench and contribute a row to ADR-176's
measurements table." This lab does not have a Pi 4.

**Impact.** Capacity planning for Pi 4 deploys is guesswork. ADR-177
is honest about the gap, so impact is bounded.

**Recommended close.** Defer until a Pi 4 is available. Track as
follow-up in ADR-177 only — no fix in this branch.

#### H. **LOW** — `mmwave-bridge`'s install script has the unqualified name `install-bridge.sh`

**Evidence.** `deploy/install-bridge.sh` installs the *mmwave* bridge.
Compare `deploy/install-ruview-csi-bridge.sh` (correctly named). The
cross-build script's hint at line 144 calls it unambiguously
(`install-bridge.sh /usr/local/bin/ruvector-mmwave-bridge # mmwave`)
but the filename itself is misleading once a third bridge exists.

**Impact.** Operator confusion only. Not behavior-affecting.

**Recommended close.** Folded into gap A. When `install-ruvllm-bridge.sh`
lands, rename `install-bridge.sh` → `install-mmwave-bridge.sh` in the
same commit; update the line-144 hint.

### 3.3 Bridge deploy parity matrix

| Artifact | mmwave | ruview-csi | ruvllm |
|---|:-:|:-:|:-:|
| `install-*.sh` | yes (misnamed, gap H) | yes | **no** |
| `*.service` | yes | yes | **no** |
| `*.env.example` | yes | yes (post iter 207) | **no** |
| README "What it ships" mention | implicit | implicit | **no** |
| `--tls-ca` flag | yes (iter 120) | yes (iter 123) | yes (iter 124) |
| `--tls-domain` flag | yes | yes (post iter 207) | yes |
| `--tls-client-cert` / `--tls-client-key` | yes | yes | yes |
| Cross-build script coverage | yes (line 36) | yes | yes |
| CLI integration tests | yes (iter 118) | yes (6, iter 125) | yes (iter 125) |
| Real Pi 5 e2e validation | n/a | n/a | yes (iter 149) |

The `ruvllm` column is the one with all the "no"s. Code is parity;
deploy isn't.

## 4. Decision

Status: **core embed path integrated; library trait wiring +
ruvllm-bridge deploy + downstream consumers incomplete.** The hailo
crates are a tightly-bundled production system that does not plug into
the rest of ruvector. Three pieces have to land for ADR-167's original
"drop-in alternative" framing to be honest:

1. The `EmbeddingProvider` trait impl on both `HailoEmbedder` and
   `HailoClusterEmbedder` (gap B) — without this, every other claim
   collapses.
2. The `ruvllm-bridge` deploy bundle (gap A) — the iter-207 commit
   message called this out and it remained open through iter 213.
3. Honest CSI semantics docs (gap C) — the ruview integration is real
   but does telemetry indexing, not pose embedding; ADR-171's diagrams
   overstate what shipped.

Ordered remediation list, sized to the existing iter cadence:

| Iter | Gap | Deliverable |
|---:|---|---|
| 214 | A | Land `deploy/install-ruvllm-bridge.sh` + `.service` + `.env.example`. Mirror the ruview-csi bundle. README row + cross-build hint. |
| 215 | B | `ruvector-hailo` path dep on `ruvector-core`; `impl EmbeddingProvider for HailoEmbedder`. Replace no-op signature-parity test with real trait-bound static assertion. |
| 216 | B+E | Same on `ruvector-hailo-cluster`; `impl EmbeddingProvider for HailoClusterEmbedder`. Drop both crates from workspace `exclude`; rejoin `members`. Verify `cargo build --workspace` clean on x86. |
| 217 | C | Update ADR-171 status block + bridge docstring to say "telemetry indexing, not pose embedding." Mention `csi-pose-bridge` as future work. |
| 218 | F | Collapse ADR-167's snapshot stratigraphy. Single status block (iter 163 NPU default). Move iter 99-145 history to appendix. |
| 219 | D | `examples/hailo-cluster-as-provider.rs`: 5k-doc corpus → cluster embed → ruvector AgenticDb HNSW → recall measurement vs ONNX baseline. Validates ADR-167's "every other subsystem inherits the speedup" claim. |
| 220 | H | Rename `install-bridge.sh` → `install-mmwave-bridge.sh`; cross-build hint update. |

Gap G (Pi 4 measurement) defers indefinitely; no in-repo fix.

This is six doc/code iters spread across two days at the existing
pace. None grow the surface area; all close existing claims.

## 5. Consequences

**After the remediation list lands**, the integration story is
consistent: a caller holding `Arc<dyn EmbeddingProvider>` can swap to
a single Pi (`HailoEmbedder`) or a fleet (`HailoClusterEmbedder`) with
no code change; the workspace builds the hailo crates by default; the
three sensor bridges have parity deploy bundles; a worked example
exercises the ruvector → hailo path end-to-end.

**What stays open after the list:**

- No mcp-brain integration on cognitum-v0 — ADR-171 §198-212 is a
  separate ADR's worth of work (mcp-brain.service, brain.sock IPC,
  pi.ruv.io REST plumbing).
- No real CSI→pose embedding — gap C's long-term close needs a hailo8
  pose HEF (Hailo Model Zoo doesn't ship today, ADR-167
  line 184-194), a `HailoPipeline<I, O>` trait, and host-side I/Q
  preprocessing. Multi-month, separate ADR.
- No LoRa transport — ADR-171 §128-142 sketched `LoRaTransport` impl
  of `EmbeddingTransport`; iter never came.
- `ruvector-cli --backend hailo` flag — ADR-167 §2.3 promised this.
  Trivially follows iter 215, but is its own follow-up.
- Pi 4 measurement — gap G — needs hardware not in lab.

The branch's actual production claim, after the remediation, is
narrower than ADR-167's original framing but defensible: "ruvector
ships a Hailo-8 cluster as an `EmbeddingProvider` impl with three
sensor bridges (telemetry-grade), eight-layer DoS hardening, and 9.6×
CPU-fallback throughput on Pi 5 + AI HAT+." That's what's real.

## 6. References

ADRs: ADR-167 (§2.5/§8.4 trait promise; status stratigraphy gap F),
ADR-168 (CLI surface, bridges out of scope), ADR-171 (brain
unimplemented per its own status; CSI semantics overstate gap C),
ADR-172 (security acceptance gate), ADR-173 (last-touched iter 138;
deploy bundle never landed, gap A), ADR-176 (accepted iter 163),
ADR-177 (Pi 4 numbers estimated, gap G).

Code: `crates/ruvector-core/src/embeddings.rs:41-50` (trait
definition); `crates/ruvector-hailo/Cargo.toml:33-49` (no
`ruvector-core` dep); `crates/ruvector-hailo/src/lib.rs:280-356`
(inherent methods that shape-match); `lib.rs:396-405` (no-op parity
test, gap B); `crates/ruvector-hailo-cluster/Cargo.toml:42-67` (no
`ruvector-core` dep); `crates/ruvector-hailo-cluster/src/lib.rs:140-143`
(verbatim admission of gap B);
`src/bin/ruview-csi-bridge.rs:95-108` (`summary_to_text`, gap C);
`src/bin/ruvllm-bridge.rs` (338 lines, no deploy, gap A);
`deploy/cross-build-bridges.sh:36,144-145` (3 built, 2 installed);
root `Cargo.toml:10-26` (workspace exclude + unresolved iter-14
comment, gaps B + E).

Iters: `d5e3019b6` (124, ruvllm-bridge feature), `5a0384418` (149,
ruvllm-bridge Pi 5 e2e), `a7477f404` (163, NPU production default),
`6f5af8b1d` (145, worker self-test), `2e1a47b06` (174, HEF sha256
pin), `88a4ea429` (207, csi-bridge --tls-domain doc fix; commit names
the open ruvllm-bridge deploy gap), `a838d9e9e` (213, current HEAD).
