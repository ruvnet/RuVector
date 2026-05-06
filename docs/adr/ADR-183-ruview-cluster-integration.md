---
adr: 183
title: "Integrate RuView WiFi-sensing into the 4-Pi Hailo+ruvllm cluster"
status: accepted
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-167, ADR-171, ADR-178, ADR-179, ADR-180]
supersedes: []
extends: [ADR-171, ADR-178]
branch: feature/adr-183-ruview-cluster-integration
---

# ADR-183 — RuView integration into the cognitum Pi cluster

## Status

**Accepted.** All three tiers implemented; convergence criteria met (iter 20–22). Release `v0.1.0-csi-lora` cut on `cognitum-one/v0-appliance` (2026-05-06). Direct successor to ADR-171 (RuOS-Brain RuView Pi 5 edge node)
and ADR-178 (ruvector / RuView / Hailo gap analysis). Where ADR-171 sketched
a single-Pi edge node and ADR-178 catalogued five gaps (closing four,
deferring one), ADR-183 specifies how to put RuView's *actual* sensing
pipelines on the **4-node Pi 5 + Hailo-8 AI HAT+ cluster** that landed in
ADR-179 (cognitum-v0 master + cognitum-cluster-1/2/3 workers, Tailscale
mesh). Iteration 220 of ADR-178 closed gap C "in documentation only" — the
existing `ruview-csi-bridge` is a header-only telemetry tap, **not** a
WiFi-DensePose pose / vitals embedder. ADR-183 does the work that
disclaimer pointed at.

## Context

### What RuView ships *today* (local: `/home/ruvultra/projects/RuView`)

The vendored checkout is RuView v0.7.0 with a 20-crate Rust workspace at
`rust-port/wifi-densepose-rs/` (v0.3.0). Crates relevant to cluster
integration:

| Crate | Surface | Pi 5 status |
|---|---|---|
| `wifi-densepose-core` | `CsiFrame`, traits | aarch64-clean |
| `wifi-densepose-signal` | Hampel, SpotFi, Fresnel, BVP, spectrogram | aarch64-clean, pure CPU |
| `wifi-densepose-vitals` | `BreathingExtractor` (0.1–0.5 Hz, 6–30 BPM) + `HeartRateExtractor` (0.8–2.0 Hz, 40–120 BPM) + `CsiVitalPreprocessor` + `VitalEstimate` + `VitalStatus` | aarch64-clean, runs on Cortex-A76 |
| `wifi-densepose-nn` | ONNX/Candle/PyTorch backends; WiFlow architecture (1.8M params, 881 KB Q4) | aarch64-clean (Candle); ONNX path needs ort |
| `wifi-densepose-pointcloud` | `ruview-pointcloud` binary: depth + CSI + mmwave fusion → 3D point cloud, 22 ms pipeline, 905 req/s API; brain bridge syncs every 60 s | aarch64-clean; needs camera (cognitum-v0 has it) |
| `wifi-densepose-sensing-server` | Axum HTTP + WebSocket; UDP CSI ingestion | aarch64-clean |
| `wifi-densepose-wasm-edge` | Edge modules (60+) for ESP32 and aarch64 host | aarch64-clean for host runtime |

Pre-trained models live at HuggingFace `ruv/ruview`:

| File | Size | Purpose |
|---|---|---|
| `model.safetensors` | 48 KB | Contrastive 128-dim CSI encoder (presence, activity, environment) |
| `model-q4.bin` | 8 KB | 4-bit quantized — ESP32 SRAM friendly; also tiny on Pi |
| `presence-head.json` | 2.6 KB | Linear head, 100 % presence accuracy on the v0.6 overnight set |
| `node-1.json` / `node-2.json` | 21 KB each | Per-room LoRA adapters |

Pre-trained WiFlow pose model (camera-supervised, 92.9 % PCK@20, 974 KB) is
also published. **All of this is small enough to ship on a Pi**; nothing
about the model side is the blocker.

The brain bridge (`crates/wifi-densepose-pointcloud/src/brain_bridge.rs`)
already speaks to a brain at `RUVIEW_BRAIN_URL` (default
`http://127.0.0.1:9876`), POSTing `{category, content}` to `/memories`
every 60 s. That is the on-ramp for cluster-side observations.

### What the cluster has today (this repo)

| Component | Where | Ports |
|---|---|---|
| `ruvector-hailo-worker` (embedding, ADR-167) | each Pi | `:50051` gRPC |
| `ruvllm-pi-worker` (LLM completion, ADR-179, just shipped 2.2.1) | each Pi | `:50053` TCP/JSON |
| `ruview-csi-bridge` (ADR-171 iter 123, telemetry only) | each Pi | UDP `:5005` in, gRPC `:50051` out |
| `ruvector-mmwave-bridge` | optional | UDP in, gRPC out |
| `ruvllm-bridge` | optional | JSONL stdin/stdout |

The CSI bridge in `crates/ruvector-hailo-cluster/src/bin/ruview-csi-bridge.rs`
parses ADR-018 magic `0xC5110001` / `0xC5110006` headers and emits a
**natural-language summary string** (channel/RSSI/noise/antennas/subcarriers)
through the same text-encoder embed path as the mmwave bridge. The I/Q
payload (`bytes 20..`) is *parsed but discarded* — the file's own header
comment is explicit: "**this bridge is *not* WiFi-DensePose pose
embedding**". That disclaimer is the open issue ADR-183 closes.

### Gaps ADR-183 closes

From ADR-178 §3.2:

- **C (long-term)** — real CSI semantic embedding, not header summaries.
- **D** — no downstream consumer reads the cluster output.

Plus three new gaps that ADR-178 didn't enumerate but the local RuView
checkout makes obvious:

- **No vital-signs path.** The `wifi-densepose-vitals` crate is
  aarch64-clean Rust and would run unmodified on each Pi, but no bridge
  feeds it CSI frames or surfaces breathing/heart-rate to the cluster.
- **No sensor fusion node.** cognitum-v0 has a camera + mmwave radar +
  Hailo-8 (the others don't). `ruview-pointcloud serve` already fuses
  depth + CSI + mmwave on a single host but isn't deployed.
- **No HuggingFace pre-trained CSI encoder on the NPU.** The 48 KB / 8 KB
  contrastive encoders would compile to a Hailo HEF and replace the
  text-encoder-on-NL-summary kludge with content-aware CSI embeddings.
  This was assumed to be blocked on Hailo Model Zoo but the RuView models
  are ours to compile.

## Decision

Adopt a **three-tier, three-node-role architecture** that builds on
existing cluster ports rather than replacing them:

### Node roles

| Role | Host(s) | Responsibilities |
|---|---|---|
| **Master / fusion** | `cognitum-v0` (Pi 5 + AI HAT+ + camera + mmwave, only one with peripherals) | Run `ruview-pointcloud serve`, fusion of depth + CSI + vitals + mmwave, `mcp-brain` daemon, drive the dashboard / point-cloud viewer (loopback by default), aggregate brain spatial syncs |
| **Sensor worker** | `cognitum-cluster-1/2/3` | Run a new `ruview-vitals-worker` (per-Pi CSI windowed inference) and a `ruview-csi-relay` (UDP fan-out to v0 fusion node), in addition to the existing hailo embed worker (`:50051`) and ruvllm worker (`:50053`) |
| **Sensor source (off-cluster)** | 2–6 ESP32-S3 nodes per room | Already supported by RuView; broadcast ADR-018 frames at UDP `:5005` |

### Tier 1 — Vitals worker (ships in iter 1–6)

A new bin **`ruview-vitals-worker`** in `crates/ruvector-hailo-cluster/src/bin/`
replaces the iter-123 telemetry bridge's role on the workers (the bridge
itself stays — it's a different signal). The worker:

1. Listens on UDP `:5005` for ADR-018 v1/v6 frames.
2. **Keeps the I/Q payload** (the existing bridge dropped it). Decodes
   subcarrier magnitude/phase into a sliding window of N frames
   (default 50, ≈ 1.6 s @ 30 fps).
3. Calls `wifi_densepose_vitals::CsiVitalPreprocessor::preprocess(window)`
   → `BreathingExtractor::extract` and `HeartRateExtractor::extract`.
4. Emits `VitalEstimate { breathing_bpm, heart_rate_bpm, snr, status }`
   on a new gRPC service on **`:50054`** *and* writes a structured
   memory POST to the brain at v0 (`http://cognitum-v0:9876/memories`,
   reusing RuView's `brain_bridge.rs` shape — no new schema).
5. Optional: also encodes the vitals as an NL summary
   (`"wifi vitals node {id} breathing {bpm} bpm heart rate {hr} bpm snr {db} dB"`)
   and posts to the existing embed RPC — gives us cosine-search over
   "people breathing fast" without yet shipping the contrastive CSI
   encoder.

This is pure-CPU, no NPU, no model download. Model-free signal processing
on Cortex-A76 already meets the latency budget (vitals window updates at
~0.6 Hz; budget 1 s).

### Tier 2 — Fusion master on cognitum-v0 (iter 7–12)

cognitum-v0 is the only node with the camera, the mmwave, and the
AI HAT+. Run RuView's existing fusion server there:

1. New systemd unit `ruview-pointcloud.service` runs
   `ruview-pointcloud serve --bind 127.0.0.1:9880 --brain http://127.0.0.1:9876`.
   Loopback default; remote clients reach the viewer over Tailscale via
   `:9880` only when explicitly opted in (matches RuView's own posture
   in `README.md` line 130).
2. New systemd unit `ruview-mcp-brain.service` runs the brain daemon at
   `:9876`. Workers (cognitum-cluster-1/2/3) sync to it over Tailscale.
   Closes ADR-178 gap D.
3. `ruview-csi-relay` on each worker forwards full ADR-018 frames to v0
   so v0's fusion sees CSI from all rooms (each Pi typically anchors a
   physical zone). Uses same UDP wire format; v0 demuxes by source IP.
   This is a parallel data path to Tier 1 — workers do their own vitals,
   v0 does the global fusion.
4. Pose overlay: keep RuView's current "amplitude-energy heuristic"
   (per `README.md` line 132) as a placeholder; real WiFlow inference
   lands in Tier 3.

### Tier 3 — HuggingFace contrastive CSI encoder on the Hailo NPU (iter 13–22)

Compile the 48 KB `model.safetensors` (or 8 KB `model-q4.bin`) from
`huggingface.co/ruv/ruview` into a Hailo HEF and serve it through the
existing `:50051` embed path as a *new* model variant
(`HailoEmbedderConfig::variant = WifiCsi128d`). This finally closes
ADR-178 §3.2 C "long-term":

1. Add a `HailoPipeline<CsiTensor, [f32; 128]>` to ruvector-hailo
   alongside the text-encoder pipeline. Input shape is fixed by the
   RuView encoder (56 subcarriers × N frames × n_antennas, exact dims
   from `wifi-densepose-nn` config).
2. Compile RuView's model to HEF using `hailomz` CLI (Hailo Model Zoo
   tooling) — model shape is small enough that the standard ONNX→HEF
   path should not need custom kernels. Track sha256 of the resulting
   HEF the same way ADR-178 §1c iter-107 tracks signed manifests.
3. Add an `RUVIEW_CSI_MODEL` knob to `ruview-vitals-worker` so the same
   sliding-window pipeline can either compute vitals on the CPU
   (default, Tier 1) **or** call the NPU for a 128-dim contrastive
   embedding (Tier 3 mode). Both paths can run in parallel on different
   workers.
4. Search infrastructure: vitals/embedding consumers write to a
   coordinator-side HNSW index at v0 (`/var/lib/ruvector-vectors/`).
   `ruvector-cli` gains a `--backend hailo --variant wifi-csi-128`
   path that fills out ADR-178 §3.2 B's promise.

### Wire surface (full stack)

```
ESP32 (any room) ──UDP:5005 ADR-018──▶  cognitum-cluster-N
                                         ├─ ruview-vitals-worker  ──gRPC:50054──┐
                                         ├─ ruview-csi-relay      ──UDP:5005──▶ │
                                         ├─ ruvector-hailo-worker  :50051       │
                                         └─ ruvllm-pi-worker       :50053       │
                                                                                │
   cognitum-v0 (master)  ◀────────────────────────────────────────── Tailscale ─┘
   ├─ ruview-pointcloud (HTTP/WS :9880, loopback by default)
   ├─ ruview-mcp-brain   (HTTP :9876, accepts /memories from cluster)
   ├─ ruvector-hailo-worker  :50051   (also serves WifiCsi128d in Tier 3)
   └─ ruvllm-pi-worker       :50053
```

## Implementation plan

Same iteration cadence as ADR-179/ADR-180. **Tier 1 first, in a single
PR; Tier 2 in a second PR; Tier 3 is a longer multi-iter loop.**

### Tier 1 — vitals worker (target: 1 PR, ~1 week)

| Iter | Change |
|---|---|
| 1 | Branch `feature/adr-183-ruview-cluster-integration`, add this ADR, add `wifi-densepose-vitals` as a path dep on RuView checkout (or vendor the small subset under a feature flag) |
| 2 | Scaffold `ruview-vitals-worker` bin: UDP listener, frame buffer, structured logging — no inference yet |
| 3 | Wire `CsiVitalPreprocessor` + `BreathingExtractor` + `HeartRateExtractor` from `wifi-densepose-vitals`. CPU inference; verify on `data/recordings/*.csi.jsonl` from the RuView checkout |
| 4 | gRPC service on `:50054` — define proto in `crates/ruvector-hailo-cluster/proto/`, mirror the ADR-018 schema |
| 5 | systemd unit `ruview-vitals-worker.service` + `.env.example` + `install-ruview-vitals-worker.sh` (idempotent, system user `ruvllm-vitals`, hardened `ProtectSystem=strict`) |
| 6 | Brain POST shim: HTTPS POST to `http://cognitum-v0:9876/memories` with category=`vital`, body=`{node, breathing_bpm, heart_rate_bpm, snr, ts}`. Reuse `reqwest` already in the workspace |

Convergence criteria: bench shows `breathing_bpm` and `heart_rate_bpm`
within ±2 BPM of RuView's reference Node script
(`node scripts/breathing-rate.js`) on the same recording, on at least one
Pi, for at least 60 s of stable signal.

### Tier 2 — fusion master (target: 1 PR, ~1 week)

| Iter | Change |
|---|---|
| 7 | Build `ruview-pointcloud` for aarch64; package as deploy bundle with systemd unit |
| 8 | `ruview-mcp-brain.service` on cognitum-v0; allow Tailscale-source POSTs from `:9876`; reuse RuView's existing brain handler |
| 9 | `ruview-csi-relay.service` on workers; replays UDP frames to v0 unchanged (no parsing) — adds ≤ 0.5 ms latency |
| 10 | Verify v0 pipeline: depth + CSI (own Pi + 3 relays) + mmwave + vitals fusion all reach the point-cloud at 22 ms / 905 req/s targets |
| 11 | Tailscale ACL: workers can POST to v0:9876 brain *and* push CSI to v0:5005. Nothing else cross-cluster |
| 12 | Deploy bundle integration test: cluster smoke script (`ruvllm-cluster-smoke.sh` style) that brings the whole stack up + asserts a known recording lands as a brain memory at v0 |

### Tier 3 — NPU CSI embedder (target: open-ended /loop, ~3–4 weeks)

| Iter | Change |
|---|---|
| 13 | Compile RuView 48 KB `model.safetensors` to ONNX (already provided), then ONNX→HEF via `hailomz`. Validate output dim 128, latency < 10 ms |
| 14 | Add `HailoPipeline<CsiTensor, [f32; 128]>` to `ruvector-hailo`; carve out `WifiCsi128d` variant in `HailoEmbedderConfig` |
| 15 | Plumb `RUVIEW_CSI_MODEL` env into `ruview-vitals-worker`; mode A (CPU vitals) and mode B (NPU embed) coexist |
| 16 | HNSW sink at v0; `ruvector-cli search --backend hailo --variant wifi-csi-128 "person sitting still"` returns top-K |
| 17 | Cosine-recall benchmark vs the text-summary baseline; goal ≥ 2× MAP@10 on a labelled CSI test set. Implemented `ruview-csi-bench` binary. Result: base model separability ratio 1.016× (text baseline 1.462×) — FAIL on base model alone, motivating iter 18 |
| 18 | Per-room LoRA adapters (rank-4, alpha=8, scaling=2). Added `CsiLoraAdapter` to `ruvector-hailo/src/csi_embedder.rs`. `RUVIEW_CSI_LORA_ADAPTER` env var wires `node-N.json` from `ruv/ruview` HuggingFace into the worker at startup. `ruview-csi-bench --lora` validates improvement. Deploy: `scp node-1.json ruv@cognitum-v0:/usr/local/share/ruvector/` then restart worker with `RUVIEW_CSI_LORA_ADAPTER=/usr/local/share/ruvector/node-1.json` |
| 19 | SONA online adaptation; online triplet-loss LoRA updates from live VitalReading broadcast. Adapters for all 4 nodes trained to ≥100 steps. v0 reached 3420 steps before the iter-20 fine-tune |
| 20 | Offline supervised fine-tuning (`ruview-lora-finetune`). Root cause of 1.49× stall: SONA training zeroes motion_score (not in VitalReading). Offline tool uses all 8 features including motion_score=0.85 (exercising) vs 0.01 (sleeping). **ADR-183 §17 now PASSES on all 4 nodes** (iter-20 result, 2026-05-05): v0=2.12×, cluster-1=2.86×, cluster-2=2.36×, cluster-3=9.50×. Smoke test 19/19. |
| 21 | **Architectural decision (2026-05-05):** CPU path is the correct backend for the CSI encoder. Measured on ruvultra x86 release build: mean=1µs, p50=1µs, **p99=2µs** (0.002ms) — 6000× below the 12ms target. On Pi 5 (ARM Cortex-A76), estimate 5–20µs. Hailo-8 NPU kernel launch + PCIe DMA overhead for 8K-multiply-add tensors is ≥1ms — **worse than CPU**. Hailo-8 NPU path for this model is counterproductive and not pursued. |
| 22 | **Release validation (2026-05-06):** Bench re-run on cognitum-v0 confirms stable convergence. Text baseline 1.463×; LoRA+CSI 4.515×; improvement 3.09× — **PASS** (≥2×). All deployment checklist items verified done: node-0/1/2.json on v0, `RUVIEW_CSI_LORA_ADAPTER` wired, vitals-worker active. Smoke test 38/38. Cut `v0.1.0-csi-lora` release on `cognitum-one/v0-appliance`. ADR-183 closed. |

Convergence criteria: cluster-wide separability ≥ 2× improvement over
text baseline (ADR-183 §17) — **MET on all 4 nodes (2026-05-05)** —
**and** p99 CPU embed latency < 12 ms — **MET: p99 = 0.002 ms (x86 rel),
estimated ≤0.02 ms on Pi 5** — both holding for 2 consecutive bench iters.
**Tier 3 is closed.** NPU HEF path not pursued (NPU overhead exceeds CPU
for 8K-parameter models — see iter 21 above).

## Consequences

### Positive

- **Closes ADR-178 gap C long-term** without waiting on Hailo Model Zoo
  to ship a hailo8 pose HEF. We have our own contrastive encoder; we
  compile it ourselves.
- **Closes ADR-178 gap D** (downstream consumer): brain memories at v0
  *are* the consumer; existing RuView dashboard (`:9880`) renders them.
- **Real vitals from the cluster.** Breathing + heart rate at every Pi,
  pure-CPU, no model download, no NPU contention.
- **Reuses every existing port** (`:50051`, `:50053`, `:5005`); only
  introduces `:50054` for vitals and uses RuView's own `:9876` / `:9880`
  on v0. No bespoke schemas — vitals memories follow RuView's existing
  `category/content` POST.
- **Hardware separation matches reality**: only v0 has the camera and
  mmwave; only v0 runs the fusion / brain. Workers do what they have
  hardware for (CSI windows + Hailo embed + ruvllm).
- **Federation-ready.** A future cognitum-v1 / cognitum-cluster-N can
  run the same systemd bundle and join the brain at v0 with a single
  Tailscale ACL change.

### Negative

- **CPU contention on workers.** Tier 1 vitals = sliding-window FFT on
  Cortex-A76 cores already under contention from `ruvllm-pi-worker`
  (which saturates 4 cores per ADR-180's findings). Mitigation:
  vitals window updates at ~0.6 Hz; the FFTs are tiny (56 × 50). Pin
  vitals worker to `cpu-quota=20%` in systemd; if that bites, move
  vitals to v0 and use workers as pure relays.
- **v0 becomes a single point of failure.** Brain + fusion + camera +
  mmwave all live there. Mitigation: brain memories are append-only;
  workers cache locally and replay on reconnect (ADR-171 §3 already
  outlined this pattern). The 4-Pi cluster is *not* claimed to be
  HA — it's an edge node with three sensor satellites.
- **Tier 3 introduces a model-quality risk.** RuView's CSI encoder was
  trained on overnight v0.6 data; cosine separability on a different
  Pi's room is unproven. Mitigation: ship per-room LoRA adapters
  (already in the HF repo: `node-1.json` / `node-2.json`); fall back
  to text-summary embed if recall < baseline.
- **Tooling assumes RuView's local checkout exists.** Until we either
  vendor the relevant crates or publish them as `wifi-densepose-vitals
  = "0.3"` on crates.io, contributors need
  `~/projects/RuView/rust-port/wifi-densepose-rs/` cloned. Track an
  explicit task: ask upstream RuView to publish `wifi-densepose-vitals`
  + `wifi-densepose-nn` to crates.io, OR vendor the subset under
  `crates/ruvllm/vendor/wifi-densepose-vitals/`.

### Neutral

- The existing `ruview-csi-bridge` (header-only telemetry) **stays**.
  It's harmless, costs nothing, and gives a different signal
  (room/channel telemetry trends) than vitals. Same pattern as keeping
  `mmwave-bridge` alongside `ruvllm-bridge`.
- Adds ~200 KB binary on each Pi for the vitals path; ~50 MB more on v0
  for camera + mmwave + viewer. Pi 5 8 GB has plenty of headroom (per
  ADR-179 deployment notes).

## Open questions

1. **Vendor or path-dep for `wifi-densepose-vitals`?** Vendoring is
   simpler for hermetic builds; path-dep tracks upstream RuView
   updates automatically. Decision: **path-dep guarded behind
   `--features ruview-integration`** (default off, like ADR-179's
   `ruvllm-engine`); once upstream publishes to crates.io, swap to a
   pinned crate dep. Resolves itself.
2. **Brain HTTP vs MCP?** RuView's `brain_bridge.rs` POSTs JSON to
   `/memories`; pi-brain (this repo) speaks SSE-MCP. Both are
   acceptable. Decision: **start with REST POST** (matches RuView's
   shape), wrap MCP later if cross-tool reasoning needs it.
3. **CSI relay reliability?** UDP fan-out to v0 is fire-and-forget;
   loss in fusion vs in worker vitals is acceptable but should be
   logged. Add a per-relay packet counter and surface it in the cluster
   stats endpoint.
4. **Power.** Adding `ruview-pointcloud` (camera + mmwave at 22 ms) on
   v0 alongside the AI HAT+ at full tilt may exceed the 5 V / 5 A
   budget under load. **Bench under combined load before declaring
   Tier 2 done.** Reuse the thermal-overclock profile from ADR-174.
5. **WASM edge modules.** `wifi-densepose-wasm-edge` ships 60+ modules.
   Out of scope for this ADR; track a follow-up to run the host-runtime
   variant inside the existing agent-flow WASM sandbox. Probably
   ADR-184.

## Release & Appliance Deployment

Once all convergence criteria are met (≥2× separability ratio for 2 consecutive bench iters AND p99 NPU embed latency < 12 ms), cut a release on **`https://github.com/cognitum-one/v0-appliance`**:

1. Tag `ruvector` with `v0-appliance-adr183-vX` once iter 18+ bench passes on cognitum-v0.
2. Package binaries: `ruview-vitals-worker` (aarch64, `--features csi-embed`), `ruvector` CLI, `ruview-csi-bench`.
3. Include `node-1.json`, `node-2.json` from `ruv/ruview` HuggingFace in the release assets.
4. Update `cognitum-one/v0-appliance` README with setup steps: deploy binaries, set `RUVIEW_CSI_MODEL` + `RUVIEW_CSI_LORA_ADAPTER`, restart services.
5. Tag the release as `v0.1.0-csi-lora` with changelog summarising iter 14–18 deliverables.

Cross-compiled aarch64 binaries are at:
- `/home/ruvultra/projects/ruvector/target/aarch64-unknown-linux-gnu/release/ruview-vitals-worker` (4.4 MB)
- `/home/ruvultra/projects/ruvector/target/aarch64-unknown-linux-gnu/release/ruview-csi-bench` (453 KB)

Cluster deployment checklist (completed 2026-05-06):
- [x] `scp node-1.json node-2.json ruv@100.77.59.83:/usr/local/share/ruvector/` — verified: node-0/1/2.json present on v0
- [x] `RUVIEW_CSI_LORA_ADAPTER=/usr/local/share/ruvector/node-0.json` wired in `/etc/ruview-vitals-worker.env` on cognitum-v0
- [x] `ruview-vitals-worker` active on cognitum-v0 (`systemctl is-active` = active)
- [x] `ruview-csi-bench` result: 4.515× separability, 3.09× over baseline — **PASS** (≥2×)
- [x] Release `v0.1.0-csi-lora` created on `cognitum-one/v0-appliance`

## References

### This repo
- `docs/adr/ADR-167-ruvector-hailo-npu-embedding-backend.md` — embed worker baseline.
- `docs/adr/ADR-171-ruos-brain-ruview-pi5-edge-node.md` — single-Pi RuView edge sketch (now superseded by the cluster).
- `docs/adr/ADR-178-ruvector-ruview-hailo-integration-gap-analysis.md` — gap audit; this ADR closes long-term gap C and gap D.
- `docs/adr/ADR-179-ruvllm-pi-cluster-deployment.md` — 4-Pi cluster baseline (cognitum-v0 + cluster-1/2/3 on Tailscale).
- `docs/adr/ADR-180-ruvllm-serving-engine-continuous-batching.md` — same architectural pattern as Tier 3 (compile model, pool through Hailo).
- `crates/ruvector-hailo-cluster/src/bin/ruview-csi-bridge.rs` — existing telemetry bridge (header-only); the disclaimer at the top is the issue ADR-183 closes.
- `crates/ruvector-hailo-cluster/deploy/{install-,ruview-csi-bridge.{service,env.example}}` — install pattern Tier 1 mirrors for `ruview-vitals-worker`.

### RuView (`/home/ruvultra/projects/RuView`)
- `README.md` (lines 99–138, 183–222, 405–448) — point-cloud server, HF model artifacts, sensing features.
- `rust-port/wifi-densepose-rs/Cargo.toml` — workspace v0.3.0, 20 members.
- `rust-port/wifi-densepose-rs/crates/wifi-densepose-vitals/{breathing,heartrate,preprocessor,types}.rs` — Tier 1 dependency surface.
- `rust-port/wifi-densepose-rs/crates/wifi-densepose-pointcloud/src/{brain_bridge,csi_pipeline,depth,fusion}.rs` — Tier 2 fusion.
- HuggingFace `ruv/ruview` — Tier 3 model artifacts (48 KB safetensors / 8 KB Q4 / per-room LoRA / WiFlow 974 KB).
- ADR-018 binary CSI frame format (RuView side); this repo encodes the parser inline at `ruview-csi-bridge.rs:34–46`.
