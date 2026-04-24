# ADR-157: Optional Accelerator Plane — `VectorKernel` Trait + Dispatch

## Status

**Proposed** — scaffolding-only decision. No kernel implementations
commit to this ADR; implementations ship on their own cadence in
separate crates and must pass the acceptance test below to be
promoted past experimental.

## Date

2026-04-23

## Authors

ruv.io · RuVector research. Strategic review 2026-04-22 (GPU note)
proposed the shape; this ADR locks the trait + dispatch contract and
the acceptance gate.

## Relates To

- ADR-154 — RaBitQ rotation-based 1-bit quantization (the kernels)
- ADR-155 — ruLake: cache-first vector execution fabric (the consumer)
- ADR-156 — ruLake as substrate for agent brain systems (the user story)

---

## Context

The measured intermediary tax on a cache hit is 1.02× direct RaBitQ
(`crates/ruvector-rulake/BENCHMARK.md`). The cache is not the
bottleneck. The bottleneck when it appears is in the **kernel** —
RaBitQ popcount scan + exact L2² rerank — which scales with
`n × D` for scan and `rerank_factor × k × D` for rerank. Both
become load-bearing at high dim (D ≥ 768), large collections
(n ≥ 1M), or large batch (`n_queries ≥ 256` per wave).

Three kinds of deployment target the same code:

| Target        | Accelerator                                   |
|---------------|-----------------------------------------------|
| Laptop / dev  | CPU only (scalar + portable SIMD)             |
| Server / CI   | CPU + AVX-512 or NEON                         |
| Cognitum box  | CPU + CUDA / ROCm / Metal                     |
| Browser / edge| WASM SIMD (no GPU)                            |

If the kernel is pluggable and the dispatch is explicit, every target
runs the same crate surface and we hold determinism + witness across
all of them. If we let GPU implementations creep into the core crate,
laptop and WASM builds either fail or ship dead-weight bindings.

## Decision

**An optional accelerator plane: a `VectorKernel` trait in
`ruvector-rabitq` plus explicit dispatch in `ruvector-rulake`, with
determinism as a hard gate on witness-sealed output.**

### Where each piece lives

| Concern                            | Crate                   | Rationale                                                                 |
|------------------------------------|-------------------------|---------------------------------------------------------------------------|
| `VectorKernel` trait               | `ruvector-rabitq`       | Kernels are RaBitQ primitives. Cache is a consumer, not a kernel host.    |
| Default CPU kernel                 | `ruvector-rabitq`       | Wraps existing `symmetric_scan_topk` + rerank. Always available.          |
| SIMD kernel                        | `ruvector-rabitq`, feature-gated | Portable SIMD via `std::simd` when stable; AVX-512/NEON intrinsics otherwise. |
| GPU kernels (CUDA, ROCm, Metal)    | **Separate crates** (`ruvector-rabitq-cuda`, etc.) | Each has its own CI matrix, driver dependency, and license footprint. |
| WASM SIMD                          | Feature-gated in `ruvector-rabitq`   | No new crate; it's the same source compiled with `--target=wasm32-*`.      |
| Dispatch policy                    | `ruvector-rulake`       | Uses live signals (batch size, hit rate, rerank pressure) only the cache has. |
| Kernel registration + caps query   | `ruvector-rulake`       | `RuLake::register_kernel(Arc<dyn VectorKernel>)` mirrors `register_backend`. |

### Trait shape (normative)

```rust
/// A vector kernel executes popcount scans and exact L2² rerank for
/// one or more queries against a compressed RaBitQ index. The kernel
/// is stateless w.r.t. the index — the index lives in the cache and
/// is passed in by reference on every call. This keeps GPU kernels
/// from needing to own index lifetime.
pub trait VectorKernel: Send + Sync {
    /// Stable identifier surfaced in stats + logs.
    fn id(&self) -> &str;

    /// Advertise what this kernel can do.
    fn caps(&self) -> KernelCaps;

    /// Top-k scan for one or more queries. The kernel is responsible
    /// for returning results in the same order as `queries`; the
    /// cache's pos→id mapping applied by the caller.
    fn scan(&self, req: ScanRequest<'_>) -> Result<ScanResponse>;
}

#[derive(Debug, Clone, Copy)]
pub struct KernelCaps {
    /// Minimum batch size at which this kernel is ever chosen.
    /// CPU kernels report 1; GPU kernels typically ≥64.
    pub min_batch: usize,
    /// Maximum D the kernel supports without fallback.
    pub max_dim: usize,
    /// Does the kernel produce byte-identical output vs the reference
    /// CPU kernel? Only deterministic kernels can feed witness-sealed
    /// outputs. Non-deterministic kernels (float-reorder GPU) are fine
    /// for "Eventual" / recall-tolerant paths but must be filtered out
    /// on "Fresh" or "Frozen" paths.
    pub deterministic: bool,
    /// Symbolic accelerator label: "cpu", "cpu-simd", "cuda", "metal",
    /// "wasm-simd", etc. Surfaced in stats.
    pub accelerator: &'static str,
}
```

### Dispatch policy (normative)

```rust
impl RuLake {
    fn pick_kernel(&self, batch_size: usize, dim: usize, frozen: bool) -> Arc<dyn VectorKernel> {
        // 1. Always honor Fresh / Frozen determinism requirement.
        let deterministic_required = frozen || self.consistency == Consistency::Fresh;
        for k in self.kernels_by_preference() {
            let c = k.caps();
            if batch_size < c.min_batch { continue; }
            if dim > c.max_dim { continue; }
            if deterministic_required && !c.deterministic { continue; }
            return k;
        }
        self.default_cpu_kernel()
    }
}
```

Preference order (strictly most-accelerated first, CPU as fallback):
`cuda`/`rocm`/`metal` → `cpu-simd` → `cpu`. Kernels are registered by
the operator; ruLake does not ship with GPU kernels enabled.

### Determinism as a hard gate

Witness-sealed output must be bit-reproducible across kernels. Two
rules:

1. **Scan phase must be deterministic on all kernels.** The 1-bit
   popcount Hamming distance is integer math; every kernel must
   produce the same set of candidates in the same order.
2. **Rerank phase may be float-nondeterministic.** Exact L2² depends
   on reduction order, which GPU kernels may reorder. Kernels that
   can't guarantee identical rerank output for tied scores set
   `caps().deterministic = false`, and the dispatch policy refuses
   to use them on `Consistency::Fresh` or `Consistency::Frozen`
   paths. They stay available for `Consistency::Eventual`, which
   tolerates recall drift by design.

The witness chain is NOT recomputed per kernel; it stays anchored on
`(data_ref, dim, rotation_seed, rerank_factor, generation)` as
before. Kernel identity is surfaced in stats, not in the witness.

### Acceptance test

A new GPU kernel is **promoted past experimental** iff, on a fixed
hardware class + a fixed dataset (clustered D=768 n=1M rerank×20 as
the reference), it demonstrates:

> **Either** p95 query latency ≥ 2× lower than the reference CPU
> kernel at identical recall@10, **or** cost per 1M queries ≥ 30%
> lower at identical recall@10.

Plus:

- Deterministic output on the scan phase (bit-exact set of top-k
  candidates pre-rerank).
- Test coverage: the kernel must pass the full rulake smoke suite.
- Memory safety: the kernel must not leak more than 2× the index
  bytes during steady-state serving.

A kernel failing the acceptance gate stays in its experimental crate.
It does not land in the default dispatch preference; operators who
want it enable it explicitly.

## Alternatives considered

### A. Single-crate kernels (CUDA/ROCm/Metal all inside `ruvector-rabitq`)

Rejected: every GPU dep is 1k+ lines of FFI + driver matrix + its own
CI pain. Pulling CUDA into `ruvector-rabitq` breaks laptop and WASM
builds unless everything is feature-gated, and feature-gate matrices
are a maintenance sink. Separate crates amortize the pain and let
customers pick the one their platform supports.

### B. Dispatch inside `ruvector-rabitq`

The rabitq crate sees single-index single-query calls and has no
visibility into batch size or hit rate — the two signals that
actually determine the CPU/GPU crossover. Dispatch belongs in the
caller, which is ruLake. rabitq exposes kernels; ruLake picks.

### C. VectorKernel in a new `ruvector-kernel` crate

Would be cleaner if multiple non-RaBitQ index types needed the same
kernel shape. Today only RaBitQ exists. Defer the split until a
second consumer (hnsw, PQ) appears; over-engineer warning.

### D. No trait, just static dispatch via feature flags

Simplest possible: `cfg(feature = "cuda")` picks the CUDA kernel,
`cfg(feature = "simd")` picks SIMD. Rejected because it bakes the
choice at build time — a single binary that detects the available
accelerator at runtime cannot switch. Trait-based dispatch lets a
laptop binary detect "no GPU, use CPU" and a server binary detect
"GPU present, batch size ≥ 64 so use it."

### E. Build on `wgpu` for portable GPU

WebGPU is maturing but not yet a viable RaBitQ backend — its shader
model does not yet expose the popcount + reduction primitives we'd
need to match native CUDA/Metal on the scan phase. Kept on the v2
radar; the trait is designed to accommodate a `wgpu` kernel when
wgpu stabilizes.

## Consequences

### Positive

- **Laptop / server / edge / Cognitum run the same source.** No
  per-target forks.
- **GPU is an opt-in, not a default.** Customers who don't have GPUs
  don't pay for the code path or the build-time complexity.
- **Determinism is preserved by default.** The witness chain stays
  valid on every target; only non-deterministic kernels on Eventual
  paths can diverge, and they do so explicitly.
- **Acceptance gate stops speculation.** 2× p95 or 30% cost is the
  only path to default preference; absent that, GPU kernels stay
  experimental no matter how much engineering time they absorbed.

### Negative

- **Adds a plug-in surface.** Kernel registration, capability query,
  dispatch policy — all new code paths. Mitigation: the default
  dispatch is "use the CPU kernel"; everything else is strictly
  additive.
- **Kernel nondeterminism is a sharp edge.** Operators who enable a
  non-deterministic GPU kernel on an `Eventual` collection will see
  different top-k orderings across restarts. The caps bit is the
  warning, but it's easy to miss. Document it as a first-class
  deployment concern.
- **Two-repo maintenance per GPU backend.** Each of `ruvector-rabitq-cuda`
  / `-rocm` / `-metal` is its own release train. Acceptable given
  how different their driver stories are, but real cost.

### Neutral

- Existing `RabitqPlusIndex::search` and `search_with_rerank` stay —
  the trait wraps them as the default CPU kernel. No API breakage.
- The trait shape is designed to accommodate `wgpu` when it matures;
  no ADR revision needed to add WebGPU later.

## Open questions

1. **Where does the WASM SIMD build live?** Feature gate in
   `ruvector-rabitq` or separate `ruvector-rabitq-wasm` crate? Leaning
   feature gate — it's the same source — but the WASM story has its
   own CI pipeline concerns.
2. **Does kernel identity enter the witness?** Current decision says no
   (kernels are execution substrate, data is witness-sealed), but a
   customer might want "only results from deterministic kernels are
   auditable." That's a caps-based filter, not a witness-chain change.
3. **Batch size autotuning.** `min_batch` in `KernelCaps` is a
   conservative static hint. Real crossover depends on D, n, rerank
   pressure. Deferred: tune empirically once two kernels exist to
   measure against each other.
4. **Does `Consistency::Frozen` forbid non-deterministic kernels
   outright, or just warn?** Current design: forbid. The whole point
   of Frozen is audit-tier reproducibility. Revisit if a customer
   has a use case.
5. **Do we expose `KernelCaps` in `CacheStats`?** Useful for
   operators — "which kernel answered the last 1,000 queries?" —
   but scope-creep toward observability features that should live
   in the caller's tracing layer.
