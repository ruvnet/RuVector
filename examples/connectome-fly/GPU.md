# connectome-fly — GPU path (status)

## Summary

The `gpu-cuda` feature flag in `Cargo.toml` declares a GPU SDPA path for the motif-retrieval encoder. The stub in `src/analysis/gpu.rs::CudaBackend` is shipped in this commit; a fully-functional kernel is **not**. The CPU path (`CpuBackend`) remains the correctness reference and is exercised by every acceptance test.

## Why a stub

The current host (NVIDIA RTX 5080 / CUDA 13.0 driver / Linux 6.17) does not yet have a resolving `cudarc` release that exposes a public driver API stable across the CUDA 13 toolkit ABI. `cudarc` 0.13 / 0.14 series on crates.io bundle CUDA 12.x headers and the runtime kernel-launch surface assumes the 12.x driver layout. Attempting to compile against CUDA 13 yields `undefined reference` errors at link time on the NVRTC kernel bootstrap path.

Two fixes are known:

1. **Wait for `cudarc` with CUDA 13 support.** Tracked upstream in the `cudarc` 0.14+ milestone; no commitment on timing here.
2. **Pin a CUDA 12 toolkit alongside CUDA 13** and set `CUDA_PATH` to the 12.x install before building with `--features gpu-cuda`. Operational on the 9950X dev host but fragile across developer machines.

Rather than commit a half-working kernel, this ADR ships:

- A `ComputeBackend` trait (`src/analysis/gpu.rs`) that the CPU path already implements.
- A `CudaBackend::new() -> Result<Self, String>` constructor that returns an actionable error message when the GPU backend is unavailable.
- A `CudaBackend::sdpa_batch` impl that is `unimplemented!()` — fail-fast if anyone calls it by mistake.
- A `benches/gpu_sdpa.rs` that publishes the CPU number unconditionally and adds the `cuda` arm only if `CudaBackend::new()` succeeds.

## What lands in this commit

- The trait seam is stable. Production code that wants a backend calls `gpu::preferred_backend(&cfg)` and receives a `Box<dyn ComputeBackend>`. Today that is always `CpuBackend`.
- The bench compiles under `--features gpu-cuda` and runs — it just skips the CUDA arm.
- ADR-154 §12 documents the scope, target speedups, determinism contract, and positioning.

## What to do when `cudarc` is ready

The expected kernel is a batched SDPA: one block per `(batch, q_pos)`, lanes parallel over `kv_len` for the score dot-product, warp-reduce for the softmax normalizer, one final warp-level weighted sum over `d`. Target numerics: fp32, ≤ 1e-5 absolute error versus the CPU path.

Pseudocode (what to implement):

```rust
// In CudaBackend::new()
let ctx = cudarc::driver::CudaContext::new(0)?;
let module = ctx.load_ptx(include_str!("sdpa.ptx"), "sdpa_batch_kernel", &["sdpa_batch"])?;

// In sdpa_batch(&self, q, k, v, dims)
let q_dev = ctx.htod_copy(q)?;
let k_dev = ctx.htod_copy(k)?;
let v_dev = ctx.htod_copy(v)?;
let out_dev = ctx.alloc_zeros::<f32>(dims.batch * dims.q_len * dims.d)?;
let cfg = LaunchConfig {
    grid_dim: (dims.batch as u32, dims.q_len as u32, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 8 * dims.kv_len as u32, // scores buffer
};
unsafe {
    module.launch(
        cfg, (&q_dev, &k_dev, &v_dev, &out_dev, dims.kv_len as u32, dims.d as u32)
    )?;
}
let out = ctx.sync_reclaim(out_dev)?;
out
```

The PTX kernel itself is ~80 lines of CUDA C (classical flash-attention pattern, single-head). A vanilla version is adequate; fused variants are future work.

## How to verify without the kernel

The CPU path's determinism and correctness are covered by:

- `src/analysis/gpu.rs::tests::cpu_sdpa_is_deterministic` — bit-identical on repeat.
- `src/analysis/gpu.rs::tests::cpu_sdpa_weighted_value_in_range` — uniform-attention sanity check.

Once the CUDA kernel lands, add a cross-backend test:

```rust
#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_sdpa_matches_cpu_within_tolerance() {
    let dims = Dims { q_len: 1, kv_len: 10, d: 64, batch: 100 };
    // deterministic q/k/v fill
    let cpu = CpuBackend.sdpa_batch(&q, &k, &v, dims);
    let gpu = CudaBackend::new().unwrap().sdpa_batch(&q, &k, &v, dims);
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}
```

## Positioning

GPU is **scaling infrastructure**, not a correctness claim (ADR-154 §12.4). Nothing in the ADR's acceptance-criterion matrix depends on the GPU backend. When `cudarc` support lands, the bench in `BENCHMARK.md §8` will publish a CPU/GPU speedup number alongside the current CPU baseline. Until then, the CPU path is what this example is.

## References

- ADR-154 §6.4 "GPU acceleration path" (deferred scope).
- ADR-154 §12 "GPU acceleration path" (expanded in commit 2).
- `src/analysis/gpu.rs` — the `ComputeBackend` trait and `CpuBackend` / `CudaBackend` stubs.
- `benches/gpu_sdpa.rs` — the GPU/CPU comparison bench (runs under `--features gpu-cuda`).
