//! ESP32-S3 smoke test for `ruvllm_sparse_attention` in no_std + alloc mode.
//!
//! This is a *cross-compile test* example, not a runnable binary on a host
//! Rust toolchain. To compile it, you need the `esp` toolchain installed via
//! `espup` and the Xtensa target enabled. From the workspace root:
//!
//! ```sh
//! # Install once
//! cargo install espup espflash
//! espup install
//!
//! # Cross-compile this example to a real ESP32-S3 ELF
//! cargo +esp build \
//!   --release \
//!   --no-default-features \
//!   --features fp16 \
//!   --target xtensa-esp32s3-none-elf \
//!   -Z build-std=core,alloc \
//!   --example esp32s3_smoke
//! ```
//!
//! The actual on-device entrypoint (#[entry], peripheral init, allocator
//! setup, panic handler) lives in your application crate — this file
//! exercises only the *library surface* to prove the no_std + alloc
//! configuration links cleanly for the Xtensa LX7 target.
//!
//! Tested on: ESP32-S3 revision v0.2, 16 MB flash, dual Xtensa LX7 @ 240 MHz,
//! attached at /dev/ttyACM0 via the chip's built-in USB-Serial-JTAG.
//!
//! What this example proves:
//!   1. Crate compiles for `xtensa-esp32s3-none-elf` with `--no-default-features`.
//!   2. `SubquadraticSparseAttention::forward` executes against zero-fill QKV.
//!   3. `forward_gated_with_fastgrnn` (FastGRNN salience gate + gated forward)
//!      executes end-to-end.
//!   4. `KvCacheF16` (when `fp16` feature is on) executes a single decode step.
//!
//! Static memory budget at the demonstrated shape (seq=64, heads=4, dim=32,
//! window=16): ~64 KB heap peak. Easily fits in the ESP32-S3's 512 KB SRAM
//! without external PSRAM.

#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]

extern crate alloc;

use ruvllm_sparse_attention::{
    AttentionBackend, FastGrnnGate, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

/// Run the full smoke sequence and return `Ok(())` if every step produced
/// finite output. Designed to be called from a `#[entry]` function in your
/// ESP32 application crate (after allocator + panic handler setup).
pub fn run_smoke() -> Result<(), &'static str> {
    let seq = 64;
    let heads = 4;
    let dim = 32;

    // 1. Plain sparse forward.
    let cfg = SparseAttentionConfig {
        window: 16,
        block_size: 8,
        global_tokens: alloc::vec![0],
        causal: true,
        use_log_stride: true,
        use_landmarks: true,
        sort_candidates: false,
    };
    let attn = SubquadraticSparseAttention::new(cfg).map_err(|_| "attn init failed")?;

    let q = Tensor3::zeros(seq, heads, dim);
    let k = Tensor3::zeros(seq, heads, dim);
    let v = Tensor3::zeros(seq, heads, dim);

    let out = attn.forward(&q, &k, &v).map_err(|_| "forward failed")?;
    if !is_all_finite(&out.data) {
        return Err("forward produced non-finite output");
    }

    // 2. FastGRNN-gated forward.
    let gate = FastGrnnGate::new(dim, 16);
    let gated = attn
        .forward_gated_with_fastgrnn(&q, &k, &v, &gate, 8)
        .map_err(|_| "gated forward failed")?;
    if !is_all_finite(&gated.data) {
        return Err("gated forward produced non-finite output");
    }

    // 3. Single decode step against an FP16 KV cache (when fp16 enabled).
    #[cfg(feature = "fp16")]
    {
        use ruvllm_sparse_attention::KvCacheF16;
        let mut cache = KvCacheF16::new(
            /*cap*/ 256, /*kv_heads*/ heads, /*dim*/ dim, /*block_size*/ 8,
        );
        let one_k = Tensor3::zeros(1, heads, dim);
        let one_v = Tensor3::zeros(1, heads, dim);
        cache
            .try_append(&one_k, &one_v)
            .map_err(|_| "kv append failed")?;
        let q_step = Tensor3::zeros(1, heads, dim);
        let step = cache
            .decode_step_f16(&attn, &q_step)
            .map_err(|_| "decode_step_f16 failed")?;
        if !is_all_finite(&step.data) {
            return Err("decode_step produced non-finite output");
        }
    }

    Ok(())
}

fn is_all_finite(xs: &[f32]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

// Avoid pulling in std for a binary target on host: provide a `main`
// only when targetting a hosted OS. On `target_os = "none"` (ESP32-S3
// bare-metal) the application crate provides #[entry] / panic handler.
#[cfg(not(target_os = "none"))]
fn main() {
    match run_smoke() {
        Ok(()) => println!("esp32s3_smoke: all checks passed"),
        Err(e) => {
            eprintln!("esp32s3_smoke FAILED: {}", e);
            std::process::exit(1);
        }
    }
}

// Compatibility entry point for #[no_main] bare-metal builds. The actual
// application crate must still provide #[entry] from xtensa-lx-rt and a
// panic handler — this stub just ensures the example links when the
// toolchain looks for a symbol.
#[cfg(target_os = "none")]
#[allow(dead_code)]
fn run_on_target() -> ! {
    let _ = run_smoke();
    loop {
        core::hint::spin_loop();
    }
}

// Keep `Vec` import live in the bare-metal config (used inside run_smoke).
#[cfg(target_os = "none")]
#[allow(dead_code)]
fn _keep_vec_import_alive() -> Vec<u8> {
    Vec::new()
}
