//! Native vs. WASM (wasmtime) latency for Ed25519 witness-record signing
//! (ADR-361, docs/research/nightly/2026-10-17-wasm-witness-signing-cost/).
//!
//! Loads the actual compiled `rvf-wasm` artifact (built with the opt-in
//! `witness-signing` feature) and calls its exported
//! `rvf_witness_ed25519_sign` / `rvf_witness_ed25519_verify` functions
//! through a real `wasmtime` host instance, comparing against the same
//! `rvf_types::ed25519` primitive called natively.
//!
//! Build the WASM artifact first (from `crates/rvf`):
//! ```bash
//! cargo build -p rvf-wasm --release --target wasm32-unknown-unknown --features witness-signing
//! ```
//! then run this bench from `crates/rvf`:
//! ```bash
//! cargo bench -p rvf-benches --bench witness_wasm_bench
//! ```
//! Override the artifact path with `RVF_WASM_PATH` if it is not at the
//! default `target/wasm32-unknown-unknown/release/rvf_wasm.wasm`.

use criterion::{criterion_group, criterion_main, Criterion};
use rvf_types::ed25519::{
    ed25519_sign, ed25519_verify, Ed25519Keypair, SECRET_KEY_SIZE, SIGNATURE_SIZE,
};
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

// Must match crates/rvf/rvf-wasm/src/memory.rs's MESSAGE_IO region, which
// witness.rs reuses for the signing I/O buffer (see witness.rs module doc).
const MESSAGE_IO_OFFSET: usize = 0x0800;
const KEY_OFFSET: usize = MESSAGE_IO_OFFSET;
const MSG_OFFSET: usize = MESSAGE_IO_OFFSET + 32;
const SIG_OFFSET: usize = MESSAGE_IO_OFFSET + 2048 - SIGNATURE_SIZE;

/// Representative witness-record payload size (matches the 73-byte
/// `WitnessEntry` encoding in rvf-crypto::witness, rounded to a plausible
/// signed-record size with headroom for a context tag).
const MESSAGE_LEN: usize = 96;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn fixed_message(seed: u64) -> [u8; MESSAGE_LEN] {
    let mut rng = Lcg(seed);
    let mut buf = [0u8; MESSAGE_LEN];
    for chunk in buf.chunks_mut(8) {
        let bytes = rng.next_u64().to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&bytes[..n]);
    }
    buf
}

fn wasm_artifact_path() -> String {
    std::env::var("RVF_WASM_PATH")
        .unwrap_or_else(|_| "../target/wasm32-unknown-unknown/release/rvf_wasm.wasm".to_string())
}

struct WasmWitness {
    store: Store<()>,
    memory: Memory,
    sign_fn: TypedFunc<i32, i32>,
    verify_fn: TypedFunc<i32, i32>,
}

impl WasmWitness {
    fn load() -> Self {
        let path = wasm_artifact_path();
        let engine = Engine::default();
        let module = Module::from_file(&engine, &path).unwrap_or_else(|e| {
            panic!(
                "failed to load {path}: {e}\n\
                 Build it first: cargo build -p rvf-wasm --release \
                 --target wasm32-unknown-unknown --features witness-signing"
            )
        });
        let mut store = Store::new(&engine, ());
        let instance =
            Instance::new(&mut store, &module, &[]).expect("instantiate rvf-wasm module");
        let memory = instance
            .get_memory(&mut store, "memory")
            .expect("exported linear memory");
        let sign_fn = instance
            .get_typed_func::<i32, i32>(&mut store, "rvf_witness_ed25519_sign")
            .expect("rvf_witness_ed25519_sign export (build with --features witness-signing)");
        let verify_fn = instance
            .get_typed_func::<i32, i32>(&mut store, "rvf_witness_ed25519_verify")
            .expect("rvf_witness_ed25519_verify export (build with --features witness-signing)");
        Self {
            store,
            memory,
            sign_fn,
            verify_fn,
        }
    }

    fn sign(&mut self, secret: &[u8; SECRET_KEY_SIZE], msg: &[u8]) -> [u8; SIGNATURE_SIZE] {
        self.memory
            .write(&mut self.store, KEY_OFFSET, secret)
            .unwrap();
        self.memory.write(&mut self.store, MSG_OFFSET, msg).unwrap();
        let rc = self
            .sign_fn
            .call(&mut self.store, msg.len() as i32)
            .unwrap();
        assert_eq!(rc, 0, "wasm sign failed");
        let mut sig = [0u8; SIGNATURE_SIZE];
        self.memory.read(&self.store, SIG_OFFSET, &mut sig).unwrap();
        sig
    }

    fn verify(&mut self, public: &[u8; 32], msg: &[u8], sig: &[u8; SIGNATURE_SIZE]) -> bool {
        self.memory
            .write(&mut self.store, KEY_OFFSET, public)
            .unwrap();
        self.memory.write(&mut self.store, MSG_OFFSET, msg).unwrap();
        self.memory.write(&mut self.store, SIG_OFFSET, sig).unwrap();
        self.verify_fn
            .call(&mut self.store, msg.len() as i32)
            .unwrap()
            == 1
    }
}

/// Cross-checks native <-> wasm wire compatibility before benchmarking.
/// Panics loudly (rather than silently skipping) on any mismatch.
fn assert_cross_compatible(wasm: &mut WasmWitness, kp: &Ed25519Keypair, msg: &[u8]) {
    let native_sig = ed25519_sign(&kp.secret_key(), msg);
    assert!(
        wasm.verify(&kp.public_key(), msg, &native_sig),
        "wasm rejected a native-produced signature"
    );

    let wasm_sig = wasm.sign(&kp.secret_key(), msg);
    assert!(
        ed25519_verify(&kp.public_key(), msg, &wasm_sig),
        "native rejected a wasm-produced signature"
    );
    assert_eq!(native_sig, wasm_sig, "native and wasm signatures diverge for identical input (should be byte-identical: Ed25519 is deterministic)");
    println!("[witness_wasm_bench] native <-> wasm cross-compatibility: PASS");
}

fn bench_sign(c: &mut Criterion) {
    struct TestRng(u64);
    impl rand_core::RngCore for TestRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let mut i = 0;
            while i < dest.len() {
                let v = self.next_u64().to_le_bytes();
                let n = (dest.len() - i).min(8);
                dest[i..i + n].copy_from_slice(&v[..n]);
                i += n;
            }
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }
    impl rand_core::CryptoRng for TestRng {}

    let mut rng = TestRng(0xA11CE_B0B_1234_5678);
    let kp = Ed25519Keypair::generate(&mut rng);
    let msg = fixed_message(0xDEAD_BEEF);
    let mut wasm = WasmWitness::load();

    assert_cross_compatible(&mut wasm, &kp, &msg);

    let secret = kp.secret_key();
    let public = kp.public_key();
    let native_sig = ed25519_sign(&secret, &msg);

    let mut group = c.benchmark_group("witness_sign");
    group.bench_function("native", |b| b.iter(|| ed25519_sign(&secret, &msg)));
    group.bench_function("wasm_wasmtime", |b| b.iter(|| wasm.sign(&secret, &msg)));
    group.finish();

    let mut group = c.benchmark_group("witness_verify");
    group.bench_function("native", |b| {
        b.iter(|| ed25519_verify(&public, &msg, &native_sig))
    });
    group.bench_function("wasm_wasmtime", |b| {
        b.iter(|| wasm.verify(&public, &msg, &native_sig))
    });
    group.finish();
}

criterion_group!(benches, bench_sign);
criterion_main!(benches);
