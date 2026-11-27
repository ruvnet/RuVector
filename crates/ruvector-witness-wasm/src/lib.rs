//! A minimal, self-contained port of `ruvector-agent-memory`'s ADR-347
//! span-signed witness chain design (`witness_signing.rs`), reduced to the
//! parts that matter for one question: does the design's measured
//! `PerRecord`-vs-`BatchTail` amortization advantage survive compilation to
//! `wasm32-unknown-unknown`, and what does that cost in binary size and
//! sandboxed latency relative to native?
//!
//! This crate intentionally does **not** depend on `ruvector-agent-memory`
//! (its `DashMap`/`roaring` dependencies are not wasm32-friendly) or
//! reproduce its full ledger API. It ports only the signing scheme:
//! records are grouped into spans, each span's SHA-256 digest over its
//! records is Ed25519-signed together with the previous span's chained
//! link, exactly mirroring `witness_signing.rs`'s message layout. The
//! domain tags below are distinct from that crate's, so a signature
//! produced here can never be confused with (or replayed as) a real TARL
//! ledger signature.
//!
//! See `examples/wasm_witness_bench.rs` for the benchmark that compiles
//! this crate to `wasm32-unknown-unknown`, loads it with `wasmtime`, and
//! compares it against the native (`rlib`) build of the same functions.

use rvf_types::ed25519::{ed25519_sign, ed25519_verify, Ed25519Keypair, SECRET_KEY_SIZE};
use rvf_types::sha256::{sha256, Sha256};

/// Deterministic signing key, matching the fixed-seed convention used by
/// the 2026-09-16 nightly's `witness_signing_bench` ("fix random seeds").
pub const FIXED_SECRET: [u8; SECRET_KEY_SIZE] = [11u8; SECRET_KEY_SIZE];

/// One record's canonical size, matching `LedgerWitnessRecord::to_bytes()`'s
/// 64-byte encoding in `ruvector-agent-memory::ops` (same order of
/// magnitude, not byte-identical -- this crate owns its own deterministic
/// synthetic record, not the real schema).
pub const RECORD_SIZE: usize = 64;

const MSG_TAG: &[u8] = b"ruvector-witness-wasm:v1:span\0";
const RECORDS_TAG: &[u8] = b"ruvector-witness-wasm:v1:records\0";
/// `purpose` (1) + `from`/`to` (2 x 8 LE) + `prev_link` (32) + `records_digest` (32).
const MESSAGE_LEN: usize = MSG_TAG.len() + 1 + 16 + 32 + 32;

/// Which signing strategy to run, mirroring `witness_signing::SigningStrategy`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Strategy {
    /// Sign every record's own span (span length 1).
    PerRecord = 1,
    /// Sign only the tail span of every `batch_size`-record run.
    BatchTail = 2,
}

impl Strategy {
    fn span_len(self, batch_size: u32) -> u32 {
        match self {
            Strategy::PerRecord => 1,
            Strategy::BatchTail => batch_size.max(1),
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn from_i32(v: i32) -> Strategy {
        if v == Strategy::BatchTail as i32 {
            Strategy::BatchTail
        } else {
            Strategy::PerRecord
        }
    }
}

/// One signed span: "records `[from, to]` have this SHA-256 digest, and
/// this span directly follows the span whose chained link was `prev_link`
/// at signing time" (the `prev_link` itself is not stored -- the verifier
/// re-derives it by walking the chain, same as `witness_signing.rs`).
#[derive(Clone, Copy)]
pub struct SignedSpan {
    pub purpose: u8,
    pub from: u32,
    pub to: u32,
    pub records_digest: [u8; 32],
    pub signature: [u8; 64],
}

fn build_message(
    purpose: u8,
    from: u32,
    to: u32,
    prev_link: &[u8; 32],
    digest: &[u8; 32],
) -> [u8; MESSAGE_LEN] {
    let mut m = [0u8; MESSAGE_LEN];
    let mut off = 0usize;
    m[off..off + MSG_TAG.len()].copy_from_slice(MSG_TAG);
    off += MSG_TAG.len();
    m[off] = purpose;
    off += 1;
    m[off..off + 8].copy_from_slice(&(from as u64).to_le_bytes());
    off += 8;
    m[off..off + 8].copy_from_slice(&(to as u64).to_le_bytes());
    off += 8;
    m[off..off + 32].copy_from_slice(prev_link);
    off += 32;
    m[off..off + 32].copy_from_slice(digest);
    m
}

/// A tiny deterministic xorshift64 stream, same construction as the
/// `TestRng` used by `rvf-types/src/ed25519.rs`'s own test suite -- reused
/// here as a *generator*, not a cryptographic primitive, purely to produce
/// reproducible synthetic record bytes without pulling in a `rand` crate.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Deterministic synthetic record `i`: the same (seed, i) pair always
/// produces the same 64 bytes, on any target.
pub fn record_bytes(i: u32) -> [u8; RECORD_SIZE] {
    let mut rng =
        XorShift64::new(0xD1CE_1234_5678_9ABCu64 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut out = [0u8; RECORD_SIZE];
    let mut off = 0usize;
    while off < RECORD_SIZE {
        let v = rng.next_u64().to_le_bytes();
        let take = (RECORD_SIZE - off).min(8);
        out[off..off + take].copy_from_slice(&v[..take]);
        off += take;
    }
    out
}

fn public_key() -> [u8; 32] {
    Ed25519Keypair::from_secret(&FIXED_SECRET).public_key()
}

/// Result of an honest (untampered) run: how many signatures were
/// produced, and how many of them failed to verify (0 = fully correct).
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkloadResult {
    pub signature_count: u32,
    pub verify_failures: u32,
}

/// Runs the deterministic `n`-record workload under `strategy`, signing
/// and immediately verifying every span. This is the function timed by
/// the benchmark, both natively (direct call) and inside `wasmtime` (via
/// the `wasm32` export below).
pub fn run_workload(strategy: Strategy, n: u32, batch_size: u32) -> WorkloadResult {
    let span_len = strategy.span_len(batch_size);
    let purpose = strategy as u8;
    let pubkey = public_key();

    let mut prev_link = [0u8; 32];
    let mut hasher = Sha256::new();
    hasher.update(RECORDS_TAG);
    let mut from = 0u32;
    let mut covered = 0u32;
    let mut result = WorkloadResult::default();

    for i in 0..n {
        hasher.update(&record_bytes(i));
        covered += 1;
        let is_last = i + 1 == n;
        if covered == span_len || is_last {
            let to = i;
            let digest = hasher.clone().finalize();
            let message = build_message(purpose, from, to, &prev_link, &digest);
            let signature = ed25519_sign(&FIXED_SECRET, &message);
            result.signature_count += 1;
            if !ed25519_verify(&pubkey, &message, &signature) {
                result.verify_failures += 1;
            }
            prev_link = sha256(&message);
            hasher = Sha256::new();
            hasher.update(RECORDS_TAG);
            from = to + 1;
            covered = 0;
        }
    }
    result
}

/// Builds the full signed span chain for `n` records under `strategy`,
/// flips one byte in the middle record, and checks whether re-deriving
/// that span's `records_digest` from the (now tampered) record bytes
/// still matches the digest that was actually signed. A faithful forgery
/// must change the digest (the same "diligent forgery" property the
/// 2026-09-16 nightly's `witness_signing.rs` test exercises) -- returns
/// `true` when the tamper is detected (the expected, correct outcome).
pub fn tamper_check(strategy: Strategy, n: u32, batch_size: u32) -> bool {
    if n == 0 {
        return true;
    }
    let span_len = strategy.span_len(batch_size);
    let purpose = strategy as u8;

    let mut records: Vec<[u8; RECORD_SIZE]> = Vec::with_capacity(n as usize);
    for i in 0..n {
        records.push(record_bytes(i));
    }

    let mut prev_link = [0u8; 32];
    let mut spans: Vec<SignedSpan> = Vec::new();
    let mut hasher = Sha256::new();
    hasher.update(RECORDS_TAG);
    let mut from = 0u32;
    let mut covered = 0u32;

    for (i, rec) in records.iter().enumerate() {
        hasher.update(rec);
        covered += 1;
        let is_last = (i as u32) + 1 == n;
        if covered == span_len || is_last {
            let to = i as u32;
            let digest = hasher.clone().finalize();
            let message = build_message(purpose, from, to, &prev_link, &digest);
            let signature = ed25519_sign(&FIXED_SECRET, &message);
            spans.push(SignedSpan {
                purpose,
                from,
                to,
                records_digest: digest,
                signature,
            });
            prev_link = sha256(&message);
            hasher = Sha256::new();
            hasher.update(RECORDS_TAG);
            from = to + 1;
            covered = 0;
        }
    }

    // Tamper exactly one byte of the middle record.
    let mid = (n / 2) as usize;
    records[mid][0] ^= 0xFF;

    let span = spans
        .iter()
        .find(|s| (s.from as usize) <= mid && mid <= (s.to as usize))
        .expect("every record is covered by exactly one span");

    let mut h = Sha256::new();
    h.update(RECORDS_TAG);
    for rec in &records[(span.from as usize)..=(span.to as usize)] {
        h.update(rec);
    }
    let actual_digest = h.finalize();

    actual_digest != span.records_digest
}

#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use super::*;

    /// Packs `(signature_count, verify_failures)` into one `i64` so the
    /// host only needs a single typed call across the wasm boundary per
    /// timed operation (crossing it once per record would measure call
    /// overhead, not signing cost).
    #[no_mangle]
    pub extern "C" fn wasm_run_workload(strategy: i32, n: u32, batch_size: u32) -> i64 {
        let r = run_workload(Strategy::from_i32(strategy), n, batch_size);
        ((r.signature_count as i64) << 32) | (r.verify_failures as i64)
    }

    #[no_mangle]
    pub extern "C" fn wasm_tamper_check(strategy: i32, n: u32, batch_size: u32) -> i32 {
        i32::from(tamper_check(Strategy::from_i32(strategy), n, batch_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_run_has_zero_verify_failures_per_record() {
        let r = run_workload(Strategy::PerRecord, 500, 1);
        assert_eq!(r.verify_failures, 0);
        assert_eq!(r.signature_count, 500);
    }

    #[test]
    fn honest_run_has_zero_verify_failures_batch_tail() {
        for &batch in &[16u32, 64, 256] {
            let r = run_workload(Strategy::BatchTail, 500, batch);
            assert_eq!(r.verify_failures, 0, "batch_size={batch}");
            assert_eq!(
                r.signature_count,
                500u32.div_ceil(batch),
                "batch_size={batch}"
            );
        }
    }

    #[test]
    fn tamper_is_detected_per_record() {
        assert!(tamper_check(Strategy::PerRecord, 200, 1));
    }

    #[test]
    fn tamper_is_detected_batch_tail() {
        assert!(tamper_check(Strategy::BatchTail, 200, 64));
    }

    #[test]
    fn record_bytes_is_deterministic() {
        assert_eq!(record_bytes(42), record_bytes(42));
        assert_ne!(record_bytes(42), record_bytes(43));
    }
}
