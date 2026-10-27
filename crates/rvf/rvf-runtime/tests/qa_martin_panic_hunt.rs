//! Randomized panic hunt over the stream ingest (no nightly / cargo-fuzz needed).
#[path = "support/stream_fixture.rs"]
mod fx;
use fx::*;
use rvf_runtime::progressive_stream::StreamIngest;
use rvf_wire::stream::StreamPolicy;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rng(seed: u64) -> impl FnMut() -> u64 { let mut x = seed | 1; move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x } }

fn run(keys: &[[u8; 32]], b: &[u8], piece: usize) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut ing = StreamIngest::new(StreamPolicy::new(keys));
        for c in b.chunks(piece.max(1)) { if ing.feed(c).refused.is_some() { break; } }
        ing.end_of_input();
        let _ = ing.query(&[0.0; 16], 5, 16);
        let _ = ing.query(&[], 5, 16);
        let _ = ing.query(&[f32::NAN; 16], 0, 0);
    })).map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())
}

#[test]
fn ingest_never_panics_on_mutated_or_random_input() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let desc_len = frames(&f)[0].len();
    let mut r = rng(0xC0FFEE);
    let mut panics = Vec::new();
    for i in 0..3000 {
        let mut b = match i % 4 {
            // pure garbage
            0 => (0..(r() % 20_000) as usize).map(|_| r() as u8).collect::<Vec<u8>>(),
            // many random byte overwrites anywhere
            1 => { let mut b = full.clone(); for _ in 0..(1 + r() % 64) { let at = (r() as usize) % b.len(); b[at] = r() as u8; } b }
            // descriptor-only mutations (parse paths before/around signature checks)
            2 => { let mut b = full.clone(); for _ in 0..(1 + r() % 16) { let at = (r() as usize) % desc_len; b[at] = r() as u8; } b }
            // random cut + random splice of a later region
            _ => { let mut b = full.clone(); let cut = (r() as usize) % b.len(); let from = (r() as usize) % b.len(); let tail = b[from..].to_vec(); b.truncate(cut); b.extend(tail); b }
        };
        if i % 7 == 0 { b.truncate((r() as usize) % (b.len() + 1)); }
        let piece = [1usize, 7, 64, 4096, 1 << 20][(r() % 5) as usize];
        if let Err(msg) = run(&keys, &b, piece) { panics.push((i, msg)); if panics.len() >= 5 { break; } }
    }
    assert!(panics.is_empty(), "panics: {panics:?}");
}
