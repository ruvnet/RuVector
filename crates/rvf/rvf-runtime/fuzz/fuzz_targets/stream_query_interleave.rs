#![no_main]
//! Round 4: `query()` interleaved with `feed()` over a mutated real stream.
//! Invariants after every step: every returned id is visible and its vector
//! equals the dataset's; `sealed` only in `Complete`; coverage in [0,1] and
//! non-decreasing until a terminal failure; after Poisoned/Rejected nothing is
//! visible, results are empty, sealed is false; before any vector arrived the
//! answer is empty.

use libfuzzer_sys::fuzz_target;
use rvf_index::VectorStore;
use rvf_runtime::progressive_stream::build::{build_stream_store, descriptor_for, BuildParams, StreamDataset};
use rvf_runtime::progressive_stream::{QueryMode, StreamIngest};
use rvf_wire::stream::{StreamPolicy, StreamState};
use std::sync::OnceLock;

struct Fx {
    frames: Vec<Vec<u8>>,
    ds: StreamDataset,
    public: [u8; 32],
}

fn fx() -> &'static Fx {
    static F: OnceLock<Fx> = OnceLock::new();
    F.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvf");
        let ds = StreamDataset::clustered(2600, 8, 8, 0.4, 5);
        let p = BuildParams { vec_batch: 300, chunk_log2: 12, epoch: 1, ..Default::default() };
        let built = build_stream_store(&path, &ds, &p, &[0x11; 32]).unwrap();
        let key = ed25519_dalek::SigningKey::from_bytes(&[0xA7; 32]);
        let (body, bundle) = descriptor_for(&path, built.file_id, key.clone(), &p).unwrap();
        Fx { frames: bundle.frames_stage_order(&body).collect(), ds, public: key.verifying_key().to_bytes() }
    })
}

fuzz_target!(|data: &[u8]| {
    let f = fx();
    let mut frames = f.frames.clone();
    let mut i = 0;
    // mutation script (first bytes), then query/feed schedule
    while i + 3 <= data.len() && i < 3 * 6 {
        let (op, a, b) = (data[i], data[i + 1] as usize, data[i + 2] as usize);
        i += 3;
        let n = frames.len();
        let a = a % n;
        match op % 6 {
            0 => { let l = frames[a].len(); if l > 0 { frames[a][b % l] ^= 1 << (b % 8); } }
            1 if n > 1 => { frames.remove(a); }
            2 => { let c = frames[a].clone(); frames.insert(b % (n + 1), c); }
            3 => frames.swap(a, b % n),
            4 => { let l = frames[a].len(); frames[a].truncate(b % (l + 1)); }
            _ => {}
        }
    }
    let keys = [f.public];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let stream: Vec<u8> = frames.concat();
    let sched = &data[i.min(data.len())..];
    let mut pos = 0usize;
    let mut s = 0usize;
    let mut last_cov = 0.0f64;
    let mut terminal = false;
    let check = |ing: &StreamIngest<'_>, mode: QueryMode, qsel: u8, k: usize, last_cov: &mut f64, terminal: &mut bool| {
        let q: Vec<f32> = if qsel % 5 == 0 { vec![0.0; 3] } else { (0..8).map(|j| (qsel as f32) * 0.01 + j as f32 * 0.1).collect() };
        let a = ing.query_with(&q, k, 1 + (qsel as usize % 64), mode);
        let st = ing.state();
        assert_eq!(a.sealed && st != StreamState::Complete, false, "sealed outside Complete");
        assert!((0.0..=1.0).contains(&a.coverage_vectors), "coverage out of range");
        assert!((0.0..=1.0).contains(&a.coverage_bytes));
        if matches!(st, StreamState::Poisoned(_) | StreamState::Rejected(_)) {
            *terminal = true;
            assert!(a.envelope.results.is_empty(), "results after terminal failure");
            assert_eq!(a.visible_vectors, 0, "vectors visible after terminal failure");
            assert!(!a.sealed);
        } else if !*terminal {
            assert!(a.coverage_vectors + 1e-9 >= *last_cov, "coverage went backwards");
            *last_cov = a.coverage_vectors;
        }
        for r in &a.envelope.results {
            let v = ing.vectors().get_vector(r.id).expect("returned id is not visible");
            assert_eq!(v, f.ds.vector(r.id as usize), "returned id with unauthentic vector");
        }
        if a.visible_vectors == 0 {
            assert!(a.envelope.results.is_empty());
        }
        assert!(a.envelope.results.len() <= k);
    };
    while pos < stream.len() {
        let step = 1 + sched.get(s).map_or(2048, |&x| x as usize * 41);
        s += 1;
        let end = (pos + step).min(stream.len());
        ing.feed(&stream[pos..end]);
        pos = end;
        let sel = sched.get(s).copied().unwrap_or(0);
        s += 1;
        let mode = match sel % 3 { 0 => QueryMode::Auto, 1 => QueryMode::LayersOnly, _ => QueryMode::ScanOnly };
        check(&ing, mode, sel, 1 + (sel as usize % 20), &mut last_cov, &mut terminal);
    }
    ing.end_of_input();
    check(&ing, QueryMode::Auto, 7, 5, &mut last_cov, &mut terminal);
});
