#![no_main]
//! Structure-aware fuzz of `StreamIngest::feed` (project Ofer review round 2).
//! The input is a script over the frames of a real signed stream: flips,
//! drops, duplicates, swaps, truncations, and arbitrary split sizes.
//! Invariants: no panic; every visible vector equals the dataset's vector;
//! after Poisoned nothing is visible; sealed only when Complete.

use libfuzzer_sys::fuzz_target;
use rvf_runtime::progressive_stream::build::{build_stream_store, descriptor_for, BuildParams, StreamDataset};
use rvf_runtime::progressive_stream::StreamIngest;
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
        Fx {
            frames: bundle.frames_stage_order(&body).collect(),
            ds,
            public: key.verifying_key().to_bytes(),
        }
    })
}

fuzz_target!(|data: &[u8]| {
    let f = fx();
    let mut frames = f.frames.clone();
    let mut i = 0;
    while i + 3 <= data.len() && i < 3 * 24 {
        let (op, a, b) = (data[i], data[i + 1] as usize, data[i + 2] as usize);
        i += 3;
        let n = frames.len();
        let a = if a == 255 { n - 1 } else { a % n };
        match op % 7 {
            0 => {
                let l = frames[a].len();
                if l > 0 { frames[a][b % l] ^= 1 << (b % 8); }
            }
            1 if n > 1 => { frames.remove(a); }
            2 => { let c = frames[a].clone(); frames.insert(b % (n + 1), c); }
            3 => frames.swap(a, b % n),
            4 => { let l = frames[a].len(); frames[a].truncate(b % (l + 1)); }
            5 => frames[a].extend_from_slice(&data[i.saturating_sub(3)..i]),
            _ => {
                let l = frames[a].len();
                if l > 1 { let at = b % l; frames[a][at] = data[i - 3]; }
            }
        }
    }
    let keys = [f.public];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let stream: Vec<u8> = frames.concat();
    let splits = &data[i.min(data.len())..];
    let mut pos = 0usize;
    let mut s = 0usize;
    while pos < stream.len() {
        let step = 1 + splits.get(s).map_or(4096, |&x| x as usize * 37);
        s += 1;
        let end = (pos + step).min(stream.len());
        ing.feed(&stream[pos..end]);
        pos = end;
    }
    ing.end_of_input();
    for (id, v) in ing.vectors().iter() {
        assert_eq!(v, f.ds.vector(id as usize), "unauthentic visible vector {id}");
    }
    if matches!(ing.state(), StreamState::Poisoned(_) | StreamState::Rejected(_)) {
        assert!(ing.vectors().is_empty(), "vectors visible after poison");
        assert!(!ing.layers().layer_a && !ing.layers().layer_b && !ing.layers().layer_c);
    }
    let a = ing.query(&[0.0; 8], 5, 32);
    assert_eq!(a.sealed, ing.state() == StreamState::Complete);
});
