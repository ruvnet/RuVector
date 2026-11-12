//! rv-quant persist v2 (`rbqx0002`) decoding (ADR-351 G4c, M4): the
//! QuantShard Durable Object cold-loads a shard from stored frames with
//! `persist::load_frames` (one storage row per frame) or `load_from` (a
//! concatenated stream).
//!
//! Input: `mode ‖ stream`, where `stream` is a concatenated snapshot
//! (header frame, data frames, footer). Mode bits:
//!
//! * `1` — reseal: recompute the header CRC, every data-frame CRC (walking
//!   the declared lengths, clamped to the input) and the footer CRC, so
//!   the structural checks behind them see arbitrary values;
//! * `2` — also rewrite the rotation fingerprint for the declared
//!   dimension / kind / seed (only when the rotation build fits the
//!   budget), so decoding reaches the content checks and a shard;
//! * `4` — also feed the frames to `load_frames` as separate rows.
//!
//! A small load budget keeps each execution fast (the production default
//! allows a 62M-unit Haar rebuild). Invariants: no panic; `load_from` and
//! `load_frames` agree; a loaded shard answers queries, re-saves, and the
//! re-saved bytes load back to the same keys, norms and codes.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_quant::budget::{rotation_build_units, MAX_TOP_K};
use ruvector_edge_quant::persist::format::{
    crc32, le_u32, rotation_fingerprint, Crc32, FOOTER_LEN, FRAME_PREFIX_LEN, HEADER_LEN,
};
use ruvector_edge_quant::persist::{load_frames, load_from, save_frames, save_to, Header};
use ruvector_edge_quant::{Budget, QuantConfig, QuantShard, QueryOptions};

fn budget() -> Budget {
    Budget {
        max_resident_bytes: 4 << 20,
        max_vectors: 20_000,
        max_query_units: 2_000_000,
        max_load_units: 4_000_000,
        max_rerank_candidates: 1_000,
        max_top_k: MAX_TOP_K,
    }
}

/// Split a stream into `[header, data…, footer]` frames by the declared
/// lengths (clamped to the input). Returns `None` if there is no header.
fn frames(s: &[u8]) -> Option<Vec<std::ops::Range<usize>>> {
    if s.len() < HEADER_LEN {
        return None;
    }
    let declared = le_u32(&s[48..52]) as usize;
    let mut out = vec![0..HEADER_LEN];
    let mut pos = HEADER_LEN;
    for _ in 0..declared {
        if s.len() - pos < FRAME_PREFIX_LEN {
            break;
        }
        let len = (le_u32(&s[pos + 8..pos + 12]) as usize).min(s.len() - pos - FRAME_PREFIX_LEN);
        out.push(pos..pos + FRAME_PREFIX_LEN + len);
        pos += FRAME_PREFIX_LEN + len;
    }
    out.push(pos..s.len());
    Some(out)
}

fn fix_fingerprint(s: &mut [u8]) {
    let Ok(h) = Header::decode(&s[..HEADER_LEN]) else {
        return;
    };
    let dim = h.dim as usize;
    if dim == 0 || dim > 1536 || rotation_build_units(dim, h.rotation) > budget().max_load_units {
        return;
    }
    let cfg = QuantConfig {
        dim,
        metric: h.metric,
        rotation: h.rotation,
        seed: h.seed,
        budget: budget(),
    };
    if let Ok(shard) = QuantShard::new(cfg) {
        s[40..48].copy_from_slice(&rotation_fingerprint(shard.rotation()).to_le_bytes());
        let crc = crc32(&s[0..60]);
        s[60..64].copy_from_slice(&crc.to_le_bytes());
    }
}

fn reseal(s: &mut [u8], fingerprint: bool) {
    if s.len() < HEADER_LEN {
        return;
    }
    let crc = crc32(&s[0..60]);
    s[60..64].copy_from_slice(&crc.to_le_bytes());
    if fingerprint {
        fix_fingerprint(s);
    }
    let Some(ranges) = frames(s) else {
        return;
    };
    let mut all = Crc32::new();
    for r in &ranges[1..ranges.len() - 1] {
        let f = &mut s[r.clone()];
        let mut c = Crc32::new();
        c.update(&f[0..12]);
        c.update(&f[FRAME_PREFIX_LEN..]);
        let crc = c.finish();
        f[12..16].copy_from_slice(&crc.to_le_bytes());
        all.update(&crc.to_le_bytes());
    }
    let footer = ranges.last().unwrap().clone();
    if footer.len() >= FOOTER_LEN {
        let at = footer.start + 12;
        s[at..at + 4].copy_from_slice(&all.finish().to_le_bytes());
    }
}

fn exercise(shard: &QuantShard) {
    let dim = shard.dim();
    let q: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    let opts = QueryOptions {
        top_k: 5,
        rerank_factor: 1,
    };
    if let Ok(out) = shard.query(&q, &opts, None) {
        assert!(out.hits.len() <= 5.min(shard.len()));
        for h in &out.hits {
            assert!(shard.contains(h.key));
        }
    }
    let frames = save_frames(shard, 4096).expect("a loaded shard re-saves");
    let back =
        load_frames(frames.iter().map(|f| f.as_slice()), budget()).expect("a re-saved shard loads");
    assert_eq!(back.keys(), shard.keys());
    assert_eq!(back.packed(), shard.packed());
    let (a, b): (Vec<u32>, Vec<u32>) = (
        back.norms().iter().map(|x| x.to_bits()).collect(),
        shard.norms().iter().map(|x| x.to_bits()).collect(),
    );
    assert_eq!(a, b);
    let mut stream = Vec::new();
    save_to(shard, &mut stream, 4096).expect("stream save");
    let again = load_from(&mut stream.as_slice(), budget()).expect("stream reload");
    assert_eq!(again.keys(), shard.keys());
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let mut s = rest.to_vec();
    if mode & 1 != 0 {
        reseal(&mut s, mode & 2 != 0);
    }
    let streamed = load_from(&mut s.as_slice(), budget());
    if mode & 4 != 0 {
        if let Some(ranges) = frames(&s) {
            let rows = ranges.iter().map(|r| &s[r.clone()]);
            let framed = load_frames(rows, budget());
            // The frame split follows the declared lengths, so both readers
            // see the same frames.
            assert_eq!(
                streamed.is_ok(),
                framed.is_ok(),
                "stream {:?} vs frames {:?}",
                streamed.as_ref().err(),
                framed.as_ref().err()
            );
        }
    }
    if let Ok(shard) = streamed {
        exercise(&shard);
    }
});
