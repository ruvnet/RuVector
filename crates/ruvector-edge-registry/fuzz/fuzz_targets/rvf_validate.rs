//! Streaming RVF validation over arbitrary bytes (ADR-351 §3 rv-registry).
//!
//! The first byte picks the chunking and the limits; the rest is the upload.
//! Invariants: no panic; one-shot and chunked validation return the same
//! result (errors included); anything accepted is internally consistent and
//! its live vectors decode, each live VEC_SEG appears once, and
//! `total_vectors` fits the live records. Built with `--cfg fuzzing`, so
//! segment content hashes are not enforced and mutations reach the parsers.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_registry::validate::{validate, StreamValidator, ValidationLimits, VecSegView};

fuzz_target!(|data: &[u8]| {
    let Some((&ctl, body)) = data.split_first() else {
        return;
    };
    let limits = if ctl & 0x80 != 0 {
        ValidationLimits {
            max_total_bytes: 4096 + u64::from(ctl & 0x3f) * 64,
            max_segments: u32::from(ctl & 0x7) + 1,
            max_manifest_payload: 256,
            max_signature_len: 128,
            max_dimension: 64,
            allow_executable: ctl & 0x40 != 0,
            ..ValidationLimits::default()
        }
    } else {
        ValidationLimits {
            allow_executable: ctl & 0x40 != 0,
            ..ValidationLimits::default()
        }
    };
    let one = validate(body, limits);

    let step = usize::from(ctl & 0x3f) + 1;
    let mut v = StreamValidator::new(limits);
    for chunk in body.chunks(step) {
        if v.push(chunk).is_err() {
            break;
        }
    }
    let chunked = v.finish();
    assert_eq!(one, chunked);

    if let Ok(ok) = one {
        assert_eq!(ok.total_size, body.len() as u64);
        assert!(ok.dim >= 1 && ok.dim <= limits.max_dimension);
        assert!(ok.segments.len() as u32 <= limits.max_segments);
        for s in &ok.segments {
            assert!(s.offset + s.length <= ok.total_size);
            assert!(s.payload_length <= limits.max_segment_payload);
        }
        // Each live VEC_SEG at most once, and the declared total fits the
        // records those segments carry (directory amplification regression).
        assert!(ok.live_vec_segments.windows(2).all(|w| w[0] < w[1]));
        let capacity: u64 = ok
            .live_vec_segments
            .iter()
            .map(|&i| {
                let s = &ok.segments[i as usize];
                let r = s.payload_range();
                VecSegView::decode(&body[r.start as usize..r.end as usize], ok.dim)
                    .expect("live segments decode")
                    .len() as u64
            })
            .sum();
        assert!(ok.total_vectors <= capacity);
        let live = ok.live_vectors(body).expect("accepted objects decode");
        assert!(live.len() as u64 <= capacity);
        assert!(live.iter().all(|(_, x)| x.len() == usize::from(ok.dim)));
        assert!(live
            .iter()
            .all(|(id, _)| ok.deleted_ids.binary_search(id).is_err()));
    }
});
