//! The MANIFEST_SEG and VEC_SEG payload parsers, driven directly (the
//! whole-object target reaches them only through valid content hashes).
//!
//! Invariants: no panic; an accepted manifest's directory has at most
//! `max_entries` entries at strictly increasing offsets; an accepted VEC_SEG
//! prefix declares exactly the payload length.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_registry::validate::fuzz_api::{check_vec_head, parse_manifest};

fuzz_target!(|data: &[u8]| {
    let Some((&ctl, body)) = data.split_first() else {
        return;
    };
    if ctl & 1 == 0 {
        let max_entries = u64::from(ctl >> 1);
        if let Ok(m) = parse_manifest(body, max_entries) {
            assert!(m.directory_offsets.len() as u64 <= max_entries);
            assert!(m.directory_offsets.windows(2).all(|w| w[0] < w[1]));
            assert!(m.deleted * 8 <= body.len());
        }
    } else if body.len() >= 14 {
        let mut head = [0u8; 6];
        head.copy_from_slice(&body[..6]);
        let len = u64::from_le_bytes(body[6..14].try_into().unwrap());
        if let Ok((dim, count)) = check_vec_head(&head, len) {
            assert!(dim > 0);
            assert_eq!(
                (u64::from(dim) * 4 + 8) * u64::from(count) + 6,
                len,
                "declared length is exact"
            );
        }
    }
});
