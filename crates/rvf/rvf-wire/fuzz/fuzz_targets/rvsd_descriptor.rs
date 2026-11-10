#![no_main]
//! Descriptor decoder. Path A feeds raw bytes (exercises the length/header/signature
//! gates). Path B signs the input as the unsigned prefix with the trusted test key so the
//! table invariants and size policy see arbitrary content.
#[path = "../common.rs"]
mod common;
use libfuzzer_sys::fuzz_target;
use rvf_wire::stream::decode_descriptor;

fuzz_target!(|data: &[u8]| {
    let b = common::base();
    let mut pol = common::policy(b, 0);
    pol.max_descriptor_bytes = 1 << 16;
    let _ = decode_descriptor(data, &pol);
    let mut d = data.to_vec();
    d.extend_from_slice(&64u16.to_le_bytes());
    let sig = rvf_crypto::sign_descriptor(&d, &common::signing_key());
    d.extend_from_slice(&sig);
    if let Ok(v) = decode_descriptor(&d, &pol) {
        // Whatever decodes must satisfy the layout invariants.
        let mut off = 0u64;
        for e in &v.entries {
            assert_eq!(e.body_offset, off);
            off += e.stored_len;
        }
        assert_eq!(off, v.header.body_len);
    }
});
