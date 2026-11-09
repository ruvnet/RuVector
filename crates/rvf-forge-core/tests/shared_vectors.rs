//! ADR-354 item 6 (ruvector side): the signed Forge-authored container with a
//! 4096-byte `RVM0` root page is a shared byte fixture
//! (`crates/rvf/tests/vectors/forge_signed_rvm0.rvf`), consumed by the rvm
//! repo. Forge is byte-deterministic, so this test regenerates it and compares
//! exactly. `OFER_REGEN_VECTORS=1` rewrites the file.
//!
//! It also pins the two signers together: `rvf-runtime`'s Level-0 page
//! helpers must produce the same signing bytes as Forge and must verify a
//! Forge-signed page.

use ed25519_dalek::SigningKey;
use rvf_forge_core::author::{
    root_manifest_page_of, root_manifest_signing_bytes, verify_root_manifest_signature,
    ContainerBuilder,
};
use rvf_forge_core::{inspect_bytes, CapabilityClass, SegmentType};
use std::path::PathBuf;

const FILE_ID: [u8; 16] = *b"ofer-forge-fixt\0";

fn key() -> SigningKey {
    SigningKey::from_bytes(&[7u8; 32])
}

fn build() -> Vec<u8> {
    ContainerBuilder::new()
        .file_id(FILE_ID)
        .declare_capability(CapabilityClass::Network)
        .segment(
            SegmentType::Wasm,
            b"\0asm\x01\0\0\0ofer fixture module".to_vec(),
        )
        .segment(SegmentType::Vec, vec![0x5a; 200])
        .sign_with(key())
        .build()
        .unwrap()
        .bytes
}

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rvf/tests/vectors/forge_signed_rvm0.rvf")
}

#[test]
fn forge_signed_fixture_is_reproduced_byte_for_byte() {
    let fresh = build();
    if std::env::var_os("OFER_REGEN_VECTORS").is_some() {
        std::fs::write(path(), &fresh).unwrap();
    }
    let committed = std::fs::read(path()).unwrap();
    assert_eq!(
        committed, fresh,
        "Forge output drifted from the shared fixture"
    );
    assert_eq!(build(), fresh, "two builds are identical");

    // Shape the rvm repo relies on.
    assert_eq!(committed.len() % 64, 0);
    let page = root_manifest_page_of(&committed).expect("ends in a root page");
    assert_eq!(&page[..4], &0x5256_4D30u32.to_le_bytes(), "RVM0");
    assert!(verify_root_manifest_signature(
        page,
        &key().verifying_key().to_bytes()
    ));
    inspect_bytes(&committed).expect("inspects clean");
}

#[test]
fn runtime_and_forge_agree_on_level0_signing_bytes_and_verification() {
    let bytes = build();
    let page = root_manifest_page_of(&bytes).unwrap();
    let public = key().verifying_key().to_bytes();

    assert_eq!(
        rvf_runtime::root_page::signing_bytes(page),
        root_manifest_signing_bytes(page),
        "one definition of what the page signature covers"
    );
    assert!(rvf_runtime::root_page::verify_signature(page, &public));
    let mut other = public;
    other[0] ^= 1;
    assert!(!rvf_runtime::root_page::verify_signature(page, &other));
    let v = rvf_runtime::root_page::view(page);
    assert!(v.magic_ok && v.crc_ok);
    // The wire tail scanner locates the same manifest segment the page names.
    let (off, _) = rvf_wire::find_latest_manifest(&bytes).unwrap();
    assert_eq!(off as u64, v.l1_offset);
}
