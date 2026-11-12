//! Without the `ed25519` feature nothing can be signed or verified, so every
//! signature-requiring entry point must fail closed rather than pass.
#![cfg(not(feature = "ed25519"))]

use rvf_runtime::{RvfOptions, RvfStore};
use rvf_types::security::SecurityPolicy;

#[test]
fn signing_and_signed_open_fail_closed_without_the_feature() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("x.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    assert!(RvfStore::create_signed(&p, o.clone(), &[1; 32]).is_err());
    assert!(!p.exists());
    RvfStore::create(&p, o.clone()).unwrap().close().unwrap();
    assert!(RvfStore::open_signed(&p, o.clone(), SecurityPolicy::Strict, &[1; 32]).is_err());
    assert!(RvfStore::open_with_policy(&p, o, SecurityPolicy::Strict).is_err());
}
