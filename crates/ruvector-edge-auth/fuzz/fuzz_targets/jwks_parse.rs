//! JWKS parsing (ADR-351 §5.4, M0 fuzz criterion): arbitrary bytes through
//! `JwkSet::parse_usable` and the RFC 7638 thumbprint / key conversion of
//! every entry that deserialises.
//!
//! Invariants: no panic; at most `MAX_JWKS_KEYS` usable keys; every usable
//! key round-trips through `Jwk::from_verifying_key` to the same point.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_auth::jwks::{Jwk, JwkSet, MAX_JWKS_KEYS};

fuzz_target!(|data: &[u8]| {
    if let Ok(keys) = JwkSet::parse_usable(data) {
        assert!(!keys.is_empty());
        assert!(keys.len() <= MAX_JWKS_KEYS);
        for key in keys.values() {
            let jwk = Jwk::from_verifying_key(key);
            assert_eq!(jwk.to_verifying_key().unwrap(), *key);
            assert_eq!(jwk.thumbprint().len(), 43);
        }
    }
    // Every entry that deserialises as a Jwk must be safe to thumbprint and
    // convert, usable or not.
    if let Ok(set) = serde_json::from_slice::<JwkSet>(data) {
        for jwk in set.keys.iter().take(MAX_JWKS_KEYS) {
            let _ = jwk.thumbprint();
            let _ = jwk.to_verifying_key();
        }
    }
});
