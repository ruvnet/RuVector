//! Deterministic content hashing for receipts and frame identity.
//!
//! Uses BLAKE3 over a canonical little-endian byte encoding. Floats are
//! hashed bit-exactly via `to_le_bytes`, so any change to a value, a
//! dimension, or ordering changes the hash — the basis of ADR-260's
//! anti-swap guarantee (§15).

/// Hash a slice of `f32` together with a domain tag, returning a hex digest.
pub fn hash_f32(tag: &str, dims: &[usize], values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(b"|");
    for d in dims {
        hasher.update(&(*d as u64).to_le_bytes());
    }
    hasher.update(b"|");
    for v in values {
        hasher.update(&v.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Hash an arbitrary byte string with a domain tag.
pub fn hash_bytes(tag: &str, bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(b"|");
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

/// Combine several hex digests into one (order-sensitive).
pub fn hash_join(tag: &str, parts: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    for p in parts {
        hasher.update(b"|");
        hasher.update(p.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}
