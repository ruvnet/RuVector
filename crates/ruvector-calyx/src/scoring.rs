//! Vector scoring primitives and a deterministic non-cryptographic hash.
//!
//! Calyx's math runtime (the paper calls it *Forge*) reduces to a handful of
//! dense-vector kernels plus content addressing.  This module is the
//! dependency-free reference version of that runtime.

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity in `[-1, 1]`; returns `0.0` for a zero-length input.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_sim shape mismatch");
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// Unit-length copy of `v`, or the zero vector when `v` has no length.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n < 1e-9 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// FNV-1a 64-bit hash of a byte slice.
///
/// Used for content-addressing lenses and chaining the provenance ledger.  It
/// is **not** cryptographic — a production deployment would swap in BLAKE3 or
/// SHA-256.  FNV-1a is chosen here to keep the reference crate dependency-free
/// and fully deterministic across platforms.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Hash a sequence of `f32`s in a byte-order-stable way.
pub fn hash_vec(v: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bits().to_le_bytes());
    }
    fnv1a64(&bytes)
}

/// Fold two hashes into one (used to chain the ledger).
#[inline]
pub fn mix_hash(a: u64, b: u64) -> u64 {
    // FNV-1a over the 16 bytes of `a` then `b`.
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&a.to_le_bytes());
    bytes[8..].copy_from_slice(&b.to_le_bytes());
    fnv1a64(&bytes)
}

/// Quantize a `f32` to its bit pattern in a way that treats `-0.0 == 0.0` and
/// maps NaN to a single canonical pattern, so hashing is stable.
trait ToLeBits {
    fn to_le_bits(self) -> u32;
}
impl ToLeBits for f32 {
    #[inline]
    fn to_le_bits(self) -> u32 {
        if self == 0.0 {
            0
        } else if self.is_nan() {
            0x7fc0_0000
        } else {
            self.to_bits()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        let a = [1.0, 0.0, 0.0];
        assert!((cosine_sim(&a, &a) - 1.0).abs() < 1e-6);
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
        let z = [0.0, 0.0, 0.0];
        assert_eq!(cosine_sim(&a, &z), 0.0);
    }

    #[test]
    fn hash_is_stable_and_sensitive() {
        let v = vec![0.1, 0.2, 0.3];
        assert_eq!(hash_vec(&v), hash_vec(&v));
        let mut w = v.clone();
        w[1] = 0.20001;
        assert_ne!(hash_vec(&v), hash_vec(&w));
        // -0.0 and 0.0 hash identically.
        assert_eq!(hash_vec(&[0.0, 1.0]), hash_vec(&[-0.0, 1.0]));
    }

    #[test]
    fn mix_is_order_sensitive() {
        assert_ne!(mix_hash(1, 2), mix_hash(2, 1));
    }
}
