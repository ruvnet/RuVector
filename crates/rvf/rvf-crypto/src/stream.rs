//! RVSD (ADR-353) hashing and signing: leaf and chunk-list commitments, the
//! signer key id, and descriptor signatures.
//!
//! SHA-256 is `rvf_types::sha256` by default (no extra dependency) or the `sha2`
//! crate under the `fast-sha256` feature (about 6x faster with SHA extensions).

/// SHA-256 backend: `sha2` with runtime CPU detection under `fast-sha256`,
/// otherwise the portable `rvf_types::sha256`. Outputs are identical.
#[cfg(feature = "fast-sha256")]
mod backend {
    use sha2::Digest;

    /// Streaming SHA-256.
    #[derive(Clone, Default)]
    pub struct Sha256(sha2::Sha256);

    impl Sha256 {
        /// New hasher.
        pub fn new() -> Self {
            Self(sha2::Sha256::new())
        }
        /// Absorb bytes.
        pub fn update(&mut self, data: &[u8]) {
            self.0.update(data);
        }
        /// Finish.
        pub fn finalize(self) -> [u8; 32] {
            self.0.finalize().into()
        }
    }

    /// One-shot SHA-256.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        sha2::Sha256::digest(data).into()
    }
}
#[cfg(not(feature = "fast-sha256"))]
mod backend {
    pub use rvf_types::sha256::{sha256, Sha256};
}
pub use backend::{sha256, Sha256};

/// Domain tag of a chunk leaf.
pub const LEAF_TAG: u8 = 0x00;
/// Domain tag of a chunk table.
pub const TABLE_TAG: u8 = 0x01;

/// `SHA-256(0x00 || file_id || seg_id u64 || chunk_index u32 || chunk_len u64 || bytes)`.
///
/// `chunk_len` is `bytes.len()`. All integers little-endian.
pub fn leaf_hash(file_id: &[u8; 16], seg_id: u64, chunk_index: u32, bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&[LEAF_TAG]);
    h.update(file_id);
    h.update(&seg_id.to_le_bytes());
    h.update(&chunk_index.to_le_bytes());
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize()
}

/// Incremental form of [`leaf_hash`] for chunks read in pieces (bounded memory).
#[derive(Clone)]
pub struct LeafHasher {
    h: Sha256,
    expected: u64,
    seen: u64,
}

impl LeafHasher {
    /// Start a leaf for a chunk of exactly `chunk_len` bytes.
    pub fn new(file_id: &[u8; 16], seg_id: u64, chunk_index: u32, chunk_len: u64) -> Self {
        let mut h = Sha256::new();
        h.update(&[LEAF_TAG]);
        h.update(file_id);
        h.update(&seg_id.to_le_bytes());
        h.update(&chunk_index.to_le_bytes());
        h.update(&chunk_len.to_le_bytes());
        Self {
            h,
            expected: chunk_len,
            seen: 0,
        }
    }
    /// Absorb the next piece of the chunk.
    pub fn update(&mut self, bytes: &[u8]) {
        self.seen = self.seen.saturating_add(bytes.len() as u64);
        self.h.update(bytes);
    }
    /// Finish; `None` if fewer or more bytes than `chunk_len` were absorbed.
    pub fn finalize(self) -> Option<[u8; 32]> {
        (self.seen == self.expected).then(|| self.h.finalize())
    }
}

/// `SHA-256(0x01 || file_id || seg_id u64 || n u32 || leaf_0 || ... || leaf_{n-1})`.
pub fn chunk_list_hash(file_id: &[u8; 16], seg_id: u64, leaves: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&[TABLE_TAG]);
    h.update(file_id);
    h.update(&seg_id.to_le_bytes());
    h.update(&(leaves.len() as u32).to_le_bytes());
    for l in leaves {
        h.update(l);
    }
    h.finalize()
}

/// Same as [`chunk_list_hash`] over a flat `n * 32` byte slice.
///
/// Returns `None` if the length is not a multiple of 32.
pub fn chunk_list_hash_flat(file_id: &[u8; 16], seg_id: u64, flat: &[u8]) -> Option<[u8; 32]> {
    if flat.len() & 31 != 0 {
        return None;
    }
    let mut h = Sha256::new();
    h.update(&[TABLE_TAG]);
    h.update(file_id);
    h.update(&seg_id.to_le_bytes());
    h.update(&((flat.len() / 32) as u32).to_le_bytes());
    h.update(flat);
    Some(h.finalize())
}

/// First 8 bytes of `SHA-256(public_key)`.
pub fn signer_key_id(public_key: &[u8; 32]) -> [u8; 8] {
    let d = sha256(public_key);
    let mut id = [0u8; 8];
    id.copy_from_slice(&d[..8]);
    id
}

/// SHA-256 of a complete signed descriptor (the pin / package-contract hash).
pub fn descriptor_hash(descriptor: &[u8]) -> [u8; 32] {
    sha256(descriptor)
}

#[cfg(feature = "ed25519")]
mod sig {
    use alloc::vec::Vec;
    use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
    use rvf_types::stream::{STREAM_SIG_DOMAIN, STREAM_SIG_LEN};

    fn message(unsigned: &[u8]) -> Vec<u8> {
        let mut m = Vec::with_capacity(STREAM_SIG_DOMAIN.len() + unsigned.len());
        m.extend_from_slice(STREAM_SIG_DOMAIN);
        m.extend_from_slice(unsigned);
        m
    }

    /// Sign `"RVF-v1-stream-descriptor\0" || unsigned`.
    ///
    /// `unsigned` is every descriptor byte before the 64-byte signature,
    /// including the `sig_len` field.
    pub fn sign_descriptor(unsigned: &[u8], key: &SigningKey) -> [u8; STREAM_SIG_LEN] {
        key.sign(&message(unsigned)).to_bytes()
    }

    /// Verify a complete descriptor (`... || sig_len || signature`) with
    /// Ed25519 `verify_strict`. Returns false on any failure.
    pub fn verify_descriptor(descriptor: &[u8], public_key: &[u8; 32]) -> bool {
        if descriptor.len() < STREAM_SIG_LEN {
            return false;
        }
        let (unsigned, sig) = descriptor.split_at(descriptor.len() - STREAM_SIG_LEN);
        let mut s = [0u8; STREAM_SIG_LEN];
        s.copy_from_slice(sig);
        let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
            return false;
        };
        vk.verify_strict(&message(unsigned), &Signature::from_bytes(&s))
            .is_ok()
    }
}

#[cfg(feature = "ed25519")]
pub use sig::{sign_descriptor, verify_descriptor};

#[cfg(all(test, feature = "ed25519"))]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn leaf_binds_every_field() {
        let f = [1u8; 16];
        let base = leaf_hash(&f, 2, 3, b"abc");
        assert_ne!(base, leaf_hash(&[2u8; 16], 2, 3, b"abc"));
        assert_ne!(base, leaf_hash(&f, 3, 3, b"abc"));
        assert_ne!(base, leaf_hash(&f, 2, 4, b"abc"));
        assert_ne!(base, leaf_hash(&f, 2, 3, b"abd"));
        assert_ne!(base, leaf_hash(&f, 2, 3, b"ab"));
    }

    #[test]
    fn table_hash_flat_matches_list() {
        let f = [9u8; 16];
        let l = [[1u8; 32], [2u8; 32]];
        let flat: alloc::vec::Vec<u8> = l.iter().flatten().copied().collect();
        assert_eq!(
            chunk_list_hash(&f, 7, &l),
            chunk_list_hash_flat(&f, 7, &flat).unwrap()
        );
        assert_ne!(chunk_list_hash(&f, 7, &l), chunk_list_hash(&f, 7, &l[..1]));
        assert!(chunk_list_hash_flat(&f, 7, &flat[..33]).is_none());
    }

    #[test]
    fn sign_verify_and_tamper() {
        let sk = SigningKey::from_bytes(&[5u8; 32]);
        let pk = sk.verifying_key().to_bytes();
        let body = b"descriptor-bytes-before-signature".to_vec();
        let sig = sign_descriptor(&body, &sk);
        let mut full = body.clone();
        full.extend_from_slice(&sig);
        assert!(verify_descriptor(&full, &pk));
        let mut bad = full.clone();
        bad[3] ^= 1;
        assert!(!verify_descriptor(&bad, &pk));
        let other = SigningKey::from_bytes(&[6u8; 32])
            .verifying_key()
            .to_bytes();
        assert!(!verify_descriptor(&full, &other));
        assert!(!verify_descriptor(&full[..10], &pk));
        assert_eq!(signer_key_id(&pk), signer_key_id(&pk));
    }
}
