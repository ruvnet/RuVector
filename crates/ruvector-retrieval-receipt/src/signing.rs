//! Ed25519 authentication for retrieval receipt roots.
//!
//! A signature authenticates a root under a supplied public key. It does not,
//! by itself, prove that the key belongs to a named organization or that the
//! issuer produced an honest result. Production identity requires an external
//! key registry, rotation policy, and revocation history.
//!
//! The signed statement binds the protocol version, purpose, issuer key ID,
//! deployment scope, issuance time, and root. This prevents a valid signature
//! from being replayed as a different kind of anchor or in a different scope.
//! [`BatchAnchor::verify_inclusion`] accepts only a [`VerifiedRoot`] returned by
//! [`verify_root`], which makes the required signature check hard to skip while
//! preserving one signature check per batch.

use core::fmt;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

const SIGNED_ROOT_DOMAIN: &[u8] = b"ruvector:retrieval:signed-root:v1:";
const SIGNED_ROOT_BYTES: usize = SIGNED_ROOT_DOMAIN.len() + 106;
const BATCH_LEAF_DOMAIN: &[u8] = b"ruvector:retrieval:batch:leaf:";
const BATCH_NODE_DOMAIN: &[u8] = b"ruvector:retrieval:batch:node:";

/// Current canonical signed statement format.
pub const SIGNED_ROOT_VERSION: u8 = 1;

/// The semantic use of a signed root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AnchorPurpose {
    Receipt = 1,
    Batch = 2,
}

/// Caller known verification context. `scope_hash` should identify the
/// deployment, tenant, or index state in which the anchor is valid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnchorContext {
    pub purpose: AnchorPurpose,
    pub scope_hash: [u8; 32],
}

impl AnchorContext {
    pub const fn new(purpose: AnchorPurpose, scope_hash: [u8; 32]) -> Self {
        Self {
            purpose,
            scope_hash,
        }
    }
}

/// The complete, canonically encoded statement covered by an Ed25519
/// signature. Public fields make transport serialization explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootStatement {
    pub version: u8,
    pub purpose: AnchorPurpose,
    pub issuer_key_id: [u8; 32],
    pub scope_hash: [u8; 32],
    pub issued_at_unix_ms: u64,
    pub root: [u8; 32],
}

impl RootStatement {
    fn canonical_bytes(&self) -> [u8; SIGNED_ROOT_BYTES] {
        let mut bytes = [0u8; SIGNED_ROOT_BYTES];
        let mut offset = SIGNED_ROOT_DOMAIN.len();
        bytes[..offset].copy_from_slice(SIGNED_ROOT_DOMAIN);
        bytes[offset] = self.version;
        offset += 1;
        bytes[offset] = self.purpose as u8;
        offset += 1;
        bytes[offset..offset + 32].copy_from_slice(&self.issuer_key_id);
        offset += 32;
        bytes[offset..offset + 32].copy_from_slice(&self.scope_hash);
        offset += 32;
        bytes[offset..offset + 8].copy_from_slice(&self.issued_at_unix_ms.to_be_bytes());
        offset += 8;
        bytes[offset..offset + 32].copy_from_slice(&self.root);
        bytes
    }
}

/// A signed root plus the metadata covered by the signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedRoot {
    pub statement: RootStatement,
    pub signature: [u8; 64],
}

/// Proof that [`verify_root`] authenticated a statement. Its fields are
/// private so callers cannot construct a trusted batch root without checking
/// the signature and expected context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedRoot {
    root: [u8; 32],
    context: AnchorContext,
    issued_at_unix_ms: u64,
}

impl VerifiedRoot {
    pub const fn root(&self) -> [u8; 32] {
        self.root
    }

    pub const fn context(&self) -> AnchorContext {
        self.context
    }

    pub const fn issued_at_unix_ms(&self) -> u64 {
        self.issued_at_unix_ms
    }
}

/// Recoverable errors for invalid batch construction and proof requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnchorError {
    EmptyBatch,
    IndexOutOfBounds { index: usize, len: usize },
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "batch must contain at least one receipt root"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "batch index {index} is out of bounds for length {len}")
            }
        }
    }
}

impl std::error::Error for AnchorError {}

/// An Ed25519 keypair representing a receipt signing key.
pub struct Issuer {
    signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl Issuer {
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Stable SHA256 identifier for this public key.
    pub fn key_id(&self) -> [u8; 32] {
        key_id(&self.verifying_key)
    }

    /// Sign a typed, scoped root statement.
    pub fn sign_root(
        &self,
        context: AnchorContext,
        root: [u8; 32],
        issued_at_unix_ms: u64,
    ) -> SignedRoot {
        let statement = RootStatement {
            version: SIGNED_ROOT_VERSION,
            purpose: context.purpose,
            issuer_key_id: self.key_id(),
            scope_hash: context.scope_hash,
            issued_at_unix_ms,
            root,
        };
        let signature = self
            .signing_key
            .sign(&statement.canonical_bytes())
            .to_bytes();
        SignedRoot {
            statement,
            signature,
        }
    }
}

fn key_id(vk: &VerifyingKey) -> [u8; 32] {
    Sha256::digest(vk.as_bytes()).into()
}

/// Strictly verify a signed root against the expected purpose and scope.
/// Returns a nonforgeable token on success and `None` on any mismatch.
pub fn verify_root(
    vk: &VerifyingKey,
    expected: AnchorContext,
    signed: &SignedRoot,
) -> Option<VerifiedRoot> {
    let statement = &signed.statement;
    if statement.version != SIGNED_ROOT_VERSION
        || statement.purpose != expected.purpose
        || statement.scope_hash != expected.scope_hash
        || statement.issuer_key_id != key_id(vk)
    {
        return None;
    }

    let signature = Signature::from_bytes(&signed.signature);
    vk.verify_strict(&statement.canonical_bytes(), &signature)
        .ok()
        .map(|()| VerifiedRoot {
            root: statement.root,
            context: expected,
            issued_at_unix_ms: statement.issued_at_unix_ms,
        })
}

fn batch_leaf(receipt_root: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(BATCH_LEAF_DOMAIN);
    h.update(receipt_root);
    h.finalize().into()
}

fn batch_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(BATCH_NODE_DOMAIN);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// A Merkle tree over a batch of receipt roots.
pub struct BatchAnchor {
    levels: Vec<Vec<[u8; 32]>>,
    root: [u8; 32],
}

impl BatchAnchor {
    /// Build a batch anchor without panicking on untrusted empty input.
    pub fn build(receipt_roots: &[[u8; 32]]) -> Result<Self, AnchorError> {
        if receipt_roots.is_empty() {
            return Err(AnchorError::EmptyBatch);
        }

        let leaves: Vec<[u8; 32]> = receipt_roots.iter().map(batch_leaf).collect();
        let mut levels = vec![leaves];
        while levels.last().is_some_and(|level| level.len() > 1) {
            let cur = levels
                .last()
                .expect("a nonempty batch always has a current level");
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                let right = cur.get(i + 1).unwrap_or(&cur[i]);
                next.push(batch_node(&cur[i], right));
                i += 2;
            }
            levels.push(next);
        }
        let root = levels[levels.len() - 1][0];
        Ok(Self { levels, root })
    }

    pub const fn root(&self) -> [u8; 32] {
        self.root
    }

    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Return an inclusion proof or a recoverable bounds error.
    pub fn proof_for(&self, idx: usize) -> Result<Vec<([u8; 32], bool)>, AnchorError> {
        if idx >= self.len() {
            return Err(AnchorError::IndexOutOfBounds {
                index: idx,
                len: self.len(),
            });
        }

        let mut proof = Vec::new();
        let mut i = idx;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = if i % 2 == 0 { i + 1 } else { i - 1 };
            let sibling = level.get(sibling_idx).unwrap_or(&level[i]);
            proof.push((*sibling, i % 2 == 0));
            i /= 2;
        }
        Ok(proof)
    }

    pub fn proof_bytes_for(&self, idx: usize) -> Result<usize, AnchorError> {
        Ok(32 + self.proof_for(idx)?.len() * 32)
    }

    /// Verify membership only under an already authenticated batch root.
    pub fn verify_inclusion(
        receipt_root: [u8; 32],
        proof: &[([u8; 32], bool)],
        verified_root: &VerifiedRoot,
    ) -> bool {
        if verified_root.context.purpose != AnchorPurpose::Batch {
            return false;
        }

        let mut node = batch_leaf(&receipt_root);
        for (sibling, sibling_is_right) in proof {
            node = if *sibling_is_right {
                batch_node(&node, sibling)
            } else {
                batch_node(sibling, &node)
            };
        }
        node == verified_root.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCOPE: [u8; 32] = [9u8; 32];
    const ISSUED_AT: u64 = 1_788_134_400_000;

    fn context(purpose: AnchorPurpose) -> AnchorContext {
        AnchorContext::new(purpose, SCOPE)
    }

    fn sample_roots(n: usize) -> Vec<[u8; 32]> {
        (0..n)
            .map(|i| {
                let mut root = [0u8; 32];
                root[0] = i as u8;
                root[1] = (i >> 8) as u8;
                root
            })
            .collect()
    }

    #[test]
    fn typed_root_roundtrip_succeeds() {
        let issuer = Issuer::generate();
        let signed = issuer.sign_root(context(AnchorPurpose::Receipt), [7u8; 32], ISSUED_AT);
        let verified = verify_root(
            &issuer.verifying_key,
            context(AnchorPurpose::Receipt),
            &signed,
        )
        .expect("honest signature must verify");
        assert_eq!(verified.root(), [7u8; 32]);
        assert_eq!(verified.issued_at_unix_ms(), ISSUED_AT);
    }

    #[test]
    fn every_signed_field_is_bound() {
        let issuer = Issuer::generate();
        let expected = context(AnchorPurpose::Receipt);
        let signed = issuer.sign_root(expected, [7u8; 32], ISSUED_AT);

        let mut variants = Vec::new();
        let mut value = signed;
        value.statement.version = 2;
        variants.push(value);
        let mut value = signed;
        value.statement.purpose = AnchorPurpose::Batch;
        variants.push(value);
        let mut value = signed;
        value.statement.issuer_key_id[0] ^= 0xff;
        variants.push(value);
        let mut value = signed;
        value.statement.scope_hash[0] ^= 0xff;
        variants.push(value);
        let mut value = signed;
        value.statement.issued_at_unix_ms += 1;
        variants.push(value);
        let mut value = signed;
        value.statement.root[0] ^= 0xff;
        variants.push(value);
        let mut value = signed;
        value.signature[0] ^= 0xff;
        variants.push(value);

        for tampered in variants {
            assert!(verify_root(&issuer.verifying_key, expected, &tampered).is_none());
        }
    }

    #[test]
    fn cross_purpose_and_cross_scope_replay_fail() {
        let issuer = Issuer::generate();
        let signed = issuer.sign_root(context(AnchorPurpose::Receipt), [7u8; 32], ISSUED_AT);
        assert!(verify_root(
            &issuer.verifying_key,
            context(AnchorPurpose::Batch),
            &signed
        )
        .is_none());
        assert!(verify_root(
            &issuer.verifying_key,
            AnchorContext::new(AnchorPurpose::Receipt, [8u8; 32]),
            &signed
        )
        .is_none());
    }

    #[test]
    fn wrong_issuer_key_fails_verification() {
        let issuer = Issuer::generate();
        let impostor = Issuer::generate();
        let signed = issuer.sign_root(context(AnchorPurpose::Receipt), [7u8; 32], ISSUED_AT);
        assert!(verify_root(
            &impostor.verifying_key,
            context(AnchorPurpose::Receipt),
            &signed
        )
        .is_none());
    }

    #[test]
    fn batch_anchor_verifies_all_members() {
        let issuer = Issuer::generate();
        for n in [1usize, 2, 3, 8, 17, 128] {
            let roots = sample_roots(n);
            let anchor = BatchAnchor::build(&roots).expect("nonempty batch");
            let signed = issuer.sign_root(context(AnchorPurpose::Batch), anchor.root(), ISSUED_AT);
            let verified = verify_root(
                &issuer.verifying_key,
                context(AnchorPurpose::Batch),
                &signed,
            )
            .expect("honest batch signature");
            for (i, root) in roots.iter().enumerate() {
                let proof = anchor.proof_for(i).expect("valid index");
                assert!(BatchAnchor::verify_inclusion(*root, &proof, &verified));
            }
        }
    }

    #[test]
    fn batch_input_errors_do_not_panic() {
        assert_eq!(BatchAnchor::build(&[]).err(), Some(AnchorError::EmptyBatch));
        let anchor = BatchAnchor::build(&sample_roots(2)).expect("nonempty batch");
        assert_eq!(
            anchor.proof_for(2).unwrap_err(),
            AnchorError::IndexOutOfBounds { index: 2, len: 2 }
        );
        assert_eq!(
            anchor.proof_bytes_for(usize::MAX).unwrap_err(),
            AnchorError::IndexOutOfBounds {
                index: usize::MAX,
                len: 2
            }
        );
    }

    #[test]
    fn batch_anchor_rejects_wrong_leaf_and_tampered_proof() {
        let issuer = Issuer::generate();
        let roots = sample_roots(8);
        let anchor = BatchAnchor::build(&roots).expect("nonempty batch");
        let signed = issuer.sign_root(context(AnchorPurpose::Batch), anchor.root(), ISSUED_AT);
        let verified = verify_root(
            &issuer.verifying_key,
            context(AnchorPurpose::Batch),
            &signed,
        )
        .expect("honest batch signature");
        let mut proof = anchor.proof_for(3).expect("valid index");
        let mut wrong = roots[3];
        wrong[0] ^= 0xff;
        assert!(!BatchAnchor::verify_inclusion(wrong, &proof, &verified));
        proof[0].0[0] ^= 0xff;
        assert!(!BatchAnchor::verify_inclusion(roots[3], &proof, &verified));
    }

    #[test]
    fn receipt_signature_cannot_authorize_batch_inclusion() {
        let issuer = Issuer::generate();
        let roots = sample_roots(1);
        let anchor = BatchAnchor::build(&roots).expect("nonempty batch");
        let signed = issuer.sign_root(context(AnchorPurpose::Receipt), anchor.root(), ISSUED_AT);
        let verified = verify_root(
            &issuer.verifying_key,
            context(AnchorPurpose::Receipt),
            &signed,
        )
        .expect("honest receipt signature");
        assert!(!BatchAnchor::verify_inclusion(
            roots[0],
            &anchor.proof_for(0).expect("valid index"),
            &verified
        ));
    }

    #[test]
    fn batch_anchor_proof_bytes_grow_logarithmically() {
        let small = BatchAnchor::build(&sample_roots(2)).expect("nonempty batch");
        let large = BatchAnchor::build(&sample_roots(128)).expect("nonempty batch");
        assert!(small.proof_bytes_for(0).unwrap() < large.proof_bytes_for(0).unwrap());
        assert_eq!(large.proof_bytes_for(0).unwrap(), 32 + 7 * 32);
    }
}
