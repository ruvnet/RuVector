//! Integration with ruvector-verified ProofEnvironment.
//!
//! This module provides a bridge between the ruvix-proof engine
//! and the formal verification layer in ruvector-verified.

#[cfg(feature = "verified")]
mod verified_integration {
    //! Integration with ruvector-verified's ProofEnvironment.
    //!
    //! When the `verified` feature is enabled, this module provides
    //! integration with formal verification.

    // This would be enabled when ruvector-verified is added as a dependency
    // For now, we define the interface without the actual integration

    /// A bridge between ruvix-proof and ruvector-verified.
    pub struct VerifiedProofBridge {
        /// Attestation counter.
        attestation_count: u64,
    }

    impl VerifiedProofBridge {
        /// Creates a new verified proof bridge.
        #[must_use]
        pub const fn new() -> Self {
            Self {
                attestation_count: 0,
            }
        }

        /// Returns the number of attestations created.
        #[must_use]
        pub const fn attestation_count(&self) -> u64 {
            self.attestation_count
        }
    }

    impl Default for VerifiedProofBridge {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "verified")]
pub use verified_integration::VerifiedProofBridge;

/// Trait for objects that can be formally verified.
///
/// This trait is implemented by proof tokens and attestations
/// to enable integration with the formal verification system.
pub trait FormallyVerifiable {
    /// Returns the verification tag for this object.
    fn verification_tag(&self) -> [u8; 8];

    /// Returns the proof depth (reduction steps expected).
    fn proof_depth(&self) -> u32;
}

impl FormallyVerifiable for crate::ProofToken {
    fn verification_tag(&self) -> [u8; 8] {
        let mut tag = [0u8; 8];
        tag[0..4].copy_from_slice(&self.mutation_hash[0..4]);
        tag[4..8].copy_from_slice(&self.nonce.to_le_bytes()[0..4]);
        tag
    }

    fn proof_depth(&self) -> u32 {
        match self.tier {
            crate::ProofTier::Reflex => 1,
            crate::ProofTier::Standard => 32,
            crate::ProofTier::Deep => 256,
        }
    }
}

impl FormallyVerifiable for crate::ProofAttestation {
    fn verification_tag(&self) -> [u8; 8] {
        let mut tag = [0u8; 8];
        tag.copy_from_slice(&self.proof_term_hash[0..8]);
        tag
    }

    fn proof_depth(&self) -> u32 {
        self.reduction_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProofAttestation, ProofPayload, ProofTier, ProofToken};

    #[test]
    fn test_proof_token_verification_tag() {
        let token = ProofToken::new(
            [0xABu8; 32],
            ProofTier::Standard,
            ProofPayload::Hash { hash: [0u8; 32] },
            1000,
            0x1234567890ABCDEF,
        );

        let tag = token.verification_tag();
        assert_eq!(tag[0..4], [0xAB, 0xAB, 0xAB, 0xAB]);
    }

    #[test]
    fn test_proof_token_depth() {
        let reflex = ProofToken::new(
            [0u8; 32],
            ProofTier::Reflex,
            ProofPayload::Hash { hash: [0u8; 32] },
            1000,
            1,
        );
        assert_eq!(reflex.proof_depth(), 1);

        let standard = ProofToken::new(
            [0u8; 32],
            ProofTier::Standard,
            ProofPayload::Hash { hash: [0u8; 32] },
            1000,
            1,
        );
        assert_eq!(standard.proof_depth(), 32);

        let deep = ProofToken::new(
            [0u8; 32],
            ProofTier::Deep,
            ProofPayload::Hash { hash: [0u8; 32] },
            1000,
            1,
        );
        assert_eq!(deep.proof_depth(), 256);
    }

    #[test]
    fn test_attestation_verification_tag() {
        let attestation =
            ProofAttestation::new([0xCDu8; 32], [0u8; 32], 1000, 0x00_01_00_00, 100, 5000);

        let tag = attestation.verification_tag();
        assert_eq!(tag, [0xCD; 8]);
    }

    #[test]
    fn test_attestation_depth() {
        let attestation = ProofAttestation::new([0u8; 32], [0u8; 32], 1000, 0, 42, 0);

        assert_eq!(attestation.proof_depth(), 42);
    }
}
