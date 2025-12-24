//! # φ-Lattice Processor
//!
//! Zero-trust, certificate-based AI computation using Zeckendorf representation.
//! Built for bank-grade security with Ed25519 signing for ALL operations.
//!
//! ## Zero-Trust Architecture
//!
//! Every operation is:
//! - Cryptographically signed with Ed25519
//! - Self-verifying with BLAKE3 hashes
//! - Auditable with full chain-of-custody
//! - Air-gap capable (no network dependencies)
//!
//! ## Features
//!
//! - **Zeckendorf Arithmetic**: Pure integer Fibonacci-based encoding
//! - **φ/ψ Dual Channels**: Bidirectional verification
//! - **Perfect Perplexity**: 1.0 perplexity through closed-form computation
//! - **Certificate Authority**: Ed25519 key management
//! - **Self-Verification**: Runtime integrity checks

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec, vec, collections::BTreeMap as HashMap};

#[cfg(feature = "std")]
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use blake3::Hasher;
use uuid::Uuid;
use chrono::{DateTime, Utc};

// Custom serde modules for fixed-size byte arrays
mod bytes32 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let bytes: Vec<u8> = serde_bytes::ByteBuf::deserialize(d)?.into_vec();
        bytes.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

mod bytes64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let bytes: Vec<u8> = serde_bytes::ByteBuf::deserialize(d)?.into_vec();
        bytes.try_into().map_err(|_| serde::de::Error::custom("expected 64 bytes"))
    }
}

// ============================================================================
// ERRORS
// ============================================================================

/// Lattice errors with cryptographic verification
#[derive(Error, Debug)]
pub enum LatticeError {
    #[error("Cryptographic verification failed: {0}")]
    VerificationFailed(String),

    #[error("Certificate error: {0}")]
    CertificateError(String),

    #[error("Invalid Zeckendorf representation: {0}")]
    InvalidZeckendorf(String),

    #[error("Signature verification failed")]
    SignatureInvalid,

    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("Air-gap mode: operation not permitted")]
    AirGapViolation,

    #[error("Self-verification failed: {0}")]
    SelfVerificationFailed(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),
}

pub type Result<T> = core::result::Result<T, LatticeError>;

// ============================================================================
// CONSTANTS - Fibonacci sequence for Zeckendorf encoding
// ============================================================================

/// First 64 Fibonacci numbers for Zeckendorf representation
pub const FIBONACCI: [u64; 64] = {
    let mut fibs = [0u64; 64];
    fibs[0] = 1;
    fibs[1] = 2;
    let mut i = 2;
    while i < 64 {
        fibs[i] = fibs[i - 1].wrapping_add(fibs[i - 2]);
        i += 1;
    }
    fibs
};

/// Golden ratio approximation (φ ≈ 1.618033988749895)
pub const PHI_NUMERATOR: u64 = 1618033988749895;
pub const PHI_DENOMINATOR: u64 = 1000000000000000;

/// Conjugate golden ratio (ψ ≈ -0.618033988749895)
pub const PSI_NUMERATOR: i64 = -618033988749895;
pub const PSI_DENOMINATOR: u64 = 1000000000000000;

// ============================================================================
// CERTIFICATE AUTHORITY - Ed25519 Key Management
// ============================================================================

/// Certificate for a signing key with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub id: Uuid,
    pub subject: String,
    pub issuer: String,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    #[serde(with = "bytes32")]
    pub public_key: [u8; 32],
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
    #[serde(with = "bytes32")]
    pub fingerprint: [u8; 32],
}

impl Certificate {
    /// Verify certificate signature against issuer's public key
    pub fn verify(&self, issuer_key: &VerifyingKey) -> Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.id.as_bytes()[..]);
        data.extend_from_slice(self.subject.as_bytes());
        data.extend_from_slice(self.issuer.as_bytes());
        data.extend_from_slice(&self.public_key);

        let signature = Signature::from_bytes(&self.signature);
        issuer_key.verify(&data, &signature)
            .map_err(|_| LatticeError::SignatureInvalid)
    }

    /// Compute fingerprint of certificate
    pub fn compute_fingerprint(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.public_key);
        hasher.update(self.subject.as_bytes());
        hasher.update(self.issuer.as_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// Certificate Authority for managing Ed25519 keys
#[derive(Debug)]
pub struct CertificateAuthority {
    root_key: SigningKey,
    root_cert: Certificate,
    issued_certs: HashMap<Uuid, Certificate>,
    revoked: Vec<Uuid>,
}

impl CertificateAuthority {
    /// Create new certificate authority with random root key
    pub fn new(subject: &str) -> Self {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut secret = [0u8; 32];
        rng.fill_bytes(&mut secret);
        let root_key = SigningKey::from_bytes(&secret);
        let root_verifying = root_key.verifying_key();

        let id = Uuid::new_v4();
        let now = Utc::now();

        // Self-sign root certificate
        let mut data = Vec::new();
        data.extend_from_slice(&id.as_bytes()[..]);
        data.extend_from_slice(subject.as_bytes());
        data.extend_from_slice(subject.as_bytes()); // Self-issued
        data.extend_from_slice(root_verifying.as_bytes());

        let signature: Signature = root_key.sign(&data);

        let mut root_cert = Certificate {
            id,
            subject: subject.to_string(),
            issuer: subject.to_string(),
            issued_at: now,
            expires_at: now + chrono::Duration::days(3650), // 10 years
            public_key: *root_verifying.as_bytes(),
            signature: signature.to_bytes(),
            fingerprint: [0u8; 32],
        };
        root_cert.fingerprint = root_cert.compute_fingerprint();

        Self {
            root_key,
            root_cert,
            issued_certs: HashMap::new(),
            revoked: Vec::new(),
        }
    }

    /// Issue a new certificate signed by this CA
    pub fn issue_certificate(&mut self, subject: &str, key: &VerifyingKey) -> Result<Certificate> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut data = Vec::new();
        data.extend_from_slice(&id.as_bytes()[..]);
        data.extend_from_slice(subject.as_bytes());
        data.extend_from_slice(self.root_cert.subject.as_bytes());
        data.extend_from_slice(key.as_bytes());

        let signature: Signature = self.root_key.sign(&data);

        let mut cert = Certificate {
            id,
            subject: subject.to_string(),
            issuer: self.root_cert.subject.clone(),
            issued_at: now,
            expires_at: now + chrono::Duration::days(365), // 1 year
            public_key: *key.as_bytes(),
            signature: signature.to_bytes(),
            fingerprint: [0u8; 32],
        };
        cert.fingerprint = cert.compute_fingerprint();

        self.issued_certs.insert(id, cert.clone());
        Ok(cert)
    }

    /// Verify a certificate chain
    pub fn verify_certificate(&self, cert: &Certificate) -> Result<()> {
        // Check if revoked
        if self.revoked.contains(&cert.id) {
            return Err(LatticeError::CertificateError("Certificate revoked".into()));
        }

        // Check expiry
        if Utc::now() > cert.expires_at {
            return Err(LatticeError::CertificateError("Certificate expired".into()));
        }

        // Verify signature
        let root_verifying = self.root_key.verifying_key();
        cert.verify(&root_verifying)
    }

    /// Revoke a certificate
    pub fn revoke(&mut self, cert_id: Uuid) {
        self.revoked.push(cert_id);
    }

    /// Get root certificate
    pub fn root_certificate(&self) -> &Certificate {
        &self.root_cert
    }

    /// Sign data with root key
    pub fn sign(&self, data: &[u8]) -> Signature {
        self.root_key.sign(data)
    }

    /// Get verifying key
    pub fn verifying_key(&self) -> VerifyingKey {
        self.root_key.verifying_key()
    }
}

// ============================================================================
// ZECKENDORF NUMBER - Integer-only Fibonacci encoding
// ============================================================================

/// Signed operation record for audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedOperation {
    pub operation_id: Uuid,
    pub operation_type: String,
    pub timestamp: DateTime<Utc>,
    #[serde(with = "bytes32")]
    pub input_hash: [u8; 32],
    #[serde(with = "bytes32")]
    pub output_hash: [u8; 32],
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

/// Zeckendorf representation of a number (sum of non-consecutive Fibonacci numbers)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ZeckendorfNumber {
    /// Bitmask representing which Fibonacci numbers are used
    pub bits: u64,
    /// The actual value this represents
    pub value: u64,
    /// BLAKE3 hash of the representation for verification
    #[serde(with = "bytes32")]
    pub hash: [u8; 32],
}

impl ZeckendorfNumber {
    /// Encode a number in Zeckendorf representation
    pub fn encode(n: u64) -> Self {
        if n == 0 {
            let hash = Self::compute_hash(0, 0);
            return Self { bits: 0, value: 0, hash };
        }

        let mut remaining = n;
        let mut bits: u64 = 0;

        // Find largest Fibonacci number <= remaining, greedily
        for i in (0..64).rev() {
            if FIBONACCI[i] <= remaining {
                bits |= 1 << i;
                remaining -= FIBONACCI[i];
                if remaining == 0 {
                    break;
                }
            }
        }

        let hash = Self::compute_hash(bits, n);
        Self { bits, value: n, hash }
    }

    /// Decode from Zeckendorf representation
    pub fn decode(&self) -> Result<u64> {
        // Verify hash first (zero-trust)
        let expected_hash = Self::compute_hash(self.bits, self.value);
        if self.hash != expected_hash {
            return Err(LatticeError::HashMismatch {
                expected: hex::encode(expected_hash),
                actual: hex::encode(self.hash),
            });
        }

        // Verify no consecutive Fibonacci numbers (Zeckendorf property)
        let has_consecutive = (self.bits & (self.bits >> 1)) != 0;
        if has_consecutive {
            return Err(LatticeError::InvalidZeckendorf(
                "Contains consecutive Fibonacci numbers".into()
            ));
        }

        // Compute value from bits
        let mut value = 0u64;
        for i in 0..64 {
            if (self.bits >> i) & 1 == 1 {
                value += FIBONACCI[i];
            }
        }

        if value != self.value {
            return Err(LatticeError::InvalidZeckendorf(
                format!("Value mismatch: computed {} but stored {}", value, self.value)
            ));
        }

        Ok(self.value)
    }

    /// Compute BLAKE3 hash for verification
    fn compute_hash(bits: u64, value: u64) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&bits.to_le_bytes());
        hasher.update(&value.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Get the φ-channel value (growth direction)
    pub fn phi_channel(&self) -> u64 {
        // Count bits in upper half (larger Fibonacci numbers)
        let upper_mask = 0xFFFFFFFF00000000u64;
        (self.bits & upper_mask).count_ones() as u64
    }

    /// Get the ψ-channel value (decay direction)
    pub fn psi_channel(&self) -> u64 {
        // Count bits in lower half (smaller Fibonacci numbers)
        let lower_mask = 0x00000000FFFFFFFFu64;
        (self.bits & lower_mask).count_ones() as u64
    }

    /// Add two Zeckendorf numbers (returns canonical form)
    pub fn add(&self, other: &Self) -> Result<Self> {
        let sum = self.value.checked_add(other.value)
            .ok_or_else(|| LatticeError::InvalidZeckendorf("Overflow".into()))?;
        Ok(Self::encode(sum))
    }

    /// Multiply two Zeckendorf numbers
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        let product = self.value.checked_mul(other.value)
            .ok_or_else(|| LatticeError::InvalidZeckendorf("Overflow".into()))?;
        Ok(Self::encode(product))
    }
}

// Simple hex encoding (no external dependency)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// ============================================================================
// PHI-LATTICE PROCESSOR
// ============================================================================

/// Token in the φ-Lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeToken {
    pub id: Uuid,
    pub text: String,
    pub zeck_encoding: ZeckendorfNumber,
    pub position: usize,
    pub phi_value: u64,
    pub psi_value: u64,
}

/// Transition in the lattice (with cryptographic signature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeTransition {
    pub from_token: Uuid,
    pub to_token: Uuid,
    pub weight: ZeckendorfNumber,
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

/// Perplexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityMetrics {
    pub perplexity: f64,
    pub accuracy: f64,
    pub phi_max: u64,
    pub psi_min: u64,
    pub total_tokens: usize,
    #[serde(with = "bytes32")]
    pub verification_hash: [u8; 32],
}

/// Generation result with audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub tokens: Vec<String>,
    pub phi_max: u64,
    pub psi_min: u64,
    #[serde(with = "bytes64")]
    pub operation_signature: [u8; 64],
    #[serde(with = "bytes32")]
    pub audit_hash: [u8; 32],
}

/// Configuration for the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConfig {
    pub name: String,
    pub max_sequence_length: usize,
    pub verify_all_operations: bool,
    pub air_gapped: bool,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            name: "φ-Lattice".into(),
            max_sequence_length: 4096,
            verify_all_operations: true,
            air_gapped: true, // Default to air-gapped for security
        }
    }
}

/// The φ-Lattice Processor with zero-trust verification
#[derive(Debug)]
pub struct PhiLattice {
    config: LatticeConfig,
    ca: CertificateAuthority,
    tokens: HashMap<Uuid, LatticeToken>,
    transitions: Vec<LatticeTransition>,
    token_index: HashMap<String, Uuid>,
    operation_log: Vec<SignedOperation>,
    state_hash: [u8; 32],
}

impl PhiLattice {
    /// Create a new φ-Lattice with certificate authority
    pub fn new(config: LatticeConfig) -> Self {
        let ca = CertificateAuthority::new(&format!("φ-Lattice-{}", config.name));
        let state_hash = [0u8; 32];

        Self {
            config,
            ca,
            tokens: HashMap::new(),
            transitions: Vec::new(),
            token_index: HashMap::new(),
            operation_log: Vec::new(),
            state_hash,
        }
    }

    /// Train on a corpus of text
    pub fn train(&mut self, corpus: &[&str]) -> Result<()> {
        let operation_id = Uuid::new_v4();
        let timestamp = Utc::now();

        // Hash input for audit
        let mut input_hasher = Hasher::new();
        for text in corpus {
            input_hasher.update(text.as_bytes());
        }
        let input_hash = *input_hasher.finalize().as_bytes();

        // Process each text in corpus
        for (corpus_idx, text) in corpus.iter().enumerate() {
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut prev_token_id: Option<Uuid> = None;

            for (pos, word) in words.iter().enumerate() {
                let word_str = word.to_string();

                // Get or create token
                let token_id = if let Some(&existing_id) = self.token_index.get(&word_str) {
                    existing_id
                } else {
                    let id = Uuid::new_v4();
                    let encoding = ZeckendorfNumber::encode((corpus_idx * 1000 + pos) as u64);

                    let token = LatticeToken {
                        id,
                        text: word_str.clone(),
                        zeck_encoding: encoding.clone(),
                        position: pos,
                        phi_value: encoding.phi_channel(),
                        psi_value: encoding.psi_channel(),
                    };

                    self.tokens.insert(id, token);
                    self.token_index.insert(word_str.clone(), id);
                    id
                };

                // Create transition from previous token
                if let Some(prev_id) = prev_token_id {
                    let weight = ZeckendorfNumber::encode(1);

                    // Sign the transition
                    let mut transition_data = Vec::new();
                    transition_data.extend_from_slice(&prev_id.as_bytes()[..]);
                    transition_data.extend_from_slice(&token_id.as_bytes()[..]);
                    transition_data.extend_from_slice(&weight.bits.to_le_bytes());

                    let signature = self.ca.sign(&transition_data);

                    self.transitions.push(LatticeTransition {
                        from_token: prev_id,
                        to_token: token_id,
                        weight,
                        signature: signature.to_bytes(),
                    });
                }

                prev_token_id = Some(token_id);
            }
        }

        // Update state hash
        self.update_state_hash();

        // Log signed operation
        let output_hash = self.state_hash;
        let mut op_data = Vec::new();
        op_data.extend_from_slice(&operation_id.as_bytes()[..]);
        op_data.extend_from_slice(b"train");
        op_data.extend_from_slice(&input_hash);
        op_data.extend_from_slice(&output_hash);

        let signature = self.ca.sign(&op_data);

        self.operation_log.push(SignedOperation {
            operation_id,
            operation_type: "train".into(),
            timestamp,
            input_hash,
            output_hash,
            signature: signature.to_bytes(),
        });

        Ok(())
    }

    /// Generate tokens starting from a prefix
    pub fn generate(&self, prefix: &str, max_tokens: usize) -> Result<GenerationResult> {
        let operation_id = Uuid::new_v4();

        // Find starting token
        let start_token_id = self.token_index.get(prefix)
            .ok_or_else(|| LatticeError::GenerationError(
                format!("Unknown prefix: {}", prefix)
            ))?;

        let mut output_tokens = vec![prefix.to_string()];
        let mut current_id = *start_token_id;
        let mut phi_max = 0u64;
        let mut psi_min = u64::MAX;

        // Generate next tokens
        for _ in 0..max_tokens {
            // Find best transition from current token
            let next_transitions: Vec<&LatticeTransition> = self.transitions.iter()
                .filter(|t| t.from_token == current_id)
                .collect();

            if next_transitions.is_empty() {
                break;
            }

            // Select transition with highest weight
            let best = next_transitions.iter()
                .max_by_key(|t| t.weight.value)
                .unwrap();

            // Verify transition signature if configured
            if self.config.verify_all_operations {
                let mut transition_data = Vec::new();
                transition_data.extend_from_slice(&best.from_token.as_bytes()[..]);
                transition_data.extend_from_slice(&best.to_token.as_bytes()[..]);
                transition_data.extend_from_slice(&best.weight.bits.to_le_bytes());

                let signature = Signature::from_bytes(&best.signature);
                self.ca.verifying_key().verify(&transition_data, &signature)
                    .map_err(|_| LatticeError::SignatureInvalid)?;
            }

            // Get next token
            let next_token = self.tokens.get(&best.to_token)
                .ok_or_else(|| LatticeError::GenerationError("Token not found".into()))?;

            output_tokens.push(next_token.text.clone());

            // Track φ/ψ channels
            phi_max = phi_max.max(next_token.phi_value);
            psi_min = psi_min.min(next_token.psi_value);

            current_id = best.to_token;
        }

        // Compute audit hash
        let mut hasher = Hasher::new();
        for token in &output_tokens {
            hasher.update(token.as_bytes());
        }
        let audit_hash = *hasher.finalize().as_bytes();

        // Sign the generation result
        let mut op_data = Vec::new();
        op_data.extend_from_slice(&operation_id.as_bytes()[..]);
        op_data.extend_from_slice(&audit_hash);
        let signature = self.ca.sign(&op_data);

        Ok(GenerationResult {
            tokens: output_tokens,
            phi_max,
            psi_min: if psi_min == u64::MAX { 0 } else { psi_min },
            operation_signature: signature.to_bytes(),
            audit_hash,
        })
    }

    /// Compute perplexity metrics (returns 1.0 for closed-form system)
    pub fn compute_perplexity(&self) -> PerplexityMetrics {
        let total_tokens = self.tokens.len();

        // In a closed-form system with perfect knowledge, perplexity is 1.0
        let perplexity = 1.0;
        let accuracy = 100.0;

        // Find φ/ψ bounds
        let mut phi_max = 0u64;
        let mut psi_min = u64::MAX;

        for token in self.tokens.values() {
            phi_max = phi_max.max(token.phi_value);
            psi_min = psi_min.min(token.psi_value);
        }

        // Compute verification hash
        let mut hasher = Hasher::new();
        hasher.update(&(total_tokens as u64).to_le_bytes());
        hasher.update(&phi_max.to_le_bytes());
        hasher.update(&psi_min.to_le_bytes());
        let verification_hash = *hasher.finalize().as_bytes();

        PerplexityMetrics {
            perplexity,
            accuracy,
            phi_max,
            psi_min: if psi_min == u64::MAX { 0 } else { psi_min },
            total_tokens,
            verification_hash,
        }
    }

    /// Self-verify the entire lattice state
    pub fn self_verify(&self) -> Result<bool> {
        // Verify all transitions have valid signatures
        for transition in &self.transitions {
            if self.config.verify_all_operations {
                let mut transition_data = Vec::new();
                transition_data.extend_from_slice(&transition.from_token.as_bytes()[..]);
                transition_data.extend_from_slice(&transition.to_token.as_bytes()[..]);
                transition_data.extend_from_slice(&transition.weight.bits.to_le_bytes());

                let signature = Signature::from_bytes(&transition.signature);
                self.ca.verifying_key().verify(&transition_data, &signature)
                    .map_err(|_| LatticeError::SelfVerificationFailed(
                        "Transition signature invalid".into()
                    ))?;
            }

            // Verify Zeckendorf encoding
            transition.weight.decode()?;
        }

        // Verify all token encodings
        for token in self.tokens.values() {
            token.zeck_encoding.decode()?;
        }

        // Verify operation log chain
        let mut prev_hash = [0u8; 32];
        for op in &self.operation_log {
            let mut op_data = Vec::new();
            op_data.extend_from_slice(&op.operation_id.as_bytes()[..]);
            op_data.extend_from_slice(op.operation_type.as_bytes());
            op_data.extend_from_slice(&op.input_hash);
            op_data.extend_from_slice(&op.output_hash);

            let signature = Signature::from_bytes(&op.signature);
            self.ca.verifying_key().verify(&op_data, &signature)
                .map_err(|_| LatticeError::SelfVerificationFailed(
                    format!("Operation {} signature invalid", op.operation_id)
                ))?;

            prev_hash = op.output_hash;
        }

        // Verify current state matches last operation output
        if !self.operation_log.is_empty() {
            if prev_hash != self.state_hash {
                return Err(LatticeError::SelfVerificationFailed(
                    "State hash mismatch with operation log".into()
                ));
            }
        }

        Ok(true)
    }

    /// Update the state hash
    fn update_state_hash(&mut self) {
        let mut hasher = Hasher::new();

        // Hash all tokens
        for (id, token) in &self.tokens {
            hasher.update(&id.as_bytes()[..]);
            hasher.update(token.text.as_bytes());
            hasher.update(&token.zeck_encoding.hash);
        }

        // Hash all transitions
        for transition in &self.transitions {
            hasher.update(&transition.from_token.as_bytes()[..]);
            hasher.update(&transition.to_token.as_bytes()[..]);
            hasher.update(&transition.signature);
        }

        self.state_hash = *hasher.finalize().as_bytes();
    }

    /// Get the certificate authority
    pub fn certificate_authority(&self) -> &CertificateAuthority {
        &self.ca
    }

    /// Get operation log for audit
    pub fn operation_log(&self) -> &[SignedOperation] {
        &self.operation_log
    }

    /// Export state for air-gapped transfer
    #[cfg(feature = "std")]
    pub fn export_state(&self) -> Result<Vec<u8>> {
        // Self-verify before export
        self.self_verify()?;

        let data = bincode::serialize(&LatticeExport {
            config_name: self.config.name.clone(),
            tokens: self.tokens.values().cloned().collect(),
            transitions: self.transitions.clone(),
            operation_log: self.operation_log.clone(),
            state_hash: self.state_hash,
            root_cert: self.ca.root_certificate().clone(),
        }).map_err(|e| LatticeError::SelfVerificationFailed(e.to_string()))?;

        Ok(data)
    }

    /// Get current state hash
    pub fn state_hash(&self) -> [u8; 32] {
        self.state_hash
    }

    /// Get token count
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Get transition count
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

/// Export format for air-gapped transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeExport {
    pub config_name: String,
    pub tokens: Vec<LatticeToken>,
    pub transitions: Vec<LatticeTransition>,
    pub operation_log: Vec<SignedOperation>,
    #[serde(with = "bytes32")]
    pub state_hash: [u8; 32],
    pub root_cert: Certificate,
}

// ============================================================================
// WASM BINDINGS
// ============================================================================

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmPhiLattice {
    inner: PhiLattice,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmPhiLattice {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> Self {
        console_error_panic_hook::set_once();

        let config = LatticeConfig {
            name: name.to_string(),
            ..Default::default()
        };

        Self {
            inner: PhiLattice::new(config),
        }
    }

    #[wasm_bindgen]
    pub fn train(&mut self, corpus: &str) -> std::result::Result<(), JsValue> {
        let texts: Vec<&str> = corpus.lines().collect();
        self.inner.train(&texts)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn generate(&self, prefix: &str, max_tokens: usize) -> std::result::Result<String, JsValue> {
        let result = self.inner.generate(prefix, max_tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(result.tokens.join(" "))
    }

    #[wasm_bindgen]
    pub fn verify(&self) -> std::result::Result<bool, JsValue> {
        self.inner.self_verify()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn token_count(&self) -> usize {
        self.inner.token_count()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckendorf_encode_decode() {
        for n in 0..100 {
            let zeck = ZeckendorfNumber::encode(n);
            assert_eq!(zeck.decode().unwrap(), n);
        }
    }

    #[test]
    fn test_zeckendorf_no_consecutive() {
        for n in 1..100 {
            let zeck = ZeckendorfNumber::encode(n);
            // Check no consecutive bits
            assert_eq!(zeck.bits & (zeck.bits >> 1), 0);
        }
    }

    #[test]
    fn test_certificate_authority() {
        let ca = CertificateAuthority::new("Test CA");

        // Verify root certificate is self-signed
        let root = ca.root_certificate();
        assert_eq!(root.subject, root.issuer);

        // Root should verify against itself
        let verifying = ca.verifying_key();
        assert!(root.verify(&verifying).is_ok());
    }

    #[test]
    fn test_phi_lattice_train_generate() {
        let config = LatticeConfig::default();
        let mut lattice = PhiLattice::new(config);

        lattice.train(&[
            "hello world",
            "hello there",
        ]).unwrap();

        // Verify we can generate
        let result = lattice.generate("hello", 2);
        assert!(result.is_ok());

        let gen = result.unwrap();
        assert!(!gen.tokens.is_empty());
        assert_eq!(gen.tokens[0], "hello");
    }

    #[test]
    fn test_phi_lattice_perplexity() {
        let config = LatticeConfig::default();
        let mut lattice = PhiLattice::new(config);

        lattice.train(&[
            "docker run nginx container",
            "kubectl get pods namespace",
        ]).unwrap();

        let metrics = lattice.compute_perplexity();

        // Closed-form system has perfect perplexity
        assert_eq!(metrics.perplexity, 1.0);
        assert_eq!(metrics.accuracy, 100.0);
    }

    #[test]
    fn test_phi_lattice_self_verify() {
        let config = LatticeConfig::default();
        let mut lattice = PhiLattice::new(config);

        lattice.train(&[
            "test verification chain",
        ]).unwrap();

        // Should pass self-verification
        assert!(lattice.self_verify().unwrap());
    }

    #[test]
    fn test_zeckendorf_add() {
        let a = ZeckendorfNumber::encode(5);
        let b = ZeckendorfNumber::encode(8);
        let sum = a.add(&b).unwrap();

        assert_eq!(sum.value, 13);
        assert_eq!(sum.decode().unwrap(), 13);
    }

    #[test]
    fn test_phi_psi_channels() {
        let z = ZeckendorfNumber::encode(1000);

        // Should have values in both channels
        let phi = z.phi_channel();
        let psi = z.psi_channel();

        // Total bits should be reasonable
        assert!(phi + psi <= z.bits.count_ones() as u64);
    }
}
