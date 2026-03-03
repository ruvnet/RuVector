//! Zero-trust verification pipeline for incoming memories

use rvf_crypto::WitnessEntry;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("PII detected in field '{field}': {detail}")]
    PiiDetected { field: String, detail: String },
    #[error("Invalid embedding: {0}")]
    InvalidEmbedding(String),
    #[error("Content too large: {field} is {size} bytes, max {max}")]
    ContentTooLarge { field: String, size: usize, max: usize },
    #[error("Invalid witness hash: {0}")]
    InvalidWitness(String),
    #[error("Signature verification failed: {0}")]
    SignatureFailed(String),
    #[error("Too many tags: {count}, max {max}")]
    TooManyTags { count: usize, max: usize },
    #[error("Tag too long: {length} chars, max {max}")]
    TagTooLong { length: usize, max: usize },
}

/// Zero-trust verification for incoming data.
///
/// Holds a cached `PiiStripper` to avoid recompiling 12 regexes per call.
pub struct Verifier {
    max_title_len: usize,
    max_content_len: usize,
    max_tags: usize,
    max_tag_len: usize,
    max_embedding_dim: usize,
    max_embedding_magnitude: f32,
    /// Cached PII stripper — compiles 12 regexes once, reused across calls.
    pii_stripper: rvf_federation::PiiStripper,
}

impl Verifier {
    pub fn new() -> Self {
        Self {
            max_title_len: 200,
            max_content_len: 10_000,
            max_tags: 10,
            max_tag_len: 30,
            max_embedding_dim: 2048,
            max_embedding_magnitude: 100.0,
            pii_stripper: rvf_federation::PiiStripper::new(),
        }
    }

    /// Verify all fields of a share request
    pub fn verify_share(
        &self,
        title: &str,
        content: &str,
        tags: &[String],
        embedding: &[f32],
    ) -> Result<(), VerifyError> {
        self.verify_content_size(title, content, tags)?;
        self.verify_no_pii(title, "title")?;
        self.verify_no_pii(content, "content")?;
        for (i, tag) in tags.iter().enumerate() {
            self.verify_no_pii(tag, &format!("tags[{i}]"))?;
        }
        self.verify_embedding(embedding)?;
        Ok(())
    }

    /// Check content size limits
    fn verify_content_size(
        &self,
        title: &str,
        content: &str,
        tags: &[String],
    ) -> Result<(), VerifyError> {
        // Reject empty or whitespace-only titles
        if title.trim().is_empty() {
            return Err(VerifyError::ContentTooLarge {
                field: "title".into(),
                size: 0,
                max: self.max_title_len,
            });
        }
        if title.len() > self.max_title_len {
            return Err(VerifyError::ContentTooLarge {
                field: "title".into(),
                size: title.len(),
                max: self.max_title_len,
            });
        }
        if content.len() > self.max_content_len {
            return Err(VerifyError::ContentTooLarge {
                field: "content".into(),
                size: content.len(),
                max: self.max_content_len,
            });
        }
        if tags.len() > self.max_tags {
            return Err(VerifyError::TooManyTags {
                count: tags.len(),
                max: self.max_tags,
            });
        }
        for tag in tags {
            if tag.len() > self.max_tag_len {
                return Err(VerifyError::TagTooLong {
                    length: tag.len(),
                    max: self.max_tag_len,
                });
            }
        }
        Ok(())
    }

    /// Check for PII patterns using rvf-federation PiiStripper (12 regex rules).
    /// Delegates to `contains_pii()` for detection; rejection behavior is preserved.
    fn verify_no_pii(&self, text: &str, field: &str) -> Result<(), VerifyError> {
        if self.contains_pii(text) {
            return Err(VerifyError::PiiDetected {
                field: field.to_string(),
                detail: "PII detected by rvf-federation PiiStripper".to_string(),
            });
        }
        Ok(())
    }

    /// Check whether text contains PII using the cached PiiStripper (15 regex rules).
    /// Covers: Unix/Windows paths, IPv4/IPv6, emails, API keys, GitHub tokens,
    /// Bearer tokens, AWS keys, env vars, @usernames, phone numbers, SSNs, credit cards.
    pub fn contains_pii(&self, text: &str) -> bool {
        self.pii_stripper.contains_pii(text)
    }

    /// Strip PII from named fields using the cached PiiStripper.
    /// Returns (redacted fields, RedactionLog attestation).
    pub fn strip_pii_fields(
        &mut self,
        fields: &[(&str, &str)],
    ) -> (Vec<(String, String)>, rvf_federation::RedactionLog) {
        self.pii_stripper.strip_fields(fields)
    }

    /// Verify embedding is valid (no NaN, Inf, extreme magnitudes)
    fn verify_embedding(&self, embedding: &[f32]) -> Result<(), VerifyError> {
        if embedding.is_empty() {
            return Err(VerifyError::InvalidEmbedding("empty embedding".into()));
        }
        if embedding.len() > self.max_embedding_dim {
            return Err(VerifyError::InvalidEmbedding(format!(
                "dimension {} exceeds max {}",
                embedding.len(),
                self.max_embedding_dim
            )));
        }
        for (i, &val) in embedding.iter().enumerate() {
            if val.is_nan() {
                return Err(VerifyError::InvalidEmbedding(format!(
                    "NaN at index {i}"
                )));
            }
            if val.is_infinite() {
                return Err(VerifyError::InvalidEmbedding(format!(
                    "Inf at index {i}"
                )));
            }
            if val.abs() > self.max_embedding_magnitude {
                return Err(VerifyError::InvalidEmbedding(format!(
                    "magnitude {val} at index {i} exceeds max {}",
                    self.max_embedding_magnitude
                )));
            }
        }
        Ok(())
    }

    /// Verify an Ed25519 signature over a message.
    /// Used to check RVF container signatures on incoming memories.
    pub fn verify_ed25519_signature(
        &self,
        public_key_bytes: &[u8; 32],
        message: &[u8],
        signature_bytes: &[u8; 64],
    ) -> Result<(), VerifyError> {
        use ed25519_dalek::{Signature, VerifyingKey};
        use ed25519_dalek::Verifier as _;

        let key = VerifyingKey::from_bytes(public_key_bytes)
            .map_err(|e| VerifyError::SignatureFailed(format!("Invalid public key: {e}")))?;
        let sig = Signature::from_bytes(signature_bytes);
        key.verify(message, &sig)
            .map_err(|e| VerifyError::SignatureFailed(format!("Ed25519 verification failed: {e}")))?;
        Ok(())
    }

    /// Verify a SHAKE-256 witness chain.
    /// Each step in the chain hashes the previous hash + the step label.
    /// Returns Ok if the final hash matches `expected_hash`.
    pub fn verify_witness_chain(
        &self,
        steps: &[&str],
        expected_hash: &str,
    ) -> Result<(), VerifyError> {
        use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

        let mut current = [0u8; 32];

        for step in steps {
            let mut hasher = Shake256::default();
            hasher.update(&current);
            hasher.update(step.as_bytes());
            let mut reader = hasher.finalize_xof();
            reader.read(&mut current);
        }

        let computed = hex::encode(current);
        // Constant-time comparison
        if computed.len() != expected_hash.len() {
            return Err(VerifyError::InvalidWitness(
                "Witness hash length mismatch".to_string(),
            ));
        }
        let equal = subtle::ConstantTimeEq::ct_eq(computed.as_bytes(), expected_hash.as_bytes());
        if bool::from(equal) {
            Ok(())
        } else {
            Err(VerifyError::InvalidWitness(
                "Witness chain verification failed".to_string(),
            ))
        }
    }

    /// Verify a SHAKE-256 content hash matches data.
    /// Delegates to rvf_crypto::shake256_256 with constant-time comparison.
    pub fn verify_content_hash(
        &self,
        data: &[u8],
        expected_hex: &str,
    ) -> Result<(), VerifyError> {
        let computed_bytes = rvf_crypto::shake256_256(data);
        let computed = hex::encode(computed_bytes);
        let equal = subtle::ConstantTimeEq::ct_eq(computed.as_bytes(), expected_hex.as_bytes());
        if bool::from(equal) {
            Ok(())
        } else {
            Err(VerifyError::InvalidWitness(
                "Content hash verification failed".to_string(),
            ))
        }
    }

    /// Verify a binary witness chain produced by rvf_crypto::create_witness_chain.
    /// Returns the decoded WitnessEntry values if the chain is valid.
    pub fn verify_rvf_witness_chain(
        &self,
        chain_data: &[u8],
    ) -> Result<Vec<WitnessEntry>, VerifyError> {
        rvf_crypto::verify_witness_chain(chain_data)
            .map_err(|e| VerifyError::InvalidWitness(format!("RVF witness chain invalid: {e}")))
    }

    /// Check whether embedding distances indicate an adversarial (degenerate) distribution.
    /// Returns true if the distribution is too uniform to trust centroid routing.
    /// Uses rvf_runtime::is_degenerate_distribution (CV < 0.05 threshold).
    pub fn verify_embedding_not_adversarial(
        distances: &[f32],
        n_probe: usize,
    ) -> bool {
        rvf_runtime::is_degenerate_distribution(distances, n_probe)
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_clean_data() {
        let v = Verifier::new();
        assert!(v.verify_share("Good title", "Clean content", &["tag1".into()], &[0.1, 0.2, 0.3]).is_ok());
    }

    #[test]
    fn test_reject_pii() {
        let v = Verifier::new();
        assert!(v.verify_share("Has /home/user path", "content", &[], &[0.1]).is_err());
        // PiiStripper requires sk- followed by 20+ alphanums (realistic API key length)
        assert!(v.verify_share("title", "has sk-abcdefghijklmnopqrstuvwxyz", &[], &[0.1]).is_err());
    }

    #[test]
    fn test_reject_nan_embedding() {
        let v = Verifier::new();
        assert!(v.verify_share("title", "content", &[], &[0.1, f32::NAN, 0.3]).is_err());
    }

    #[test]
    fn test_reject_inf_embedding() {
        let v = Verifier::new();
        assert!(v.verify_share("title", "content", &[], &[0.1, f32::INFINITY, 0.3]).is_err());
    }

    #[test]
    fn test_reject_oversized_title() {
        let v = Verifier::new();
        let long_title = "a".repeat(201);
        assert!(v.verify_share(&long_title, "content", &[], &[0.1]).is_err());
    }

    #[test]
    fn test_reject_too_many_tags() {
        let v = Verifier::new();
        let tags: Vec<String> = (0..11).map(|i| format!("tag{i}")).collect();
        assert!(v.verify_share("title", "content", &tags, &[0.1]).is_err());
    }

    #[test]
    fn test_verify_witness_chain() {
        let v = Verifier::new();
        // Build expected hash from steps
        use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
        let steps = ["pii_strip", "embed", "share"];
        let mut current = [0u8; 32];
        for step in &steps {
            let mut hasher = Shake256::default();
            hasher.update(&current);
            hasher.update(step.as_bytes());
            let mut reader = hasher.finalize_xof();
            reader.read(&mut current);
        }
        let expected = hex::encode(current);
        assert!(v.verify_witness_chain(&steps, &expected).is_ok());
        assert!(v.verify_witness_chain(&steps, "0000000000000000000000000000000000000000000000000000000000000000").is_err());
    }

    #[test]
    fn test_verify_content_hash() {
        let v = Verifier::new();
        let data = b"hello world";
        let buf = rvf_crypto::shake256_256(data);
        let expected = hex::encode(buf);
        assert!(v.verify_content_hash(data, &expected).is_ok());
        assert!(v.verify_content_hash(b"tampered", &expected).is_err());
    }

    #[test]
    fn test_ed25519_signature() {
        use ed25519_dalek::{SigningKey, Signer};
        let v = Verifier::new();
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let message = b"test message for verification";
        let signature = signing_key.sign(message);
        let pub_key = signing_key.verifying_key().to_bytes();
        let sig_bytes: [u8; 64] = signature.to_bytes();
        assert!(v.verify_ed25519_signature(&pub_key, message, &sig_bytes).is_ok());
        // Tampered message should fail
        assert!(v.verify_ed25519_signature(&pub_key, b"tampered message", &sig_bytes).is_err());
    }

    #[test]
    fn test_rvf_witness_chain_roundtrip() {
        let v = Verifier::new();
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let entries = vec![
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"pii_strip"),
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"embed"),
                timestamp_ns: now_ns,
                witness_type: 0x02,
            },
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"content"),
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
        ];
        let chain = rvf_crypto::create_witness_chain(&entries);
        assert_eq!(chain.len(), 73 * 3); // 73 bytes per entry
        let decoded = v.verify_rvf_witness_chain(&chain).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].witness_type, 0x01);
        assert_eq!(decoded[1].witness_type, 0x02);
        assert_eq!(decoded[2].witness_type, 0x01);
    }

    #[test]
    fn test_pii_strip_redacts_paths() {
        let mut v = Verifier::new();
        let fields = [("content", "See /home/user/data/file.txt for details")];
        let (stripped, log) = v.strip_pii_fields(&fields);
        assert!(!stripped[0].1.contains("/home/"));
        assert!(log.total_redactions > 0);
    }

    #[test]
    fn test_pii_strip_redacts_email() {
        let mut v = Verifier::new();
        let fields = [("content", "Contact user@example.com for help")];
        let (stripped, log) = v.strip_pii_fields(&fields);
        assert!(!stripped[0].1.contains("user@example.com"));
        assert!(log.total_redactions > 0);
    }

    #[test]
    fn test_pii_detects_phone_number() {
        let v = Verifier::new();
        assert!(v.contains_pii("Call 555-867-5309 for info"));
    }

    #[test]
    fn test_pii_detects_ssn() {
        let v = Verifier::new();
        assert!(v.contains_pii("SSN: 078-05-1120"));
    }

    #[test]
    fn test_pii_detects_credit_card() {
        let v = Verifier::new();
        assert!(v.contains_pii("Card: 4111-1111-1111-1111"));
    }

    #[test]
    fn test_pii_strip_redacts_phone_ssn() {
        let mut v = Verifier::new();
        let fields = [("content", "phone 555-867-5309, ssn 078-05-1120")];
        let (stripped, log) = v.strip_pii_fields(&fields);
        assert!(!stripped[0].1.contains("555-867-5309"));
        assert!(!stripped[0].1.contains("078-05-1120"));
        assert!(log.total_redactions >= 2);
    }

    #[test]
    fn test_contains_pii_detects_api_key() {
        let v = Verifier::new();
        // PiiStripper sk- rule requires 20+ chars after prefix
        assert!(v.contains_pii("my key is sk-abcdefghijklmnopqrstuvwxyz"));
        // ghp_ rule requires exactly 36 alphanums after prefix
        assert!(v.contains_pii("token: ghp_abcdefghijklmnopqrstuvwxyz0123456789"));
        assert!(!v.contains_pii("clean text with no secrets"));
    }

    #[test]
    fn test_adversarial_degenerate_detection() {
        // Uniform distances should be flagged as degenerate
        let uniform = vec![1.0f32; 100];
        assert!(Verifier::verify_embedding_not_adversarial(&uniform, 10));
        // Varied distances should not be flagged
        let varied: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        assert!(!Verifier::verify_embedding_not_adversarial(&varied, 10));
    }
}
