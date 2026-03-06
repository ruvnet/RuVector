//! Local processing pipeline: PII -> embed -> sign

use regex_lite::Regex;
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

/// Pipeline for processing knowledge before sharing.
/// Pre-compiles 12 PII regex patterns for efficient reuse.
pub struct BrainPipeline {
    pii_patterns: Vec<(Regex, &'static str)>,
}

impl BrainPipeline {
    pub fn new() -> Self {
        let patterns = vec![
            // 1. Unix home paths
            (Regex::new(r"/(?:home|Users|root)/[^\s]+").unwrap(), "[REDACTED_PATH]"),
            // 2. Windows paths
            (Regex::new(r"C:\\Users\\[^\s]+").unwrap(), "[REDACTED_PATH]"),
            // 3. API keys: sk-..., ghp_..., gho_..., xoxb-..., xoxp-..., AKIA...
            (Regex::new(r"(?:sk-|ghp_|gho_|xoxb-|xoxp-|AKIA)[A-Za-z0-9_/-]+").unwrap(), "[REDACTED_KEY]"),
            // 4. Bearer tokens
            (Regex::new(r"Bearer\s+[A-Za-z0-9_./-]+").unwrap(), "[REDACTED_TOKEN]"),
            // 5. IPv4 addresses
            (Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(), "[REDACTED_IP]"),
            // 6. Email addresses
            (Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(), "[REDACTED_EMAIL]"),
            // 7. SSH keys
            (Regex::new(r"ssh-(?:rsa|ed25519|ecdsa)\s+[A-Za-z0-9+/=]+").unwrap(), "[REDACTED_SSH_KEY]"),
            // 8. JWT tokens
            (Regex::new(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+").unwrap(), "[REDACTED_JWT]"),
            // 9. Private keys in PEM
            (Regex::new(r"-----BEGIN[A-Z ]+PRIVATE KEY-----").unwrap(), "[REDACTED_PRIVATE_KEY]"),
            // 10. Secret/password/token assignments
            (Regex::new(r"(?i)(?:secret|password|passwd|token|key)\s*[:=]\s*['\x22]?[A-Za-z0-9+/=_-]{16,}").unwrap(), "[REDACTED_SECRET]"),
            // 11. API key / access token credentials
            (Regex::new(r"(?i)(?:api[_-]?key|access[_-]?token)\s*[:=]\s*['\x22]?[a-f0-9-]{32,}").unwrap(), "[REDACTED_CREDENTIAL]"),
            // 12. Internal hostnames
            (Regex::new(r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0|internal\.[a-z.]+)\b").unwrap(), "[REDACTED_HOST]"),
        ];
        Self { pii_patterns: patterns }
    }

    /// Strip PII from text using all 12 pattern categories
    pub fn strip_pii(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (pattern, replacement) in &self.pii_patterns {
            result = pattern.replace_all(&result, *replacement).to_string();
        }
        result
    }

    /// Check if text contains PII (any pattern matches)
    pub fn contains_pii(&self, text: &str) -> bool {
        self.pii_patterns.iter().any(|(pat, _)| pat.is_match(text))
    }

    /// Build a linked witness chain from a list of operations
    pub fn build_witness_chain(operations: &[&str]) -> WitnessChain {
        let mut chain = WitnessChain::new();
        for op in operations {
            chain.append(op);
        }
        chain
    }
}

impl Default for BrainPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// A single witness entry in the chain
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    pub action: String,
    pub hash: [u8; 32],
    pub timestamp_ns: u64,
}

/// Linked chain of witness entries using SHAKE-256.
/// Each entry's hash = SHAKE256(prev_hash || action || timestamp).
pub struct WitnessChain {
    entries: Vec<WitnessEntry>,
    prev_hash: [u8; 32],
}

impl WitnessChain {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            prev_hash: [0u8; 32],
        }
    }

    /// Append an action to the witness chain
    pub fn append(&mut self, action: &str) -> &WitnessEntry {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut hasher = Shake256::default();
        hasher.update(&self.prev_hash);
        hasher.update(action.as_bytes());
        hasher.update(&timestamp_ns.to_le_bytes());
        let mut reader = hasher.finalize_xof();
        let mut hash = [0u8; 32];
        reader.read(&mut hash);

        let entry = WitnessEntry {
            action: action.to_string(),
            hash,
            timestamp_ns,
        };

        self.prev_hash = hash;
        self.entries.push(entry);
        self.entries.last().unwrap()
    }

    /// Get the final witness hash as hex string
    pub fn finalize(&self) -> String {
        hex::encode(self.prev_hash)
    }

    /// Verify chain integrity by recomputing each hash
    pub fn verify(&self) -> bool {
        let mut prev = [0u8; 32];
        for entry in &self.entries {
            let mut hasher = Shake256::default();
            hasher.update(&prev);
            hasher.update(entry.action.as_bytes());
            hasher.update(&entry.timestamp_ns.to_le_bytes());
            let mut reader = hasher.finalize_xof();
            let mut expected = [0u8; 32];
            reader.read(&mut expected);
            if expected != entry.hash {
                return false;
            }
            prev = entry.hash;
        }
        true
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for WitnessChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a witness hash for an operation chain (backward compat)
pub fn witness_hash(operations: &[&str]) -> String {
    let mut chain = WitnessChain::new();
    for op in operations {
        chain.append(op);
    }
    chain.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_pii_unix_paths() {
        let pipeline = BrainPipeline::new();
        let input = "Found at /home/alice/secrets.txt";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("/home/alice"));
        assert!(output.contains("[REDACTED_PATH]"));
    }

    #[test]
    fn test_strip_pii_windows_paths() {
        let pipeline = BrainPipeline::new();
        let input = r"Found at C:\Users\bob\documents\secret.txt";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains(r"C:\Users\bob"));
        assert!(output.contains("[REDACTED_PATH]"));
    }

    #[test]
    fn test_strip_pii_api_keys() {
        let pipeline = BrainPipeline::new();
        let input = "Using key sk-abc123xyz and ghp_TokenValue123";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("sk-abc123xyz"));
        assert!(!output.contains("ghp_TokenValue123"));
        assert!(output.contains("[REDACTED_KEY]"));
    }

    #[test]
    fn test_strip_pii_bearer_token() {
        let pipeline = BrainPipeline::new();
        let input = "Header: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("eyJhbGciOiJSUzI1NiJ9"));
        assert!(output.contains("[REDACTED_TOKEN]"));
    }

    #[test]
    fn test_strip_pii_ip_address() {
        let pipeline = BrainPipeline::new();
        let input = "Server at 192.168.1.100 is running";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("192.168.1.100"));
        assert!(output.contains("[REDACTED_IP]"));
    }

    #[test]
    fn test_strip_pii_email() {
        let pipeline = BrainPipeline::new();
        let input = "Contact user@example.com for help";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("user@example.com"));
        assert!(output.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn test_strip_pii_ssh_key() {
        let pipeline = BrainPipeline::new();
        let input = "Key: ssh-rsa AAAAB3NzaC1yc2EAAA";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("ssh-rsa AAAA"));
        assert!(output.contains("[REDACTED_SSH_KEY]"));
    }

    #[test]
    fn test_strip_pii_jwt() {
        let pipeline = BrainPipeline::new();
        let input = "Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("eyJhbGciOiJIUzI1NiJ9"));
        assert!(output.contains("[REDACTED_JWT]"));
    }

    #[test]
    fn test_strip_pii_private_key() {
        let pipeline = BrainPipeline::new();
        let input = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpA...\n-----END RSA PRIVATE KEY-----";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("-----BEGIN RSA PRIVATE KEY-----"));
        assert!(output.contains("[REDACTED_PRIVATE_KEY]"));
    }

    #[test]
    fn test_strip_pii_secret_assignment() {
        let pipeline = BrainPipeline::new();
        let input = "secret=abcdefghij1234567890abcdefghij12";
        let output = pipeline.strip_pii(input);
        assert!(output.contains("[REDACTED_SECRET]"));
    }

    #[test]
    fn test_strip_pii_localhost() {
        let pipeline = BrainPipeline::new();
        let input = "Connect to localhost for debugging";
        let output = pipeline.strip_pii(input);
        assert!(!output.contains("localhost"));
        assert!(output.contains("[REDACTED_HOST]"));
    }

    #[test]
    fn test_contains_pii_detects_patterns() {
        let pipeline = BrainPipeline::new();
        assert!(pipeline.contains_pii("path /home/user/.ssh"));
        assert!(pipeline.contains_pii("token sk-abc123"));
        assert!(pipeline.contains_pii("email user@host.com"));
        assert!(pipeline.contains_pii("server at 10.0.0.1 port 80"));
        assert!(!pipeline.contains_pii("this is clean text with no PII"));
    }

    #[test]
    fn test_contains_pii_clean_after_strip() {
        let pipeline = BrainPipeline::new();
        let dirty = "Send to user@example.com at /home/alice/docs from 192.168.1.1";
        let clean = pipeline.strip_pii(dirty);
        assert!(!pipeline.contains_pii(&clean));
    }

    #[test]
    fn test_witness_chain_integrity() {
        let mut chain = WitnessChain::new();
        chain.append("step_1");
        chain.append("step_2");
        chain.append("step_3");
        chain.append("step_4");
        chain.append("step_5");
        assert_eq!(chain.len(), 5);
        assert!(chain.verify());
    }

    #[test]
    fn test_witness_chain_tamper() {
        let mut chain = WitnessChain::new();
        chain.append("step_1");
        chain.append("step_2");
        chain.append("step_3");
        // Tamper with the second entry's hash
        chain.entries[1].hash[0] ^= 0xFF;
        assert!(!chain.verify());
    }

    #[test]
    fn test_witness_chain_finalize() {
        let mut chain = WitnessChain::new();
        chain.append("op1");
        chain.append("op2");
        let hex = chain.finalize();
        assert_eq!(hex.len(), 64); // 32 bytes hex-encoded
        assert_ne!(hex, "0".repeat(64));
    }

    #[test]
    fn test_witness_chain_empty() {
        let chain = WitnessChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.verify()); // empty chain is valid
        assert_eq!(chain.finalize(), "0".repeat(64));
    }

    #[test]
    fn test_witness_hash_backward_compat() {
        let hash = witness_hash(&["op1", "op2"]);
        assert_eq!(hash.len(), 64);
        assert_ne!(hash, "0".repeat(64));
    }

    #[test]
    fn test_build_witness_chain() {
        let chain = BrainPipeline::build_witness_chain(&["a", "b", "c"]);
        assert_eq!(chain.len(), 3);
        assert!(chain.verify());
    }

    #[test]
    fn test_pipeline_default() {
        let pipeline = BrainPipeline::default();
        // Default should work identically to new()
        let clean = pipeline.strip_pii("key sk-test123");
        assert!(clean.contains("[REDACTED_KEY]"));
    }
}
