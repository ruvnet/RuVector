//! # Security Notice for QuDAG Cryptography
//!
//! ## ⚠️ CRITICAL SECURITY WARNING ⚠️
//!
//! The cryptographic implementations in this module are **PLACEHOLDER IMPLEMENTATIONS**
//! intended for **API COMPATIBILITY TESTING ONLY**.
//!
//! ## Current Status
//!
//! | Component | Status | Security Level |
//! |-----------|--------|----------------|
//! | ML-DSA-65 | Placeholder (HMAC-SHA256) | **NOT QUANTUM-RESISTANT** |
//! | ML-KEM-768 | Placeholder (HKDF-SHA256) | **NOT QUANTUM-RESISTANT** |
//! | Differential Privacy | Working | Production-ready |
//! | Keystore | Working | Uses zeroize |
//!
//! ## Production Requirements
//!
//! Before deploying to production, you MUST:
//!
//! 1. **Replace ML-DSA-65** with a real implementation:
//!    - `pqcrypto-dilithium` (libpqcrypto bindings)
//!    - `ml-dsa` crate when available
//!    - NIST-approved ML-DSA implementation
//!
//! 2. **Replace ML-KEM-768** with a real implementation:
//!    - `pqcrypto-kyber` (libpqcrypto bindings)
//!    - `ml-kem` crate when available
//!    - NIST-approved ML-KEM implementation
//!
//! 3. **Audit all cryptographic code** with a qualified security professional
//!
//! ## What the Placeholders Provide
//!
//! - **ML-DSA**: HMAC-SHA256 based signatures (classical security only)
//! - **ML-KEM**: HKDF-SHA256 based key derivation (classical security only)
//!
//! These provide basic integrity and key derivation but:
//! - Are NOT resistant to quantum computer attacks
//! - Do NOT meet the security requirements of real ML-DSA/ML-KEM
//! - Should NEVER be used for sensitive data in production
//!
//! ## Threat Model Limitations
//!
//! The placeholders are vulnerable to:
//! - Quantum computer attacks (Shor's algorithm for DSA, Grover's for KEM)
//! - Side-channel attacks (no constant-time implementation)
//! - Sophisticated classical attacks (not formally verified)
//!
//! ## Recommended Production Configuration
//!
//! ```toml
//! [dependencies]
//! # When available, use real PQ crypto:
//! # pqcrypto-dilithium = "0.5"
//! # pqcrypto-kyber = "0.8"
//! # Or wait for official NIST crates
//! ```
//!
//! ## Security Contact
//!
//! Report security issues to: security@ruvector.io
//!
//! ## Compliance Notes
//!
//! - NIST PQC standardization: FIPS 203 (ML-KEM), FIPS 204 (ML-DSA)
//! - Current placeholders do NOT comply with these standards
//! - Migration path documented in docs/security/pq-migration.md

/// Compile-time security check
///
/// This function should be called during initialization to warn about
/// placeholder crypto usage.
#[cold]
pub fn check_crypto_security() {
    #[cfg(not(feature = "production-crypto"))]
    {
        tracing::warn!(
            "⚠️ SECURITY WARNING: Using placeholder cryptography. \
             NOT suitable for production. See security_notice.rs for details."
        );
    }
}

/// Runtime check for production readiness
pub fn is_production_ready() -> bool {
    #[cfg(feature = "production-crypto")]
    {
        true
    }
    #[cfg(not(feature = "production-crypto"))]
    {
        false
    }
}

/// Get security status report
pub fn security_status() -> SecurityStatus {
    SecurityStatus {
        ml_dsa_ready: false,
        ml_kem_ready: false,
        dp_ready: true,
        keystore_ready: true,
        production_ready: is_production_ready(),
    }
}

/// Security status of cryptographic components
#[derive(Debug, Clone)]
pub struct SecurityStatus {
    /// ML-DSA-65 uses real implementation
    pub ml_dsa_ready: bool,
    /// ML-KEM-768 uses real implementation
    pub ml_kem_ready: bool,
    /// Differential privacy is properly implemented
    pub dp_ready: bool,
    /// Keystore uses proper zeroization
    pub keystore_ready: bool,
    /// Overall production readiness
    pub production_ready: bool,
}

impl std::fmt::Display for SecurityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QuDAG Cryptography Security Status:")?;
        writeln!(
            f,
            "  ML-DSA-65:  {} ({})",
            if self.ml_dsa_ready { "✓" } else { "✗" },
            if self.ml_dsa_ready {
                "Production"
            } else {
                "PLACEHOLDER"
            }
        )?;
        writeln!(
            f,
            "  ML-KEM-768: {} ({})",
            if self.ml_kem_ready { "✓" } else { "✗" },
            if self.ml_kem_ready {
                "Production"
            } else {
                "PLACEHOLDER"
            }
        )?;
        writeln!(
            f,
            "  DP:         {} ({})",
            if self.dp_ready { "✓" } else { "✗" },
            if self.dp_ready { "Ready" } else { "Not Ready" }
        )?;
        writeln!(
            f,
            "  Keystore:   {} ({})",
            if self.keystore_ready { "✓" } else { "✗" },
            if self.keystore_ready {
                "Ready"
            } else {
                "Not Ready"
            }
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  OVERALL:    {}",
            if self.production_ready {
                "✓ PRODUCTION READY"
            } else {
                "✗ NOT PRODUCTION READY - DO NOT USE FOR SENSITIVE DATA"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_status() {
        let status = security_status();
        assert!(!status.ml_dsa_ready);
        assert!(!status.ml_kem_ready);
        assert!(status.dp_ready);
        assert!(status.keystore_ready);
    }

    #[test]
    fn test_production_ready() {
        // Without production-crypto feature, should not be ready
        assert!(!is_production_ready());
    }
}
