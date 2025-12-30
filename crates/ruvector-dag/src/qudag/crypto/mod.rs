//! Quantum-Resistant Cryptography for QuDAG
//!
//! # Security Warning
//!
//! This module contains **PLACEHOLDER** implementations for ML-DSA and ML-KEM.
//! See [`security_notice`] module for full security status and production requirements.
//!
//! ## Production Readiness
//!
//! | Component | Status |
//! |-----------|--------|
//! | ML-DSA-65 | ⚠️ Placeholder (HMAC-SHA256) |
//! | ML-KEM-768 | ⚠️ Placeholder (HKDF-SHA256) |
//! | Differential Privacy | ✓ Production-ready |
//! | Keystore | ✓ Uses zeroize |
//!
//! Call [`security_notice::check_crypto_security()`] at startup to log warnings.

mod differential_privacy;
mod identity;
mod keystore;
mod ml_dsa;
mod ml_kem;
mod security_notice;

pub use differential_privacy::{DifferentialPrivacy, DpConfig};
pub use identity::{IdentityError, QuDagIdentity};
pub use keystore::{KeystoreError, SecureKeystore};
pub use ml_dsa::{
    DsaError, MlDsa65, MlDsa65PublicKey, MlDsa65SecretKey, Signature, ML_DSA_65_PUBLIC_KEY_SIZE,
    ML_DSA_65_SECRET_KEY_SIZE, ML_DSA_65_SIGNATURE_SIZE,
};
pub use ml_kem::{
    EncapsulatedKey, KemError, MlKem768, MlKem768PublicKey, MlKem768SecretKey,
    ML_KEM_768_CIPHERTEXT_SIZE, ML_KEM_768_PUBLIC_KEY_SIZE, ML_KEM_768_SECRET_KEY_SIZE,
    SHARED_SECRET_SIZE,
};
pub use security_notice::{
    check_crypto_security, is_production_ready, security_status, SecurityStatus,
};
