//! # Leviathan Compliance Framework
//!
//! Bank-grade regulatory compliance for Northern Trust and financial institutions.
//! Implements FFIEC, BCBS 239, SR 11-7, GDPR, and other regulatory frameworks.
//!
//! ## Features
//!
//! - **FFIEC**: IT examination handbook compliance
//! - **BCBS 239**: Risk data aggregation and governance
//! - **SR 11-7**: Model risk management
//! - **GDPR**: Data protection and privacy
//! - **Audit Trail**: Cryptographic evidence and reporting
//! - **Automated Validation**: Continuous compliance monitoring
//!
//! ## Example
//!
//! ```rust
//! use leviathan_compliance::{ComplianceValidator, ComplianceFramework};
//!
//! let validator = ComplianceValidator::new();
//! let report = validator.validate_framework(ComplianceFramework::BCBS239);
//! ```

pub mod ffiec;
pub mod bcbs239;
pub mod sr117;
pub mod gdpr;
pub mod audit_report;
pub mod validator;
pub mod controls;

// Re-export commonly used types
pub use validator::ComplianceValidator;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Compliance framework errors
#[derive(Debug, Error)]
pub enum ComplianceError {
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Required control missing: {0}")]
    ControlMissing(String),

    #[error("Evidence insufficient for requirement: {0}")]
    InsufficientEvidence(String),

    #[error("Framework not supported: {0}")]
    UnsupportedFramework(String),

    #[error("Audit trail verification failed: {0}")]
    AuditTrailVerificationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ComplianceError>;

/// Base trait for all compliance requirements
pub trait ComplianceRequirement {
    /// Unique identifier for the requirement
    fn requirement_id(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Regulatory framework this belongs to
    fn framework(&self) -> ComplianceFramework;

    /// Severity if non-compliant
    fn severity(&self) -> Severity;

    /// Validate compliance
    fn validate(&self, evidence: &[Evidence]) -> Result<ValidationResult>;
}

/// Compliance frameworks supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    /// Federal Financial Institutions Examination Council
    FFIEC,
    /// Basel Committee on Banking Supervision 239
    BCBS239,
    /// Federal Reserve SR 11-7 Model Risk Management
    SR117,
    /// General Data Protection Regulation
    GDPR,
    /// Comprehensive Capital Analysis and Review
    CCAR,
    /// Sarbanes-Oxley Act
    SOX,
    /// Custom framework
    Custom,
}

/// Severity levels for compliance violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational - no impact
    Info,
    /// Low severity - minor impact
    Low,
    /// Medium severity - moderate impact
    Medium,
    /// High severity - significant impact
    High,
    /// Critical - severe regulatory risk
    Critical,
}

/// Validation result for a single requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub requirement_id: String,
    pub compliant: bool,
    pub findings: Vec<String>,
    pub evidence_count: usize,
    pub validated_at: DateTime<Utc>,
    pub severity: Severity,
}

/// Evidence supporting compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: Uuid,
    pub evidence_type: EvidenceType,
    pub description: String,
    pub artifact_hash: [u8; 32], // Blake3 hash
    pub collected_at: DateTime<Utc>,
    pub collected_by: String,
    pub metadata: HashMap<String, String>,
}

/// Types of compliance evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Documentation (policies, procedures)
    Documentation,
    /// System logs and audit trails
    AuditLog,
    /// Test results and validation reports
    TestResult,
    /// Code review and security scan results
    CodeReview,
    /// Data lineage and provenance
    DataLineage,
    /// Access control records
    AccessControl,
    /// Cryptographic proof
    CryptographicProof,
    /// Third-party attestation
    Attestation,
    /// Other evidence type
    Other(String),
}

impl Evidence {
    /// Create new evidence with Blake3 hash
    pub fn new(
        evidence_type: EvidenceType,
        description: String,
        artifact: &[u8],
        collected_by: String,
    ) -> Self {
        let artifact_hash: [u8; 32] = blake3::hash(artifact).into();

        Self {
            id: Uuid::new_v4(),
            evidence_type,
            description,
            artifact_hash,
            collected_at: Utc::now(),
            collected_by,
            metadata: HashMap::new(),
        }
    }

    /// Verify artifact matches stored hash
    pub fn verify_artifact(&self, artifact: &[u8]) -> bool {
        let computed_hash: [u8; 32] = blake3::hash(artifact).into();
        computed_hash == self.artifact_hash
    }

    /// Add metadata to evidence
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Risk level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Control effectiveness rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlEffectiveness {
    /// Control is operating effectively
    Effective,
    /// Control has minor deficiencies
    PartiallyEffective,
    /// Control is not operating as designed
    Ineffective,
    /// Control has not been tested
    NotTested,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_creation() {
        let artifact = b"test compliance document";
        let evidence = Evidence::new(
            EvidenceType::Documentation,
            "Test policy document".to_string(),
            artifact,
            "auditor@example.com".to_string(),
        );

        assert!(evidence.verify_artifact(artifact));
        assert!(!evidence.verify_artifact(b"wrong content"));
    }

    #[test]
    fn test_evidence_metadata() {
        let artifact = b"test";
        let evidence = Evidence::new(
            EvidenceType::AuditLog,
            "Test".to_string(),
            artifact,
            "system".to_string(),
        )
        .with_metadata("version".to_string(), "1.0".to_string())
        .with_metadata("source".to_string(), "audit_system".to_string());

        assert_eq!(evidence.metadata.len(), 2);
        assert_eq!(evidence.metadata.get("version"), Some(&"1.0".to_string()));
    }
}
