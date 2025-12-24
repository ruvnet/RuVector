//! FFIEC (Federal Financial Institutions Examination Council) Compliance
//!
//! Implements IT examination handbook requirements for financial institutions.
//! Based on FFIEC IT Examination Handbook (2021 edition).

use crate::{
    ComplianceError, ComplianceFramework, ComplianceRequirement, ControlEffectiveness, Evidence,
    Result, RiskLevel, Severity, ValidationResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FFIEC requirement categories from IT Examination Handbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FFIECCategory {
    /// Architecture, Infrastructure, and Operations
    ArchitectureInfrastructure,
    /// Audit and Monitoring
    AuditMonitoring,
    /// Business Continuity Planning
    BusinessContinuity,
    /// Development and Acquisition
    DevelopmentAcquisition,
    /// E-Banking
    EBanking,
    /// Information Security
    InformationSecurity,
    /// Management
    Management,
    /// Outsourcing Technology Services
    OutsourcingTech,
    /// Retail Payment Systems
    RetailPayments,
    /// Wholesale Payment Systems
    WholesalePayments,
}

/// FFIEC compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFIECRequirement {
    /// Requirement ID (e.g., "FFIEC-IS-001")
    pub id: String,
    /// Category
    pub category: FFIECCategory,
    /// Requirement description
    pub description: String,
    /// Control objectives
    pub control_objectives: Vec<String>,
    /// Expected evidence
    pub expected_evidence: Vec<String>,
    /// Risk level if non-compliant
    pub risk_level: RiskLevel,
    /// Related handbook reference
    pub handbook_reference: String,
}

impl ComplianceRequirement for FFIECRequirement {
    fn requirement_id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn framework(&self) -> ComplianceFramework {
        ComplianceFramework::FFIEC
    }

    fn severity(&self) -> Severity {
        match self.risk_level {
            RiskLevel::VeryLow => Severity::Info,
            RiskLevel::Low => Severity::Low,
            RiskLevel::Medium => Severity::Medium,
            RiskLevel::High => Severity::High,
            RiskLevel::VeryHigh => Severity::Critical,
        }
    }

    fn validate(&self, evidence: &[Evidence]) -> Result<ValidationResult> {
        let mut findings = Vec::new();
        let evidence_count = evidence.len();

        // Check if we have minimum evidence
        if evidence_count < self.expected_evidence.len() {
            findings.push(format!(
                "Insufficient evidence: expected {}, found {}",
                self.expected_evidence.len(),
                evidence_count
            ));
        }

        // Validate evidence completeness
        for expected in &self.expected_evidence {
            let found = evidence
                .iter()
                .any(|e| e.description.contains(expected));

            if !found {
                findings.push(format!("Missing evidence for: {}", expected));
            }
        }

        let compliant = findings.is_empty();

        Ok(ValidationResult {
            requirement_id: self.id.clone(),
            compliant,
            findings,
            evidence_count,
            validated_at: Utc::now(),
            severity: self.severity(),
        })
    }
}

/// FFIEC IT Examination assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFIECAssessment {
    pub assessment_id: String,
    pub institution_name: String,
    pub assessment_date: DateTime<Utc>,
    pub examiner: String,
    pub requirements: Vec<FFIECRequirement>,
    pub overall_rating: ControlEffectiveness,
    pub findings: Vec<Finding>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub finding_id: String,
    pub requirement_id: String,
    pub severity: Severity,
    pub description: String,
    pub remediation_required: bool,
    pub due_date: Option<DateTime<Utc>>,
}

/// Pre-defined FFIEC requirements for common IT controls
pub struct FFIECRequirements;

impl FFIECRequirements {
    /// Information Security - Access Control
    pub fn access_control() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-IS-001".to_string(),
            category: FFIECCategory::InformationSecurity,
            description: "Implement strong access controls and authentication mechanisms"
                .to_string(),
            control_objectives: vec![
                "Unique user identification".to_string(),
                "Multi-factor authentication for privileged access".to_string(),
                "Least privilege principle enforced".to_string(),
                "Regular access reviews".to_string(),
            ],
            expected_evidence: vec![
                "Access control policy".to_string(),
                "User access logs".to_string(),
                "MFA implementation documentation".to_string(),
                "Quarterly access review reports".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
            handbook_reference: "Information Security Booklet, Pages 32-45".to_string(),
        }
    }

    /// Audit and Monitoring - Logging
    pub fn audit_logging() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-AM-001".to_string(),
            category: FFIECCategory::AuditMonitoring,
            description: "Maintain comprehensive audit logs for all critical systems".to_string(),
            control_objectives: vec![
                "All transactions logged with timestamps".to_string(),
                "User actions tracked and attributed".to_string(),
                "Logs protected from tampering".to_string(),
                "Log retention meets regulatory requirements".to_string(),
            ],
            expected_evidence: vec![
                "Logging policy".to_string(),
                "Log integrity verification".to_string(),
                "Log retention schedule".to_string(),
                "Sample audit logs".to_string(),
            ],
            risk_level: RiskLevel::High,
            handbook_reference: "Audit and Monitoring Booklet, Pages 15-28".to_string(),
        }
    }

    /// Business Continuity - Disaster Recovery
    pub fn disaster_recovery() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-BCP-001".to_string(),
            category: FFIECCategory::BusinessContinuity,
            description: "Establish and test business continuity and disaster recovery plans"
                .to_string(),
            control_objectives: vec![
                "Documented BCP/DR plans".to_string(),
                "Regular testing (at least annually)".to_string(),
                "Recovery time objectives (RTO) defined".to_string(),
                "Recovery point objectives (RPO) defined".to_string(),
            ],
            expected_evidence: vec![
                "BCP/DR plan documentation".to_string(),
                "Testing results from last 12 months".to_string(),
                "RTO/RPO definitions".to_string(),
                "Backup verification logs".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
            handbook_reference: "Business Continuity Planning Booklet, Pages 10-25".to_string(),
        }
    }

    /// Information Security - Encryption
    pub fn data_encryption() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-IS-002".to_string(),
            category: FFIECCategory::InformationSecurity,
            description: "Encrypt sensitive data at rest and in transit".to_string(),
            control_objectives: vec![
                "Strong encryption algorithms (AES-256, TLS 1.2+)".to_string(),
                "Encryption key management".to_string(),
                "Certificate management".to_string(),
                "Data classification and encryption policy".to_string(),
            ],
            expected_evidence: vec![
                "Encryption policy".to_string(),
                "Key management procedures".to_string(),
                "TLS configuration evidence".to_string(),
                "Encryption inventory".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
            handbook_reference: "Information Security Booklet, Pages 50-62".to_string(),
        }
    }

    /// Development - Secure SDLC
    pub fn secure_sdlc() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-DA-001".to_string(),
            category: FFIECCategory::DevelopmentAcquisition,
            description: "Implement secure software development lifecycle (SDLC)".to_string(),
            control_objectives: vec![
                "Security requirements in design phase".to_string(),
                "Code review and security testing".to_string(),
                "Change management controls".to_string(),
                "Vulnerability management".to_string(),
            ],
            expected_evidence: vec![
                "SDLC policy documentation".to_string(),
                "Code review records".to_string(),
                "Security testing results".to_string(),
                "Change management logs".to_string(),
            ],
            risk_level: RiskLevel::High,
            handbook_reference: "Development and Acquisition Booklet, Pages 18-35".to_string(),
        }
    }

    /// Outsourcing - Third Party Risk Management
    pub fn third_party_risk() -> FFIECRequirement {
        FFIECRequirement {
            id: "FFIEC-OT-001".to_string(),
            category: FFIECCategory::OutsourcingTech,
            description: "Manage third-party technology service provider risk".to_string(),
            control_objectives: vec![
                "Due diligence before engagement".to_string(),
                "Contractual requirements for security".to_string(),
                "Regular vendor assessments".to_string(),
                "Incident notification requirements".to_string(),
            ],
            expected_evidence: vec![
                "Vendor risk assessment".to_string(),
                "Contracts with security requirements".to_string(),
                "SOC 2 or equivalent reports".to_string(),
                "Vendor monitoring reports".to_string(),
            ],
            risk_level: RiskLevel::High,
            handbook_reference: "Outsourcing Technology Services Booklet, Pages 8-22".to_string(),
        }
    }

    /// Get all standard FFIEC requirements
    pub fn all_requirements() -> Vec<FFIECRequirement> {
        vec![
            Self::access_control(),
            Self::audit_logging(),
            Self::disaster_recovery(),
            Self::data_encryption(),
            Self::secure_sdlc(),
            Self::third_party_risk(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Evidence, EvidenceType};

    #[test]
    fn test_access_control_requirement() {
        let req = FFIECRequirements::access_control();
        assert_eq!(req.id, "FFIEC-IS-001");
        assert_eq!(req.control_objectives.len(), 4);
        assert_eq!(req.risk_level, RiskLevel::VeryHigh);
    }

    #[test]
    fn test_validation_with_evidence() {
        let req = FFIECRequirements::audit_logging();
        let evidence = vec![
            Evidence::new(
                EvidenceType::Documentation,
                "Logging policy document".to_string(),
                b"policy content",
                "auditor".to_string(),
            ),
            Evidence::new(
                EvidenceType::AuditLog,
                "Log integrity verification results".to_string(),
                b"verification data",
                "system".to_string(),
            ),
        ];

        let result = req.validate(&evidence).unwrap();
        assert!(!result.compliant); // Still missing some expected evidence
        assert_eq!(result.evidence_count, 2);
    }

    #[test]
    fn test_all_requirements_coverage() {
        let requirements = FFIECRequirements::all_requirements();
        assert_eq!(requirements.len(), 6);

        // Verify all have unique IDs
        let ids: std::collections::HashSet<_> =
            requirements.iter().map(|r| &r.id).collect();
        assert_eq!(ids.len(), 6);
    }
}
