//! Control Framework
//!
//! Pre-defined controls mapped to regulatory requirements with testing procedures.

use crate::{
    ComplianceError, ComplianceFramework, ControlEffectiveness, Evidence, Result, RiskLevel,
    Severity,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Control types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlType {
    /// Prevents issues from occurring
    Preventive,
    /// Detects issues when they occur
    Detective,
    /// Corrects issues after detection
    Corrective,
    /// Discourages undesirable behavior
    Deterrent,
    /// Limits damage from incidents
    Compensating,
}

/// Implementation approach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationType {
    /// Automated control (system-enforced)
    Automated,
    /// Manual control (human-executed)
    Manual,
    /// Combination of automated and manual
    Hybrid,
}

/// Control frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFrequency {
    Continuous,
    RealTime,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    AdHoc,
}

/// Control definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Control {
    pub control_id: String,
    pub name: String,
    pub description: String,
    pub control_type: ControlType,
    pub implementation: ImplementationType,
    pub frequency: ControlFrequency,
    pub owner: String,
    pub frameworks: Vec<ComplianceFramework>,
    pub requirement_mappings: Vec<String>,
    pub testing_procedure: TestingProcedure,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingProcedure {
    pub procedure_id: String,
    pub steps: Vec<String>,
    pub expected_evidence: Vec<String>,
    pub sampling_method: SamplingMethod,
    pub test_frequency: ControlFrequency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Test all items
    Complete,
    /// Random sample
    Random { sample_size: usize },
    /// Statistical sample
    Statistical { confidence_level: f64 },
    /// Targeted/judgmental sample
    Targeted { criteria: String },
}

/// Control test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlTest {
    pub test_id: Uuid,
    pub control_id: String,
    pub tested_at: DateTime<Utc>,
    pub tester: String,
    pub test_period: (DateTime<Utc>, DateTime<Utc>),
    pub sample_size: usize,
    pub exceptions: Vec<ControlException>,
    pub effectiveness: ControlEffectiveness,
    pub evidence: Vec<Evidence>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlException {
    pub exception_id: Uuid,
    pub description: String,
    pub severity: Severity,
    pub root_cause: Option<String>,
    pub remediation: Option<String>,
    pub remediation_due: Option<DateTime<Utc>>,
}

impl ControlTest {
    pub fn new(control_id: String, tester: String, sample_size: usize) -> Self {
        let now = Utc::now();
        let period_start = now - chrono::Duration::days(30);

        Self {
            test_id: Uuid::new_v4(),
            control_id,
            tested_at: now,
            tester,
            test_period: (period_start, now),
            sample_size,
            exceptions: Vec::new(),
            effectiveness: ControlEffectiveness::NotTested,
            evidence: Vec::new(),
            notes: String::new(),
        }
    }

    pub fn add_exception(&mut self, exception: ControlException) {
        self.exceptions.push(exception);
    }

    pub fn calculate_effectiveness(&mut self) {
        if self.sample_size == 0 {
            self.effectiveness = ControlEffectiveness::NotTested;
            return;
        }

        let exception_rate = self.exceptions.len() as f64 / self.sample_size as f64;

        self.effectiveness = if exception_rate == 0.0 {
            ControlEffectiveness::Effective
        } else if exception_rate < 0.05 {
            // Less than 5% exception rate
            ControlEffectiveness::PartiallyEffective
        } else {
            ControlEffectiveness::Ineffective
        };
    }
}

/// Remediation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Remediation {
    pub remediation_id: Uuid,
    pub control_id: String,
    pub exception_id: Uuid,
    pub description: String,
    pub assigned_to: String,
    pub created_at: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub status: RemediationStatus,
    pub completion_evidence: Option<Evidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationStatus {
    Open,
    InProgress,
    PendingVerification,
    Completed,
    Overdue,
}

impl Remediation {
    pub fn new(
        control_id: String,
        exception_id: Uuid,
        description: String,
        assigned_to: String,
        due_date: DateTime<Utc>,
    ) -> Self {
        Self {
            remediation_id: Uuid::new_v4(),
            control_id,
            exception_id,
            description,
            assigned_to,
            created_at: Utc::now(),
            due_date,
            status: RemediationStatus::Open,
            completion_evidence: None,
        }
    }

    pub fn is_overdue(&self) -> bool {
        Utc::now() > self.due_date && !matches!(self.status, RemediationStatus::Completed)
    }
}

/// Pre-defined controls library
pub struct ControlLibrary;

impl ControlLibrary {
    /// Access Control - Multi-Factor Authentication
    pub fn mfa_control() -> Control {
        Control {
            control_id: "AC-001".to_string(),
            name: "Multi-Factor Authentication".to_string(),
            description: "Require MFA for all privileged account access".to_string(),
            control_type: ControlType::Preventive,
            implementation: ImplementationType::Automated,
            frequency: ControlFrequency::Continuous,
            owner: "IT Security".to_string(),
            frameworks: vec![ComplianceFramework::FFIEC, ComplianceFramework::GDPR],
            requirement_mappings: vec!["FFIEC-IS-001".to_string(), "GDPR-ART25".to_string()],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-AC-001".to_string(),
                steps: vec![
                    "Review authentication logs".to_string(),
                    "Attempt privileged access without MFA".to_string(),
                    "Verify MFA is enforced".to_string(),
                    "Test MFA bypass scenarios".to_string(),
                ],
                expected_evidence: vec![
                    "Authentication configuration".to_string(),
                    "Failed login attempts".to_string(),
                    "MFA enrollment records".to_string(),
                ],
                sampling_method: SamplingMethod::Random { sample_size: 25 },
                test_frequency: ControlFrequency::Quarterly,
            },
            risk_level: RiskLevel::VeryHigh,
        }
    }

    /// Data Encryption - At Rest
    pub fn encryption_at_rest() -> Control {
        Control {
            control_id: "DC-001".to_string(),
            name: "Data Encryption at Rest".to_string(),
            description: "Encrypt all sensitive data at rest using AES-256".to_string(),
            control_type: ControlType::Preventive,
            implementation: ImplementationType::Automated,
            frequency: ControlFrequency::Continuous,
            owner: "Data Security".to_string(),
            frameworks: vec![
                ComplianceFramework::FFIEC,
                ComplianceFramework::GDPR,
                ComplianceFramework::BCBS239,
            ],
            requirement_mappings: vec![
                "FFIEC-IS-002".to_string(),
                "GDPR-ART25".to_string(),
                "BCBS239-P3".to_string(),
            ],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-DC-001".to_string(),
                steps: vec![
                    "Review encryption configuration".to_string(),
                    "Verify encryption algorithms".to_string(),
                    "Test key management procedures".to_string(),
                    "Validate encryption at storage layer".to_string(),
                ],
                expected_evidence: vec![
                    "Encryption configuration".to_string(),
                    "Key management documentation".to_string(),
                    "Storage encryption proof".to_string(),
                ],
                sampling_method: SamplingMethod::Statistical {
                    confidence_level: 0.95,
                },
                test_frequency: ControlFrequency::Quarterly,
            },
            risk_level: RiskLevel::VeryHigh,
        }
    }

    /// Change Management
    pub fn change_management() -> Control {
        Control {
            control_id: "CM-001".to_string(),
            name: "Change Management Process".to_string(),
            description: "All production changes require approval and testing".to_string(),
            control_type: ControlType::Preventive,
            implementation: ImplementationType::Hybrid,
            frequency: ControlFrequency::AdHoc,
            owner: "DevOps".to_string(),
            frameworks: vec![
                ComplianceFramework::FFIEC,
                ComplianceFramework::SR117,
                ComplianceFramework::SOX,
            ],
            requirement_mappings: vec!["FFIEC-DA-001".to_string(), "SR117-GOV-001".to_string()],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-CM-001".to_string(),
                steps: vec![
                    "Select sample of production changes".to_string(),
                    "Verify change approval documentation".to_string(),
                    "Confirm testing was performed".to_string(),
                    "Review rollback procedures".to_string(),
                ],
                expected_evidence: vec![
                    "Change request tickets".to_string(),
                    "Approval records".to_string(),
                    "Test results".to_string(),
                    "Deployment logs".to_string(),
                ],
                sampling_method: SamplingMethod::Random { sample_size: 30 },
                test_frequency: ControlFrequency::Quarterly,
            },
            risk_level: RiskLevel::High,
        }
    }

    /// Data Quality Validation - BCBS 239
    pub fn data_quality_validation() -> Control {
        Control {
            control_id: "DQ-001".to_string(),
            name: "Data Quality Validation".to_string(),
            description: "Automated data quality checks for risk data aggregation".to_string(),
            control_type: ControlType::Detective,
            implementation: ImplementationType::Automated,
            frequency: ControlFrequency::Daily,
            owner: "Risk Data Management".to_string(),
            frameworks: vec![ComplianceFramework::BCBS239],
            requirement_mappings: vec!["BCBS239-P3".to_string(), "BCBS239-P4".to_string()],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-DQ-001".to_string(),
                steps: vec![
                    "Review data quality metrics".to_string(),
                    "Verify validation rules execution".to_string(),
                    "Test exception handling".to_string(),
                    "Confirm reconciliation processes".to_string(),
                ],
                expected_evidence: vec![
                    "Data quality dashboards".to_string(),
                    "Validation rule configuration".to_string(),
                    "Exception reports".to_string(),
                    "Reconciliation results".to_string(),
                ],
                sampling_method: SamplingMethod::Targeted {
                    criteria: "High-risk data elements".to_string(),
                },
                test_frequency: ControlFrequency::Monthly,
            },
            risk_level: RiskLevel::VeryHigh,
        }
    }

    /// Model Validation - SR 11-7
    pub fn model_validation() -> Control {
        Control {
            control_id: "MV-001".to_string(),
            name: "Independent Model Validation".to_string(),
            description: "Annual independent validation of high-risk models".to_string(),
            control_type: ControlType::Detective,
            implementation: ImplementationType::Manual,
            frequency: ControlFrequency::Annually,
            owner: "Model Risk Management".to_string(),
            frameworks: vec![ComplianceFramework::SR117],
            requirement_mappings: vec!["SR117-VAL-001".to_string()],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-MV-001".to_string(),
                steps: vec![
                    "Review model inventory".to_string(),
                    "Verify validation independence".to_string(),
                    "Assess validation scope adequacy".to_string(),
                    "Review validation findings".to_string(),
                ],
                expected_evidence: vec![
                    "Validation reports".to_string(),
                    "Independence documentation".to_string(),
                    "Finding remediation tracking".to_string(),
                ],
                sampling_method: SamplingMethod::Complete,
                test_frequency: ControlFrequency::Annually,
            },
            risk_level: RiskLevel::VeryHigh,
        }
    }

    /// Audit Logging
    pub fn audit_logging() -> Control {
        Control {
            control_id: "AL-001".to_string(),
            name: "Comprehensive Audit Logging".to_string(),
            description: "Log all critical system activities with tamper-proof storage".to_string(),
            control_type: ControlType::Detective,
            implementation: ImplementationType::Automated,
            frequency: ControlFrequency::Continuous,
            owner: "IT Security".to_string(),
            frameworks: vec![
                ComplianceFramework::FFIEC,
                ComplianceFramework::GDPR,
                ComplianceFramework::SOX,
            ],
            requirement_mappings: vec!["FFIEC-AM-001".to_string(), "GDPR-ART25".to_string()],
            testing_procedure: TestingProcedure {
                procedure_id: "TP-AL-001".to_string(),
                steps: vec![
                    "Review logging configuration".to_string(),
                    "Verify log completeness".to_string(),
                    "Test log integrity controls".to_string(),
                    "Confirm retention compliance".to_string(),
                ],
                expected_evidence: vec![
                    "Logging policy".to_string(),
                    "Sample audit logs".to_string(),
                    "Integrity verification results".to_string(),
                    "Retention schedule".to_string(),
                ],
                sampling_method: SamplingMethod::Random { sample_size: 50 },
                test_frequency: ControlFrequency::Quarterly,
            },
            risk_level: RiskLevel::High,
        }
    }

    /// Get all standard controls
    pub fn all_controls() -> Vec<Control> {
        vec![
            Self::mfa_control(),
            Self::encryption_at_rest(),
            Self::change_management(),
            Self::data_quality_validation(),
            Self::model_validation(),
            Self::audit_logging(),
        ]
    }

    /// Get controls by framework
    pub fn controls_for_framework(framework: ComplianceFramework) -> Vec<Control> {
        Self::all_controls()
            .into_iter()
            .filter(|c| c.frameworks.contains(&framework))
            .collect()
    }

    /// Get controls by risk level
    pub fn controls_by_risk(risk_level: RiskLevel) -> Vec<Control> {
        Self::all_controls()
            .into_iter()
            .filter(|c| c.risk_level == risk_level)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_library() {
        let controls = ControlLibrary::all_controls();
        assert_eq!(controls.len(), 6);

        // Verify unique IDs
        let ids: std::collections::HashSet<_> = controls.iter().map(|c| &c.control_id).collect();
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn test_control_test_effectiveness() {
        let mut test = ControlTest::new("AC-001".to_string(), "Auditor".to_string(), 100);

        test.calculate_effectiveness();
        assert_eq!(test.effectiveness, ControlEffectiveness::Effective);

        // Add exceptions
        for _ in 0..3 {
            test.add_exception(ControlException {
                exception_id: Uuid::new_v4(),
                description: "Test exception".to_string(),
                severity: Severity::Low,
                root_cause: None,
                remediation: None,
                remediation_due: None,
            });
        }

        test.calculate_effectiveness();
        assert_eq!(test.effectiveness, ControlEffectiveness::PartiallyEffective);
    }

    #[test]
    fn test_controls_for_framework() {
        let ffiec_controls = ControlLibrary::controls_for_framework(ComplianceFramework::FFIEC);
        assert!(ffiec_controls.len() > 0);

        for control in ffiec_controls {
            assert!(control.frameworks.contains(&ComplianceFramework::FFIEC));
        }
    }

    #[test]
    fn test_remediation_tracking() {
        let mut remediation = Remediation::new(
            "AC-001".to_string(),
            Uuid::new_v4(),
            "Fix MFA bypass".to_string(),
            "Security Team".to_string(),
            Utc::now() + chrono::Duration::days(30),
        );

        assert!(!remediation.is_overdue());

        remediation.due_date = Utc::now() - chrono::Duration::days(1);
        assert!(remediation.is_overdue());

        remediation.status = RemediationStatus::Completed;
        assert!(!remediation.is_overdue());
    }
}
