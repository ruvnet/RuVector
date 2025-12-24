//! BCBS 239 (Basel Committee on Banking Supervision 239) Compliance
//!
//! Principles for effective risk data aggregation and risk reporting.
//! Based on Basel Committee document "Principles for effective risk data aggregation
//! and risk reporting" (January 2013).

use crate::{
    ComplianceError, ComplianceFramework, ComplianceRequirement, Evidence, Result, RiskLevel,
    Severity, ValidationResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// 14 BCBS 239 Principles organized into 4 categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BCBS239Principle {
    // Overarching governance and infrastructure
    /// Principle 1: Governance
    Governance,
    /// Principle 2: Data Architecture and IT Infrastructure
    DataArchitecture,

    // Risk data aggregation capabilities
    /// Principle 3: Accuracy and Integrity
    AccuracyIntegrity,
    /// Principle 4: Completeness
    Completeness,
    /// Principle 5: Timeliness
    Timeliness,
    /// Principle 6: Adaptability
    Adaptability,

    // Risk reporting practices
    /// Principle 7: Accuracy
    ReportingAccuracy,
    /// Principle 8: Comprehensiveness
    Comprehensiveness,
    /// Principle 9: Clarity and Usefulness
    ClarityUsefulness,
    /// Principle 10: Frequency
    Frequency,
    /// Principle 11: Distribution
    Distribution,

    // Supervisory review, tools and cooperation
    /// Principle 12: Remedial Actions and Supervisory Measures
    RemedialActions,
    /// Principle 13: Home/Host Cooperation
    HomeHostCooperation,
    /// Principle 14: Compliance
    CompliancePrinciple,
}

impl BCBS239Principle {
    pub fn category(&self) -> &'static str {
        match self {
            Self::Governance | Self::DataArchitecture => "Overarching governance and infrastructure",
            Self::AccuracyIntegrity
            | Self::Completeness
            | Self::Timeliness
            | Self::Adaptability => "Risk data aggregation capabilities",
            Self::ReportingAccuracy
            | Self::Comprehensiveness
            | Self::ClarityUsefulness
            | Self::Frequency
            | Self::Distribution => "Risk reporting practices",
            Self::RemedialActions | Self::HomeHostCooperation | Self::CompliancePrinciple => {
                "Supervisory review, tools and cooperation"
            }
        }
    }

    pub fn number(&self) -> u8 {
        match self {
            Self::Governance => 1,
            Self::DataArchitecture => 2,
            Self::AccuracyIntegrity => 3,
            Self::Completeness => 4,
            Self::Timeliness => 5,
            Self::Adaptability => 6,
            Self::ReportingAccuracy => 7,
            Self::Comprehensiveness => 8,
            Self::ClarityUsefulness => 9,
            Self::Frequency => 10,
            Self::Distribution => 11,
            Self::RemedialActions => 12,
            Self::HomeHostCooperation => 13,
            Self::CompliancePrinciple => 14,
        }
    }
}

/// BCBS 239 compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCBS239Requirement {
    pub id: String,
    pub principle: BCBS239Principle,
    pub description: String,
    pub key_attributes: Vec<String>,
    pub validation_criteria: Vec<String>,
    pub risk_level: RiskLevel,
}

impl ComplianceRequirement for BCBS239Requirement {
    fn requirement_id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn framework(&self) -> ComplianceFramework {
        ComplianceFramework::BCBS239
    }

    fn severity(&self) -> Severity {
        match self.risk_level {
            RiskLevel::VeryLow => Severity::Low,
            RiskLevel::Low => Severity::Low,
            RiskLevel::Medium => Severity::Medium,
            RiskLevel::High => Severity::High,
            RiskLevel::VeryHigh => Severity::Critical,
        }
    }

    fn validate(&self, evidence: &[Evidence]) -> Result<ValidationResult> {
        let mut findings = Vec::new();

        for criterion in &self.validation_criteria {
            let found = evidence
                .iter()
                .any(|e| e.description.to_lowercase().contains(&criterion.to_lowercase()));

            if !found {
                findings.push(format!("Missing validation for: {}", criterion));
            }
        }

        Ok(ValidationResult {
            requirement_id: self.id.clone(),
            compliant: findings.is_empty(),
            findings,
            evidence_count: evidence.len(),
            validated_at: Utc::now(),
            severity: self.severity(),
        })
    }
}

/// Data lineage tracking for BCBS 239 compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    pub lineage_id: Uuid,
    pub data_element: String,
    pub source_system: String,
    pub transformation_steps: Vec<TransformationStep>,
    pub destination_system: String,
    pub created_at: DateTime<Utc>,
    pub last_verified: DateTime<Utc>,
    pub data_quality_score: f64, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStep {
    pub step_id: Uuid,
    pub description: String,
    pub transformation_logic: String,
    pub validation_rules: Vec<String>,
    pub executed_at: DateTime<Utc>,
    pub output_hash: [u8; 32], // Blake3 hash for verification
}

impl DataLineage {
    pub fn new(data_element: String, source_system: String, destination_system: String) -> Self {
        let now = Utc::now();
        Self {
            lineage_id: Uuid::new_v4(),
            data_element,
            source_system,
            transformation_steps: Vec::new(),
            destination_system,
            created_at: now,
            last_verified: now,
            data_quality_score: 1.0,
        }
    }

    pub fn add_transformation(
        &mut self,
        description: String,
        logic: String,
        rules: Vec<String>,
        output: &[u8],
    ) {
        let step = TransformationStep {
            step_id: Uuid::new_v4(),
            description,
            transformation_logic: logic,
            validation_rules: rules,
            executed_at: Utc::now(),
            output_hash: blake3::hash(output).into(),
        };
        self.transformation_steps.push(step);
    }

    pub fn verify_transformation(&self, step_index: usize, output: &[u8]) -> bool {
        if let Some(step) = self.transformation_steps.get(step_index) {
            let computed_hash: [u8; 32] = blake3::hash(output).into();
            computed_hash == step.output_hash
        } else {
            false
        }
    }
}

/// Risk data aggregation capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDataAggregation {
    pub assessment_id: Uuid,
    pub institution: String,
    pub assessment_date: DateTime<Utc>,
    pub accuracy_score: f64,
    pub completeness_score: f64,
    pub timeliness_score: f64,
    pub adaptability_score: f64,
    pub data_lineages: Vec<DataLineage>,
}

impl RiskDataAggregation {
    pub fn overall_score(&self) -> f64 {
        (self.accuracy_score
            + self.completeness_score
            + self.timeliness_score
            + self.adaptability_score)
            / 4.0
    }

    pub fn is_compliant(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

/// Pre-defined BCBS 239 requirements
pub struct BCBS239Requirements;

impl BCBS239Requirements {
    /// Principle 3: Accuracy and Integrity
    pub fn accuracy_integrity() -> BCBS239Requirement {
        BCBS239Requirement {
            id: "BCBS239-P3".to_string(),
            principle: BCBS239Principle::AccuracyIntegrity,
            description: "Risk data should be accurate and reflect positions precisely"
                .to_string(),
            key_attributes: vec![
                "Data reconciliation processes".to_string(),
                "Data validation at source".to_string(),
                "Automated controls to minimize manual intervention".to_string(),
                "Independent data quality verification".to_string(),
            ],
            validation_criteria: vec![
                "reconciliation".to_string(),
                "validation".to_string(),
                "quality metrics".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
        }
    }

    /// Principle 4: Completeness
    pub fn completeness() -> BCBS239Requirement {
        BCBS239Requirement {
            id: "BCBS239-P4".to_string(),
            principle: BCBS239Principle::Completeness,
            description: "Capture and aggregate all material risk data across the organization"
                .to_string(),
            key_attributes: vec![
                "All material risk exposures included".to_string(),
                "Group-wide coverage".to_string(),
                "Cross-border and cross-entity aggregation".to_string(),
                "Coverage of off-balance sheet exposures".to_string(),
            ],
            validation_criteria: vec![
                "coverage analysis".to_string(),
                "materiality assessment".to_string(),
                "gap analysis".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    /// Principle 5: Timeliness
    pub fn timeliness() -> BCBS239Requirement {
        BCBS239Requirement {
            id: "BCBS239-P5".to_string(),
            principle: BCBS239Principle::Timeliness,
            description: "Generate risk data in a timely manner for routine and crisis scenarios"
                .to_string(),
            key_attributes: vec![
                "Defined timelines for normal operations".to_string(),
                "Accelerated reporting capability for crisis".to_string(),
                "Real-time or near-real-time aggregation".to_string(),
                "Automated processes to meet deadlines".to_string(),
            ],
            validation_criteria: vec![
                "SLA compliance".to_string(),
                "crisis simulation".to_string(),
                "reporting timeliness metrics".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    /// Principle 6: Adaptability
    pub fn adaptability() -> BCBS239Requirement {
        BCBS239Requirement {
            id: "BCBS239-P6".to_string(),
            principle: BCBS239Principle::Adaptability,
            description: "Generate aggregate and detailed reports for emerging risks".to_string(),
            key_attributes: vec![
                "Flexible data architecture".to_string(),
                "Ad-hoc reporting capabilities".to_string(),
                "Stress testing and scenario analysis support".to_string(),
                "Drill-down capabilities".to_string(),
            ],
            validation_criteria: vec![
                "ad-hoc report examples".to_string(),
                "stress test support".to_string(),
                "architecture flexibility".to_string(),
            ],
            risk_level: RiskLevel::Medium,
        }
    }

    /// Principle 2: Data Architecture and IT Infrastructure
    pub fn data_architecture() -> BCBS239Requirement {
        BCBS239Requirement {
            id: "BCBS239-P2".to_string(),
            principle: BCBS239Principle::DataArchitecture,
            description: "Design and maintain robust data architecture and IT infrastructure"
                .to_string(),
            key_attributes: vec![
                "Enterprise-wide data dictionary".to_string(),
                "Consistent data definitions".to_string(),
                "Data lineage documentation".to_string(),
                "Scalable and resilient infrastructure".to_string(),
            ],
            validation_criteria: vec![
                "data dictionary".to_string(),
                "lineage documentation".to_string(),
                "architecture diagrams".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
        }
    }

    pub fn all_requirements() -> Vec<BCBS239Requirement> {
        vec![
            Self::data_architecture(),
            Self::accuracy_integrity(),
            Self::completeness(),
            Self::timeliness(),
            Self::adaptability(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_lineage_creation() {
        let mut lineage = DataLineage::new(
            "Total Exposure".to_string(),
            "Trade System".to_string(),
            "Risk Warehouse".to_string(),
        );

        let output = b"transformed_data";
        lineage.add_transformation(
            "Currency conversion".to_string(),
            "USD conversion at spot rate".to_string(),
            vec!["Rate must be from official source".to_string()],
            output,
        );

        assert_eq!(lineage.transformation_steps.len(), 1);
        assert!(lineage.verify_transformation(0, output));
        assert!(!lineage.verify_transformation(0, b"wrong_data"));
    }

    #[test]
    fn test_bcbs239_principle_numbering() {
        assert_eq!(BCBS239Principle::Governance.number(), 1);
        assert_eq!(BCBS239Principle::AccuracyIntegrity.number(), 3);
        assert_eq!(BCBS239Principle::CompliancePrinciple.number(), 14);
    }

    #[test]
    fn test_risk_data_aggregation_scoring() {
        let assessment = RiskDataAggregation {
            assessment_id: Uuid::new_v4(),
            institution: "Test Bank".to_string(),
            assessment_date: Utc::now(),
            accuracy_score: 0.95,
            completeness_score: 0.90,
            timeliness_score: 0.85,
            adaptability_score: 0.88,
            data_lineages: vec![],
        };

        let overall = assessment.overall_score();
        assert!((overall - 0.895).abs() < 0.001);
        assert!(assessment.is_compliant(0.85));
        assert!(!assessment.is_compliant(0.95));
    }
}
