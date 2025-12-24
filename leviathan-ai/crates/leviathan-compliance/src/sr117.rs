//! SR 11-7 Model Risk Management Compliance
//!
//! Federal Reserve SR 11-7: Guidance on Model Risk Management (April 2011)
//! Supervisory guidance for effective validation and governance of models.

use crate::{
    ComplianceError, ComplianceFramework, ComplianceRequirement, ControlEffectiveness, Evidence,
    Result, RiskLevel, Severity, ValidationResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Model types covered by SR 11-7
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Credit risk models (PD, LGD, EAD)
    CreditRisk,
    /// Market risk models (VaR, stress testing)
    MarketRisk,
    /// Operational risk models
    OperationalRisk,
    /// Liquidity risk models
    LiquidityRisk,
    /// Capital planning models (CCAR, DFAST)
    CapitalPlanning,
    /// Asset-liability management
    ALM,
    /// Fair value measurement
    FairValue,
    /// Anti-money laundering
    AML,
    /// Fraud detection
    FraudDetection,
    /// Machine learning / AI models
    MachineLearning,
    /// Other quantitative model
    Other(String),
}

/// Model risk rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRiskRating {
    Low,
    Moderate,
    High,
}

/// Model validation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Initial validation complete
    Validated,
    /// Validation in progress
    InProgress,
    /// Validation needed
    Required,
    /// Validation overdue
    Overdue,
    /// Model needs revalidation
    RevalidationRequired,
}

/// SR 11-7 Model Inventory Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInventoryEntry {
    pub model_id: String,
    pub model_name: String,
    pub model_type: ModelType,
    pub model_owner: String,
    pub business_unit: String,
    pub description: String,
    pub risk_rating: ModelRiskRating,
    pub validation_status: ValidationStatus,
    pub last_validation_date: Option<DateTime<Utc>>,
    pub next_validation_due: DateTime<Utc>,
    pub validation_frequency_months: u32,
    pub material_changes_since_validation: Vec<String>,
    pub known_limitations: Vec<String>,
    pub compensating_controls: Vec<String>,
}

impl ModelInventoryEntry {
    pub fn is_validation_current(&self) -> bool {
        Utc::now() < self.next_validation_due
    }

    pub fn days_until_validation(&self) -> i64 {
        (self.next_validation_due - Utc::now()).num_days()
    }

    pub fn requires_revalidation(&self) -> bool {
        !self.material_changes_since_validation.is_empty() || !self.is_validation_current()
    }
}

/// Model validation report per SR 11-7
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidation {
    pub validation_id: Uuid,
    pub model_id: String,
    pub validation_date: DateTime<Utc>,
    pub validator: String,
    pub validator_independent: bool, // Must be independent per SR 11-7
    pub validation_scope: ValidationScope,
    pub findings: Vec<ValidationFinding>,
    pub overall_assessment: ValidationAssessment,
    pub recommendations: Vec<String>,
    pub approved_for_use: bool,
    pub conditions_for_use: Vec<String>,
    pub next_validation_due: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationScope {
    pub conceptual_soundness: bool,
    pub ongoing_monitoring: bool,
    pub outcomes_analysis: bool,
    pub data_quality: bool,
    pub assumptions_review: bool,
    pub sensitivity_analysis: bool,
    pub back_testing: bool,
    pub benchmarking: bool,
}

impl Default for ValidationScope {
    fn default() -> Self {
        Self {
            conceptual_soundness: true,
            ongoing_monitoring: true,
            outcomes_analysis: true,
            data_quality: true,
            assumptions_review: true,
            sensitivity_analysis: true,
            back_testing: false,
            benchmarking: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    pub finding_id: Uuid,
    pub severity: Severity,
    pub category: FindingCategory,
    pub description: String,
    pub impact: String,
    pub recommendation: String,
    pub management_response: Option<String>,
    pub target_remediation_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingCategory {
    ConceptualSoundness,
    DataQuality,
    Implementation,
    Documentation,
    Governance,
    Monitoring,
    Assumptions,
    Limitations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationAssessment {
    pub conceptual_soundness_rating: ControlEffectiveness,
    pub implementation_rating: ControlEffectiveness,
    pub monitoring_rating: ControlEffectiveness,
    pub overall_rating: ControlEffectiveness,
    pub material_issues: Vec<String>,
}

/// Model performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMonitoring {
    pub model_id: String,
    pub monitoring_period: (DateTime<Utc>, DateTime<Utc>),
    pub metrics: HashMap<String, f64>,
    pub thresholds: HashMap<String, (f64, f64)>, // (warning, critical)
    pub breaches: Vec<ThresholdBreach>,
    pub back_testing_results: Option<BackTestingResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdBreach {
    pub metric_name: String,
    pub actual_value: f64,
    pub threshold_value: f64,
    pub breach_severity: Severity,
    pub detected_at: DateTime<Utc>,
    pub investigation_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackTestingResult {
    pub test_period: (DateTime<Utc>, DateTime<Utc>),
    pub predictions: usize,
    pub accurate_predictions: usize,
    pub accuracy_rate: f64,
    pub acceptable_threshold: f64,
    pub passed: bool,
}

/// SR 11-7 Compliance Requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SR117Requirement {
    pub id: String,
    pub component: SR117Component,
    pub description: String,
    pub key_practices: Vec<String>,
    pub validation_criteria: Vec<String>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SR117Component {
    /// Model development and implementation
    ModelDevelopment,
    /// Model validation
    ModelValidation,
    /// Governance and oversight
    Governance,
    /// Policies and procedures
    PoliciesProcedures,
    /// Model inventory
    ModelInventory,
    /// Ongoing monitoring
    OngoingMonitoring,
}

impl ComplianceRequirement for SR117Requirement {
    fn requirement_id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn framework(&self) -> ComplianceFramework {
        ComplianceFramework::SR117
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
                findings.push(format!("Missing evidence for: {}", criterion));
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

/// Pre-defined SR 11-7 requirements
pub struct SR117Requirements;

impl SR117Requirements {
    pub fn model_inventory() -> SR117Requirement {
        SR117Requirement {
            id: "SR117-INV-001".to_string(),
            component: SR117Component::ModelInventory,
            description: "Maintain comprehensive model inventory with risk ratings".to_string(),
            key_practices: vec![
                "All models cataloged".to_string(),
                "Risk rating assigned".to_string(),
                "Validation status tracked".to_string(),
                "Material changes documented".to_string(),
            ],
            validation_criteria: vec![
                "inventory documentation".to_string(),
                "risk ratings".to_string(),
                "validation schedule".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    pub fn independent_validation() -> SR117Requirement {
        SR117Requirement {
            id: "SR117-VAL-001".to_string(),
            component: SR117Component::ModelValidation,
            description: "Ensure effective, independent model validation".to_string(),
            key_practices: vec![
                "Independent validator (not model developer)".to_string(),
                "Conceptual soundness review".to_string(),
                "Ongoing monitoring review".to_string(),
                "Outcomes analysis".to_string(),
            ],
            validation_criteria: vec![
                "validation independence".to_string(),
                "validation report".to_string(),
                "outcomes analysis".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
        }
    }

    pub fn ongoing_monitoring() -> SR117Requirement {
        SR117Requirement {
            id: "SR117-MON-001".to_string(),
            component: SR117Component::OngoingMonitoring,
            description: "Implement ongoing model performance monitoring".to_string(),
            key_practices: vec![
                "Key metrics tracked".to_string(),
                "Thresholds defined".to_string(),
                "Back-testing performed".to_string(),
                "Breach escalation process".to_string(),
            ],
            validation_criteria: vec![
                "monitoring reports".to_string(),
                "back-testing results".to_string(),
                "escalation procedures".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    pub fn model_governance() -> SR117Requirement {
        SR117Requirement {
            id: "SR117-GOV-001".to_string(),
            component: SR117Component::Governance,
            description: "Establish strong model governance framework".to_string(),
            key_practices: vec![
                "Board and senior management oversight".to_string(),
                "Model risk management policy".to_string(),
                "Roles and responsibilities defined".to_string(),
                "Model approval process".to_string(),
            ],
            validation_criteria: vec![
                "governance policy".to_string(),
                "board reporting".to_string(),
                "approval documentation".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
        }
    }

    pub fn all_requirements() -> Vec<SR117Requirement> {
        vec![
            Self::model_inventory(),
            Self::independent_validation(),
            Self::ongoing_monitoring(),
            Self::model_governance(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_inventory_validation_status() {
        let mut entry = ModelInventoryEntry {
            model_id: "MDL-001".to_string(),
            model_name: "Credit Risk PD Model".to_string(),
            model_type: ModelType::CreditRisk,
            model_owner: "John Doe".to_string(),
            business_unit: "Credit Risk".to_string(),
            description: "Probability of Default model".to_string(),
            risk_rating: ModelRiskRating::High,
            validation_status: ValidationStatus::Validated,
            last_validation_date: Some(Utc::now()),
            next_validation_due: Utc::now() + chrono::Duration::days(365),
            validation_frequency_months: 12,
            material_changes_since_validation: vec![],
            known_limitations: vec![],
            compensating_controls: vec![],
        };

        assert!(entry.is_validation_current());
        assert!(!entry.requires_revalidation());

        entry.material_changes_since_validation.push("New data source".to_string());
        assert!(entry.requires_revalidation());
    }

    #[test]
    fn test_back_testing() {
        let result = BackTestingResult {
            test_period: (Utc::now() - chrono::Duration::days(365), Utc::now()),
            predictions: 1000,
            accurate_predictions: 920,
            accuracy_rate: 0.92,
            acceptable_threshold: 0.85,
            passed: true,
        };

        assert_eq!(result.accuracy_rate, 0.92);
        assert!(result.passed);
    }
}
