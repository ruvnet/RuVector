//! GDPR (General Data Protection Regulation) Compliance Helpers
//!
//! EU GDPR (Regulation 2016/679) compliance utilities for data protection.

use crate::{
    ComplianceError, ComplianceFramework, ComplianceRequirement, Evidence, Result, RiskLevel,
    Severity, ValidationResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// GDPR Legal basis for processing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalBasis {
    /// Article 6(1)(a) - Consent
    Consent,
    /// Article 6(1)(b) - Contract
    Contract,
    /// Article 6(1)(c) - Legal obligation
    LegalObligation,
    /// Article 6(1)(d) - Vital interests
    VitalInterests,
    /// Article 6(1)(e) - Public task
    PublicTask,
    /// Article 6(1)(f) - Legitimate interests
    LegitimateInterests,
}

/// GDPR Data subject rights (Chapter III)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSubjectRight {
    /// Article 15 - Right of access
    Access,
    /// Article 16 - Right to rectification
    Rectification,
    /// Article 17 - Right to erasure ("right to be forgotten")
    Erasure,
    /// Article 18 - Right to restriction of processing
    RestrictionOfProcessing,
    /// Article 20 - Right to data portability
    DataPortability,
    /// Article 21 - Right to object
    Object,
    /// Article 22 - Automated individual decision-making
    AutomatedDecisionMaking,
}

/// Data subject identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubject {
    pub subject_id: Uuid,
    pub subject_type: SubjectType,
    pub identifiers: HashMap<String, String>, // email, customer_id, etc.
    pub created_at: DateTime<Utc>,
    pub consent_records: Vec<ConsentRecord>,
    pub processing_activities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubjectType {
    Customer,
    Employee,
    Contractor,
    Visitor,
    Other(String),
}

/// Consent tracking (Article 7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub consent_id: Uuid,
    pub purpose: String,
    pub legal_basis: LegalBasis,
    pub given_at: DateTime<Utc>,
    pub consent_method: String, // "web form", "API", "paper", etc.
    pub withdrawn_at: Option<DateTime<Utc>>,
    pub version: String,
    pub text_presented: String,
    pub freely_given: bool,
    pub specific: bool,
    pub informed: bool,
    pub unambiguous: bool,
}

impl ConsentRecord {
    pub fn is_valid_consent(&self) -> bool {
        self.freely_given
            && self.specific
            && self.informed
            && self.unambiguous
            && self.withdrawn_at.is_none()
    }

    pub fn is_active(&self) -> bool {
        self.withdrawn_at.is_none()
    }
}

/// Processing activity (Article 30 - Records of processing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    pub activity_id: Uuid,
    pub name: String,
    pub controller: String,
    pub dpo_contact: Option<String>, // Data Protection Officer
    pub purposes: Vec<String>,
    pub legal_basis: LegalBasis,
    pub data_categories: Vec<DataCategory>,
    pub data_subjects: Vec<SubjectType>,
    pub recipients: Vec<String>,
    pub third_country_transfers: Vec<ThirdCountryTransfer>,
    pub retention_period: RetentionPeriod,
    pub technical_measures: Vec<String>,
    pub organizational_measures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataCategory {
    PersonalIdentifiers,
    ContactDetails,
    FinancialData,
    EmploymentData,
    SensitiveData(SensitiveDataType),
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitiveDataType {
    /// Article 9 - Special categories
    RacialEthnicOrigin,
    PoliticalOpinions,
    ReligiousBeliefs,
    TradeUnionMembership,
    GeneticData,
    BiometricData,
    HealthData,
    SexOrientation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdCountryTransfer {
    pub country: String,
    pub safeguard: TransferSafeguard,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferSafeguard {
    AdequacyDecision,
    StandardContractualClauses,
    BindingCorporateRules,
    ApprovedCodeOfConduct,
    ApprovedCertification,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPeriod {
    Days(u32),
    Months(u32),
    Years(u32),
    UntilPurposeFulfilled,
    LegalRequirement(String),
}

/// Data subject request (DSR) handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    pub request_id: Uuid,
    pub subject_id: Uuid,
    pub request_type: DataSubjectRight,
    pub received_at: DateTime<Utc>,
    pub verified_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub response_due: DateTime<Utc>, // Must respond within 1 month (Article 12)
    pub status: RequestStatus,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    Received,
    VerificationPending,
    InProgress,
    Completed,
    Rejected { reason: String },
    Extended { new_due_date: DateTime<Utc> }, // Can extend by 2 months if complex
}

impl DataSubjectRequest {
    pub fn new(subject_id: Uuid, request_type: DataSubjectRight) -> Self {
        let now = Utc::now();
        let due = now + chrono::Duration::days(30); // 1 month requirement

        Self {
            request_id: Uuid::new_v4(),
            subject_id,
            request_type,
            received_at: now,
            verified_at: None,
            completed_at: None,
            response_due: due,
            status: RequestStatus::Received,
            notes: Vec::new(),
        }
    }

    pub fn is_overdue(&self) -> bool {
        Utc::now() > self.response_due && !matches!(self.status, RequestStatus::Completed)
    }

    pub fn days_until_due(&self) -> i64 {
        (self.response_due - Utc::now()).num_days()
    }
}

/// Data portability export (Article 20)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPortabilityExport {
    pub export_id: Uuid,
    pub subject_id: Uuid,
    pub exported_at: DateTime<Utc>,
    pub format: ExportFormat,
    pub data: serde_json::Value,
    pub checksum: [u8; 32], // Blake3 hash
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    XML,
    Other(String),
}

impl DataPortabilityExport {
    pub fn new(subject_id: Uuid, data: serde_json::Value, format: ExportFormat) -> Self {
        let data_bytes = serde_json::to_vec(&data).unwrap_or_default();
        let checksum: [u8; 32] = blake3::hash(&data_bytes).into();

        Self {
            export_id: Uuid::new_v4(),
            subject_id,
            exported_at: Utc::now(),
            format,
            data,
            checksum,
        }
    }

    pub fn verify_integrity(&self) -> bool {
        let data_bytes = serde_json::to_vec(&self.data).unwrap_or_default();
        let computed: [u8; 32] = blake3::hash(&data_bytes).into();
        computed == self.checksum
    }
}

/// GDPR Compliance Requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GDPRRequirement {
    pub id: String,
    pub article: String,
    pub description: String,
    pub obligations: Vec<String>,
    pub validation_criteria: Vec<String>,
    pub risk_level: RiskLevel,
}

impl ComplianceRequirement for GDPRRequirement {
    fn requirement_id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn framework(&self) -> ComplianceFramework {
        ComplianceFramework::GDPR
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

pub struct GDPRRequirements;

impl GDPRRequirements {
    pub fn lawful_processing() -> GDPRRequirement {
        GDPRRequirement {
            id: "GDPR-ART6".to_string(),
            article: "Article 6".to_string(),
            description: "Lawfulness of processing - Legal basis required".to_string(),
            obligations: vec![
                "Identify legal basis for each processing activity".to_string(),
                "Document legal basis".to_string(),
                "Review and update as needed".to_string(),
            ],
            validation_criteria: vec![
                "legal basis documentation".to_string(),
                "processing inventory".to_string(),
            ],
            risk_level: RiskLevel::VeryHigh,
        }
    }

    pub fn data_subject_rights() -> GDPRRequirement {
        GDPRRequirement {
            id: "GDPR-ART12-22".to_string(),
            article: "Articles 12-22".to_string(),
            description: "Facilitate data subject rights exercise".to_string(),
            obligations: vec![
                "Respond within 1 month".to_string(),
                "Verify identity".to_string(),
                "Provide data in portable format if requested".to_string(),
                "No charge for first request".to_string(),
            ],
            validation_criteria: vec![
                "DSR process".to_string(),
                "response times".to_string(),
                "verification procedure".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    pub fn data_protection_by_design() -> GDPRRequirement {
        GDPRRequirement {
            id: "GDPR-ART25".to_string(),
            article: "Article 25".to_string(),
            description: "Data protection by design and by default".to_string(),
            obligations: vec![
                "Implement appropriate technical measures".to_string(),
                "Implement organizational measures".to_string(),
                "Minimize data collection".to_string(),
                "Pseudonymization where appropriate".to_string(),
            ],
            validation_criteria: vec![
                "privacy by design".to_string(),
                "data minimization".to_string(),
                "technical measures".to_string(),
            ],
            risk_level: RiskLevel::High,
        }
    }

    pub fn all_requirements() -> Vec<GDPRRequirement> {
        vec![
            Self::lawful_processing(),
            Self::data_subject_rights(),
            Self::data_protection_by_design(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_validity() {
        let valid_consent = ConsentRecord {
            consent_id: Uuid::new_v4(),
            purpose: "Marketing emails".to_string(),
            legal_basis: LegalBasis::Consent,
            given_at: Utc::now(),
            consent_method: "web form".to_string(),
            withdrawn_at: None,
            version: "1.0".to_string(),
            text_presented: "I agree to receive marketing emails".to_string(),
            freely_given: true,
            specific: true,
            informed: true,
            unambiguous: true,
        };

        assert!(valid_consent.is_valid_consent());
        assert!(valid_consent.is_active());
    }

    #[test]
    fn test_dsr_timing() {
        let mut request = DataSubjectRequest::new(Uuid::new_v4(), DataSubjectRight::Access);

        assert!(!request.is_overdue());
        assert!(request.days_until_due() > 0);

        // Simulate overdue
        request.response_due = Utc::now() - chrono::Duration::days(1);
        assert!(request.is_overdue());
    }

    #[test]
    fn test_data_portability_export() {
        let data = serde_json::json!({
            "name": "John Doe",
            "email": "john@example.com",
            "accounts": ["ACC001", "ACC002"]
        });

        let export = DataPortabilityExport::new(Uuid::new_v4(), data, ExportFormat::JSON);

        assert!(export.verify_integrity());
    }
}
