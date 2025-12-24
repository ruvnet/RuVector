//! Compliance Audit Reporting
//!
//! Generate cryptographically signed compliance reports with evidence trails.

use crate::{ComplianceError, ComplianceFramework, Evidence, Result, Severity};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Compliance status for overall assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Fully compliant with all requirements
    Compliant,
    /// Partially compliant with documented gaps
    PartiallyCompliant { gaps: Vec<String> },
    /// Not compliant with documented violations
    NonCompliant { violations: Vec<String> },
    /// Assessment not yet performed
    NotAssessed,
}

/// Individual finding in compliance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub finding_id: Uuid,
    pub requirement_id: String,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub impact: String,
    pub recommendation: String,
    pub evidence_ids: Vec<Uuid>,
    pub discovered_at: DateTime<Utc>,
}

impl Finding {
    pub fn new(
        requirement_id: String,
        severity: Severity,
        title: String,
        description: String,
    ) -> Self {
        Self {
            finding_id: Uuid::new_v4(),
            requirement_id,
            severity,
            title,
            description,
            impact: String::new(),
            recommendation: String::new(),
            evidence_ids: Vec::new(),
            discovered_at: Utc::now(),
        }
    }

    pub fn with_impact(mut self, impact: String) -> Self {
        self.impact = impact;
        self
    }

    pub fn with_recommendation(mut self, recommendation: String) -> Self {
        self.recommendation = recommendation;
        self
    }

    pub fn with_evidence(mut self, evidence_id: Uuid) -> Self {
        self.evidence_ids.push(evidence_id);
        self
    }
}

/// Comprehensive compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub report_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub framework: ComplianceFramework,
    pub institution: String,
    pub assessment_period: (DateTime<Utc>, DateTime<Utc>),
    pub assessor: String,
    pub status: ComplianceStatus,
    pub findings: Vec<Finding>,
    pub evidence: Vec<Evidence>,
    pub metrics: ComplianceMetrics,
    pub executive_summary: String,
    pub recommendations: Vec<String>,
    pub signature: Option<Vec<u8>>, // Ed25519 signature
    pub signature_public_key: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetrics {
    pub total_requirements: usize,
    pub compliant_requirements: usize,
    pub partially_compliant: usize,
    pub non_compliant: usize,
    pub not_assessed: usize,
    pub compliance_score: f64, // 0.0 to 1.0
    pub critical_findings: usize,
    pub high_findings: usize,
    pub medium_findings: usize,
    pub low_findings: usize,
}

impl ComplianceMetrics {
    pub fn new() -> Self {
        Self {
            total_requirements: 0,
            compliant_requirements: 0,
            partially_compliant: 0,
            non_compliant: 0,
            not_assessed: 0,
            compliance_score: 0.0,
            critical_findings: 0,
            high_findings: 0,
            medium_findings: 0,
            low_findings: 0,
        }
    }

    pub fn calculate_score(&mut self) {
        if self.total_requirements == 0 {
            self.compliance_score = 0.0;
            return;
        }

        let weighted_score = (self.compliant_requirements as f64 * 1.0)
            + (self.partially_compliant as f64 * 0.5);

        self.compliance_score = weighted_score / self.total_requirements as f64;
    }

    pub fn add_finding(&mut self, severity: Severity) {
        match severity {
            Severity::Critical => self.critical_findings += 1,
            Severity::High => self.high_findings += 1,
            Severity::Medium => self.medium_findings += 1,
            Severity::Low | Severity::Info => self.low_findings += 1,
        }
    }
}

impl Default for ComplianceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplianceReport {
    pub fn new(
        framework: ComplianceFramework,
        institution: String,
        assessment_period: (DateTime<Utc>, DateTime<Utc>),
        assessor: String,
    ) -> Self {
        Self {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            framework,
            institution,
            assessment_period,
            assessor,
            status: ComplianceStatus::NotAssessed,
            findings: Vec::new(),
            evidence: Vec::new(),
            metrics: ComplianceMetrics::new(),
            executive_summary: String::new(),
            recommendations: Vec::new(),
            signature: None,
            signature_public_key: None,
        }
    }

    pub fn add_finding(&mut self, finding: Finding) {
        self.metrics.add_finding(finding.severity);
        self.findings.push(finding);
    }

    pub fn add_evidence(&mut self, evidence: Evidence) {
        self.evidence.push(evidence);
    }

    pub fn calculate_status(&mut self) {
        self.metrics.calculate_score();

        if self.metrics.critical_findings > 0 || self.metrics.non_compliant > 0 {
            let violations: Vec<String> = self
                .findings
                .iter()
                .filter(|f| matches!(f.severity, Severity::Critical | Severity::High))
                .map(|f| f.title.clone())
                .collect();

            self.status = ComplianceStatus::NonCompliant { violations };
        } else if self.metrics.partially_compliant > 0 || self.metrics.high_findings > 0 {
            let gaps: Vec<String> = self
                .findings
                .iter()
                .map(|f| f.requirement_id.clone())
                .collect();

            self.status = ComplianceStatus::PartiallyCompliant { gaps };
        } else {
            self.status = ComplianceStatus::Compliant;
        }
    }

    /// Sign the report with Ed25519 private key
    pub fn sign(&mut self, signing_key: &SigningKey) -> Result<()> {
        let report_hash = self.compute_hash()?;
        let signature = signing_key.sign(&report_hash);

        self.signature = Some(signature.to_bytes().to_vec());
        self.signature_public_key = Some(signing_key.verifying_key().to_bytes().to_vec());

        Ok(())
    }

    /// Verify report signature
    pub fn verify_signature(&self) -> Result<bool> {
        let signature_bytes = self
            .signature
            .as_ref()
            .ok_or_else(|| ComplianceError::AuditTrailVerificationFailed("No signature".into()))?;

        let public_key_bytes = self.signature_public_key.as_ref().ok_or_else(|| {
            ComplianceError::AuditTrailVerificationFailed("No public key".into())
        })?;

        let signature = Signature::from_bytes(
            signature_bytes
                .as_slice()
                .try_into()
                .map_err(|_| ComplianceError::AuditTrailVerificationFailed("Invalid signature format".into()))?,
        );

        let public_key = VerifyingKey::from_bytes(
            public_key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| ComplianceError::AuditTrailVerificationFailed("Invalid public key format".into()))?,
        )
        .map_err(|e| ComplianceError::AuditTrailVerificationFailed(e.to_string()))?;

        let report_hash = self.compute_hash()?;

        Ok(public_key.verify(&report_hash, &signature).is_ok())
    }

    fn compute_hash(&self) -> Result<Vec<u8>> {
        // Create a temporary report without signature for hashing
        let mut temp_report = self.clone();
        temp_report.signature = None;
        temp_report.signature_public_key = None;

        let report_json = serde_json::to_vec(&temp_report)?;
        let hash = blake3::hash(&report_json);

        Ok(hash.as_bytes().to_vec())
    }

    /// Export report to JSON
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Export report to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Generate summary statistics
    pub fn summary(&self) -> ReportSummary {
        ReportSummary {
            report_id: self.report_id,
            framework: self.framework,
            status: self.status.clone(),
            compliance_score: self.metrics.compliance_score,
            total_findings: self.findings.len(),
            critical_findings: self.metrics.critical_findings,
            evidence_count: self.evidence.len(),
            is_signed: self.signature.is_some(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub report_id: Uuid,
    pub framework: ComplianceFramework,
    pub status: ComplianceStatus,
    pub compliance_score: f64,
    pub total_findings: usize,
    pub critical_findings: usize,
    pub evidence_count: usize,
    pub is_signed: bool,
}

/// Report builder for easier construction
pub struct ComplianceReportBuilder {
    report: ComplianceReport,
}

impl ComplianceReportBuilder {
    pub fn new(
        framework: ComplianceFramework,
        institution: String,
        assessor: String,
    ) -> Self {
        let now = Utc::now();
        let period = (now - chrono::Duration::days(90), now);

        Self {
            report: ComplianceReport::new(framework, institution, period, assessor),
        }
    }

    pub fn assessment_period(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.report.assessment_period = (start, end);
        self
    }

    pub fn add_finding(mut self, finding: Finding) -> Self {
        self.report.add_finding(finding);
        self
    }

    pub fn add_evidence(mut self, evidence: Evidence) -> Self {
        self.report.add_evidence(evidence);
        self
    }

    pub fn executive_summary(mut self, summary: String) -> Self {
        self.report.executive_summary = summary;
        self
    }

    pub fn add_recommendation(mut self, recommendation: String) -> Self {
        self.report.recommendations.push(recommendation);
        self
    }

    pub fn metrics(mut self, metrics: ComplianceMetrics) -> Self {
        self.report.metrics = metrics;
        self
    }

    pub fn build(mut self) -> ComplianceReport {
        self.report.calculate_status();
        self.report
    }

    pub fn build_and_sign(mut self, signing_key: &SigningKey) -> Result<ComplianceReport> {
        self.report.calculate_status();
        self.report.sign(signing_key)?;
        Ok(self.report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvidenceType, Severity};

    #[test]
    fn test_compliance_metrics() {
        let mut metrics = ComplianceMetrics::new();
        metrics.total_requirements = 10;
        metrics.compliant_requirements = 7;
        metrics.partially_compliant = 2;
        metrics.non_compliant = 1;

        metrics.calculate_score();

        // Score = (7*1.0 + 2*0.5) / 10 = 0.8
        assert!((metrics.compliance_score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_report_signing() {
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);

        let mut report = ComplianceReport::new(
            ComplianceFramework::FFIEC,
            "Test Bank".to_string(),
            (Utc::now() - chrono::Duration::days(30), Utc::now()),
            "Auditor Name".to_string(),
        );

        report.sign(&signing_key).unwrap();
        assert!(report.signature.is_some());
        assert!(report.verify_signature().unwrap());
    }

    #[test]
    fn test_report_builder() {
        let finding = Finding::new(
            "TEST-001".to_string(),
            Severity::High,
            "Test Finding".to_string(),
            "This is a test".to_string(),
        );

        let evidence = Evidence::new(
            EvidenceType::Documentation,
            "Test evidence".to_string(),
            b"test data",
            "tester".to_string(),
        );

        let report = ComplianceReportBuilder::new(
            ComplianceFramework::BCBS239,
            "Test Institution".to_string(),
            "Test Assessor".to_string(),
        )
        .add_finding(finding)
        .add_evidence(evidence)
        .executive_summary("Test summary".to_string())
        .add_recommendation("Fix the issues".to_string())
        .build();

        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.evidence.len(), 1);
        assert_eq!(report.recommendations.len(), 1);
    }
}
