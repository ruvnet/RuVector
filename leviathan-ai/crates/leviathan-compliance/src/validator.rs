//! Automated Compliance Validation
//!
//! Validates compliance across all frameworks with evidence collection and gap analysis.

use crate::{
    audit_report::{ComplianceReport, ComplianceReportBuilder, ComplianceStatus, Finding},
    bcbs239::{BCBS239Requirements, DataLineage},
    ffiec::FFIECRequirements,
    gdpr::GDPRRequirements,
    sr117::SR117Requirements,
    ComplianceError, ComplianceFramework, ComplianceRequirement, Evidence, EvidenceType, Result,
    Severity, ValidationResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Automated compliance validator
pub struct ComplianceValidator {
    evidence_store: HashMap<String, Vec<Evidence>>,
    validation_cache: HashMap<String, ValidationResult>,
}

impl ComplianceValidator {
    pub fn new() -> Self {
        Self {
            evidence_store: HashMap::new(),
            validation_cache: HashMap::new(),
        }
    }

    /// Add evidence for a requirement
    pub fn add_evidence(&mut self, requirement_id: String, evidence: Evidence) {
        self.evidence_store
            .entry(requirement_id)
            .or_insert_with(Vec::new)
            .push(evidence);
    }

    /// Validate a single requirement
    pub fn validate_requirement(
        &mut self,
        requirement: &dyn ComplianceRequirement,
    ) -> Result<ValidationResult> {
        let evidence = self
            .evidence_store
            .get(requirement.requirement_id())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let result = requirement.validate(evidence)?;

        // Cache the result
        self.validation_cache
            .insert(requirement.requirement_id().to_string(), result.clone());

        Ok(result)
    }

    /// Validate entire framework
    pub fn validate_framework(&mut self, framework: ComplianceFramework) -> Result<FrameworkValidationResult> {
        let requirements: Vec<Box<dyn ComplianceRequirement>> = match framework {
            ComplianceFramework::FFIEC => FFIECRequirements::all_requirements()
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn ComplianceRequirement>)
                .collect(),
            ComplianceFramework::BCBS239 => BCBS239Requirements::all_requirements()
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn ComplianceRequirement>)
                .collect(),
            ComplianceFramework::SR117 => SR117Requirements::all_requirements()
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn ComplianceRequirement>)
                .collect(),
            ComplianceFramework::GDPR => GDPRRequirements::all_requirements()
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn ComplianceRequirement>)
                .collect(),
            _ => {
                return Err(ComplianceError::UnsupportedFramework(format!(
                    "{:?}",
                    framework
                )))
            }
        };

        let mut results = Vec::new();
        let mut compliant_count = 0;
        let mut total_findings = 0;

        for requirement in &requirements {
            let result = self.validate_requirement(requirement.as_ref())?;

            if result.compliant {
                compliant_count += 1;
            } else {
                total_findings += result.findings.len();
            }

            results.push(result);
        }

        Ok(FrameworkValidationResult {
            framework,
            validated_at: Utc::now(),
            total_requirements: requirements.len(),
            compliant_requirements: compliant_count,
            validation_results: results,
            compliance_score: compliant_count as f64 / requirements.len() as f64,
        })
    }

    /// Generate comprehensive compliance report
    pub fn generate_report(
        &mut self,
        framework: ComplianceFramework,
        institution: String,
        assessor: String,
    ) -> Result<ComplianceReport> {
        let validation_result = self.validate_framework(framework)?;

        let mut builder = ComplianceReportBuilder::new(framework, institution, assessor);

        // Add findings from validation
        for result in &validation_result.validation_results {
            if !result.compliant {
                for finding_desc in &result.findings {
                    let finding = Finding::new(
                        result.requirement_id.clone(),
                        result.severity,
                        format!("Non-compliance: {}", result.requirement_id),
                        finding_desc.clone(),
                    );
                    builder = builder.add_finding(finding);
                }
            }
        }

        // Add all evidence
        for (_, evidence_list) in &self.evidence_store {
            for evidence in evidence_list {
                builder = builder.add_evidence(evidence.clone());
            }
        }

        // Generate executive summary
        let summary = format!(
            "Compliance assessment for {:?} framework. {} of {} requirements met ({:.1}% compliance). {} findings identified.",
            framework,
            validation_result.compliant_requirements,
            validation_result.total_requirements,
            validation_result.compliance_score * 100.0,
            validation_result.total_requirements - validation_result.compliant_requirements
        );

        builder = builder.executive_summary(summary);

        // Set metrics
        let mut metrics = crate::audit_report::ComplianceMetrics::new();
        metrics.total_requirements = validation_result.total_requirements;
        metrics.compliant_requirements = validation_result.compliant_requirements;
        metrics.non_compliant =
            validation_result.total_requirements - validation_result.compliant_requirements;

        for result in &validation_result.validation_results {
            if !result.compliant {
                metrics.add_finding(result.severity);
            }
        }

        metrics.calculate_score();
        builder = builder.metrics(metrics);

        Ok(builder.build())
    }

    /// Perform gap analysis
    pub fn gap_analysis(&self, framework: ComplianceFramework) -> Result<GapAnalysis> {
        let requirements: Vec<String> = match framework {
            ComplianceFramework::FFIEC => FFIECRequirements::all_requirements()
                .into_iter()
                .map(|r| r.id)
                .collect(),
            ComplianceFramework::BCBS239 => BCBS239Requirements::all_requirements()
                .into_iter()
                .map(|r| r.id)
                .collect(),
            ComplianceFramework::SR117 => SR117Requirements::all_requirements()
                .into_iter()
                .map(|r| r.id)
                .collect(),
            ComplianceFramework::GDPR => GDPRRequirements::all_requirements()
                .into_iter()
                .map(|r| r.id)
                .collect(),
            _ => {
                return Err(ComplianceError::UnsupportedFramework(format!(
                    "{:?}",
                    framework
                )))
            }
        };

        let mut gaps = Vec::new();

        for req_id in requirements {
            let evidence = self.evidence_store.get(&req_id);
            let validation = self.validation_cache.get(&req_id);

            let gap_type = if evidence.is_none() {
                GapType::NoEvidence
            } else if let Some(val) = validation {
                if !val.compliant {
                    GapType::InsufficientEvidence
                } else {
                    continue; // No gap
                }
            } else {
                GapType::NotValidated
            };

            gaps.push(Gap {
                requirement_id: req_id,
                gap_type,
                evidence_count: evidence.map(|e| e.len()).unwrap_or(0),
                severity: Severity::Medium, // Default, should be from requirement
                recommendation: format!("Collect evidence and validate requirement"),
            });
        }

        Ok(GapAnalysis {
            framework,
            analyzed_at: Utc::now(),
            total_gaps: gaps.len(),
            gaps,
        })
    }

    /// Clear all cached validation results
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }

    /// Get evidence count for requirement
    pub fn evidence_count(&self, requirement_id: &str) -> usize {
        self.evidence_store
            .get(requirement_id)
            .map(|e| e.len())
            .unwrap_or(0)
    }

    /// Verify data lineage for BCBS 239 compliance
    pub fn verify_lineage(&self, lineage: &DataLineage) -> LineageVerificationResult {
        let mut issues = Vec::new();

        if lineage.transformation_steps.is_empty() {
            issues.push("No transformation steps documented".to_string());
        }

        if lineage.data_quality_score < 0.95 {
            issues.push(format!(
                "Data quality score below threshold: {:.2}",
                lineage.data_quality_score
            ));
        }

        let days_since_verified = (Utc::now() - lineage.last_verified).num_days();
        if days_since_verified > 30 {
            issues.push(format!(
                "Lineage not verified in {} days",
                days_since_verified
            ));
        }

        LineageVerificationResult {
            lineage_id: lineage.lineage_id,
            verified_at: Utc::now(),
            is_valid: issues.is_empty(),
            issues,
            data_quality_score: lineage.data_quality_score,
        }
    }
}

impl Default for ComplianceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkValidationResult {
    pub framework: ComplianceFramework,
    pub validated_at: DateTime<Utc>,
    pub total_requirements: usize,
    pub compliant_requirements: usize,
    pub validation_results: Vec<ValidationResult>,
    pub compliance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    pub framework: ComplianceFramework,
    pub analyzed_at: DateTime<Utc>,
    pub total_gaps: usize,
    pub gaps: Vec<Gap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gap {
    pub requirement_id: String,
    pub gap_type: GapType,
    pub evidence_count: usize,
    pub severity: Severity,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapType {
    NoEvidence,
    InsufficientEvidence,
    NotValidated,
    ControlMissing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageVerificationResult {
    pub lineage_id: Uuid,
    pub verified_at: DateTime<Utc>,
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub data_quality_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_with_evidence() {
        let mut validator = ComplianceValidator::new();

        let evidence = Evidence::new(
            EvidenceType::Documentation,
            "Access control policy".to_string(),
            b"policy content",
            "auditor".to_string(),
        );

        validator.add_evidence("FFIEC-IS-001".to_string(), evidence);

        assert_eq!(validator.evidence_count("FFIEC-IS-001"), 1);
        assert_eq!(validator.evidence_count("UNKNOWN"), 0);
    }

    #[test]
    fn test_framework_validation() {
        let mut validator = ComplianceValidator::new();

        // Add some evidence
        for req in FFIECRequirements::all_requirements() {
            for _ in 0..req.expected_evidence.len() {
                let evidence = Evidence::new(
                    EvidenceType::Documentation,
                    "Test evidence".to_string(),
                    b"content",
                    "system".to_string(),
                );
                validator.add_evidence(req.id.clone(), evidence);
            }
        }

        let result = validator.validate_framework(ComplianceFramework::FFIEC);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.total_requirements > 0);
    }

    #[test]
    fn test_gap_analysis() {
        let validator = ComplianceValidator::new();
        let gaps = validator.gap_analysis(ComplianceFramework::FFIEC).unwrap();

        // Should have gaps since no evidence was added
        assert!(gaps.total_gaps > 0);
        assert_eq!(gaps.total_gaps, gaps.gaps.len());
    }
}
