//! Full Compliance Check Example
//!
//! Demonstrates comprehensive compliance validation across all frameworks.

use leviathan_compliance::{
    bcbs239::DataLineage,
    controls::{ControlLibrary, ControlTest},
    ffiec::FFIECRequirements,
    gdpr::{DataSubjectRequest, DataSubjectRight},
    sr117::{ModelInventoryEntry, ModelRiskRating, ModelType, ValidationStatus},
    ComplianceFramework, ComplianceValidator, Evidence, EvidenceType,
    Severity,
};
use chrono::Utc;
use ed25519_dalek::SigningKey;
use uuid::Uuid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Leviathan Compliance Framework Demo ===\n");

    // Initialize validator
    let mut validator = ComplianceValidator::new();

    // 1. FFIEC Compliance Check
    println!("1. FFIEC IT Examination Handbook Compliance");
    println!("--------------------------------------------------");

    let ffiec_requirements = FFIECRequirements::all_requirements();
    println!("Total FFIEC requirements: {}", ffiec_requirements.len());

    // Add evidence for access control
    let access_policy = Evidence::new(
        EvidenceType::Documentation,
        "Access Control Policy v2.0 - MFA required for all privileged accounts".to_string(),
        b"Access control policy content with MFA requirements",
        "security-team@example.com".to_string(),
    );
    validator.add_evidence("FFIEC-IS-001".to_string(), access_policy);

    // Add evidence for encryption
    let encryption_config = Evidence::new(
        EvidenceType::Documentation,
        "Encryption Policy - AES-256 for data at rest, TLS 1.3 for transit".to_string(),
        b"Encryption configuration and key management procedures",
        "security-team@example.com".to_string(),
    );
    validator.add_evidence("FFIEC-IS-002".to_string(), encryption_config);

    // Validate FFIEC
    let ffiec_result = validator.validate_framework(ComplianceFramework::FFIEC)?;
    println!(
        "FFIEC Compliance Score: {:.1}%",
        ffiec_result.compliance_score * 100.0
    );
    println!(
        "Compliant: {}/{}",
        ffiec_result.compliant_requirements, ffiec_result.total_requirements
    );
    println!();

    // 2. BCBS 239 Data Lineage Tracking
    println!("2. BCBS 239 Risk Data Aggregation");
    println!("--------------------------------------------------");

    let mut credit_exposure_lineage = DataLineage::new(
        "Total Credit Exposure".to_string(),
        "Trading System".to_string(),
        "Risk Data Warehouse".to_string(),
    );

    // Add transformation steps
    let fx_conversion = b"USD_normalized_exposure_data";
    credit_exposure_lineage.add_transformation(
        "FX Rate Normalization".to_string(),
        "Convert all exposures to USD using market FX rates".to_string(),
        vec!["FX rates from Bloomberg".to_string()],
        fx_conversion,
    );

    let netting = b"netted_exposure_data";
    credit_exposure_lineage.add_transformation(
        "Netting Calculation".to_string(),
        "Apply ISDA netting agreements".to_string(),
        vec!["ISDA master agreements validated".to_string()],
        netting,
    );

    println!(
        "Data lineage for: {}",
        credit_exposure_lineage.data_element
    );
    println!(
        "Transformation steps: {}",
        credit_exposure_lineage.transformation_steps.len()
    );
    println!(
        "Data quality score: {:.2}",
        credit_exposure_lineage.data_quality_score
    );

    // Verify lineage integrity
    let lineage_verification = validator.verify_lineage(&credit_exposure_lineage);
    println!("Lineage verification: {:?}", lineage_verification.is_valid);
    println!();

    // 3. SR 11-7 Model Risk Management
    println!("3. SR 11-7 Model Risk Management");
    println!("--------------------------------------------------");

    let credit_risk_model = ModelInventoryEntry {
        model_id: "MDL-CR-001".to_string(),
        model_name: "Retail Credit PD Model".to_string(),
        model_type: ModelType::CreditRisk,
        model_owner: "Dr. Jane Smith".to_string(),
        business_unit: "Consumer Credit Risk".to_string(),
        description: "Probability of Default model for retail credit portfolios".to_string(),
        risk_rating: ModelRiskRating::High,
        validation_status: ValidationStatus::Validated,
        last_validation_date: Some(Utc::now() - chrono::Duration::days(90)),
        next_validation_due: Utc::now() + chrono::Duration::days(275),
        validation_frequency_months: 12,
        material_changes_since_validation: vec![],
        known_limitations: vec![
            "Limited data for low-default portfolios".to_string(),
            "Economic stress scenarios may not capture tail risk".to_string(),
        ],
        compensating_controls: vec!["Expert judgment overlay for edge cases".to_string()],
    };

    println!("Model: {} ({})", credit_risk_model.model_name, credit_risk_model.model_id);
    println!("Risk Rating: {:?}", credit_risk_model.risk_rating);
    println!(
        "Validation current: {}",
        credit_risk_model.is_validation_current()
    );
    println!(
        "Days until next validation: {}",
        credit_risk_model.days_until_validation()
    );
    println!("Known limitations: {}", credit_risk_model.known_limitations.len());
    println!();

    // 4. GDPR Data Subject Rights
    println!("4. GDPR Data Subject Rights Management");
    println!("--------------------------------------------------");

    let subject_id = Uuid::new_v4();
    let access_request = DataSubjectRequest::new(subject_id, DataSubjectRight::Access);

    println!("Data Subject Request: {:?}", access_request.request_type);
    println!("Request ID: {}", access_request.request_id);
    println!("Days until response due: {}", access_request.days_until_due());
    println!("Is overdue: {}", access_request.is_overdue());
    println!();

    // 5. Control Testing
    println!("5. Control Testing");
    println!("--------------------------------------------------");

    let mfa_control = ControlLibrary::mfa_control();
    println!("Testing control: {}", mfa_control.name);
    println!("Control ID: {}", mfa_control.control_id);
    println!("Control type: {:?}", mfa_control.control_type);
    println!("Implementation: {:?}", mfa_control.implementation);

    let mut mfa_test = ControlTest::new(
        mfa_control.control_id.clone(),
        "Security Auditor".to_string(),
        100,
    );

    // Simulate testing - found 2 exceptions
    mfa_test.add_exception(leviathan_compliance::controls::ControlException {
        exception_id: Uuid::new_v4(),
        description: "Service account SA_BATCH_001 does not have MFA".to_string(),
        severity: Severity::High,
        root_cause: Some("Legacy batch processing system".to_string()),
        remediation: Some("Implement API key rotation and IP whitelisting".to_string()),
        remediation_due: Some(Utc::now() + chrono::Duration::days(30)),
    });

    mfa_test.calculate_effectiveness();
    println!("Test sample size: {}", mfa_test.sample_size);
    println!("Exceptions found: {}", mfa_test.exceptions.len());
    println!("Control effectiveness: {:?}", mfa_test.effectiveness);
    println!();

    // 6. Generate Comprehensive Report
    println!("6. Generating Compliance Report");
    println!("--------------------------------------------------");

    // Generate signing key (in production, this should be securely managed)
    let signing_key = SigningKey::from_bytes(&[42u8; 32]);

    let mut report = validator.generate_report(
        ComplianceFramework::FFIEC,
        "Northern Trust Bank".to_string(),
        "Chief Compliance Officer".to_string(),
    )?;

    println!("Report ID: {}", report.report_id);
    println!("Framework: {:?}", report.framework);
    println!("Status: {:?}", report.status);
    println!("Compliance Score: {:.1}%", report.metrics.compliance_score * 100.0);
    println!("Total Findings: {}", report.findings.len());
    println!("Evidence Count: {}", report.evidence.len());

    // Sign the report
    report.sign(&signing_key)?;
    println!("\nReport signed: {}", report.signature.is_some());

    // Verify signature
    let signature_valid = report.verify_signature()?;
    println!("Signature verified: {}", signature_valid);

    // Export to JSON
    let report_json = report.to_json()?;
    println!("\nReport JSON size: {} bytes", report_json.len());

    // Summary
    let summary = report.summary();
    println!("\n=== Report Summary ===");
    println!("Framework: {:?}", summary.framework);
    println!("Status: {:?}", summary.status);
    println!("Compliance Score: {:.1}%", summary.compliance_score * 100.0);
    println!("Total Findings: {}", summary.total_findings);
    println!("Critical Findings: {}", summary.critical_findings);
    println!("Evidence Count: {}", summary.evidence_count);
    println!("Digitally Signed: {}", summary.is_signed);

    // 7. Gap Analysis
    println!("\n7. Gap Analysis");
    println!("--------------------------------------------------");

    let gaps = validator.gap_analysis(ComplianceFramework::FFIEC)?;
    println!("Total gaps identified: {}", gaps.total_gaps);

    for (i, gap) in gaps.gaps.iter().take(5).enumerate() {
        println!("\nGap #{}", i + 1);
        println!("  Requirement: {}", gap.requirement_id);
        println!("  Type: {:?}", gap.gap_type);
        println!("  Severity: {:?}", gap.severity);
        println!("  Evidence count: {}", gap.evidence_count);
        println!("  Recommendation: {}", gap.recommendation);
    }

    println!("\n=== Compliance Check Complete ===");

    Ok(())
}
