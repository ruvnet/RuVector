# Leviathan Compliance

Bank-grade regulatory compliance framework for Northern Trust and financial institutions.

## Features

- **FFIEC Compliance**: Federal Financial Institutions Examination Council IT examination handbook
- **BCBS 239**: Basel Committee Principles for effective risk data aggregation and risk reporting
- **SR 11-7**: Federal Reserve Model Risk Management guidance
- **GDPR**: General Data Protection Regulation compliance helpers
- **Cryptographic Audit Trail**: Ed25519 signed compliance reports with Blake3 evidence hashing
- **Automated Validation**: Continuous compliance monitoring and gap analysis
- **Control Framework**: Pre-defined controls with testing procedures

## Supported Frameworks

| Framework | Description | Key Features |
|-----------|-------------|--------------|
| FFIEC | IT Examination Handbook | Access control, encryption, audit logging, BCP/DR |
| BCBS 239 | Risk Data Aggregation | Data lineage, accuracy, completeness, timeliness |
| SR 11-7 | Model Risk Management | Model inventory, validation, monitoring |
| GDPR | Data Protection | Data subject rights, consent tracking, portability |

## Quick Start

### Basic Validation

```rust
use leviathan_compliance::{
    ComplianceValidator,
    ComplianceFramework,
    Evidence,
    EvidenceType,
};

// Create validator
let mut validator = ComplianceValidator::new();

// Add evidence
let evidence = Evidence::new(
    EvidenceType::Documentation,
    "Access control policy v1.0".to_string(),
    b"policy content here",
    "auditor@example.com".to_string(),
);

validator.add_evidence("FFIEC-IS-001".to_string(), evidence);

// Validate framework
let result = validator.validate_framework(ComplianceFramework::FFIEC)?;

println!("Compliance Score: {:.1}%", result.compliance_score * 100.0);
println!("Compliant: {}/{}",
    result.compliant_requirements,
    result.total_requirements
);
```

### Generate Compliance Report

```rust
use leviathan_compliance::{
    ComplianceValidator,
    ComplianceFramework,
    audit_report::ComplianceReportBuilder,
};
use ed25519_dalek::SigningKey;

let mut validator = ComplianceValidator::new();

// ... add evidence ...

// Generate signed report
let mut csprng = rand::rngs::OsRng;
let signing_key = SigningKey::generate(&mut csprng);

let report = validator
    .generate_report(
        ComplianceFramework::BCBS239,
        "Northern Trust Bank".to_string(),
        "Chief Compliance Officer".to_string(),
    )?;

// Sign report
let mut signed_report = report;
signed_report.sign(&signing_key)?;

// Verify signature
assert!(signed_report.verify_signature()?);

// Export to JSON
let json = signed_report.to_json()?;
println!("{}", json);
```

### BCBS 239 Data Lineage Tracking

```rust
use leviathan_compliance::bcbs239::DataLineage;

let mut lineage = DataLineage::new(
    "Total Credit Exposure".to_string(),
    "Trading System".to_string(),
    "Risk Data Warehouse".to_string(),
);

// Add transformation step
let output = b"transformed_data";
lineage.add_transformation(
    "Currency normalization to USD".to_string(),
    "Apply FX rate from market data feed".to_string(),
    vec!["FX rate must be from official source".to_string()],
    output,
);

// Verify transformation integrity
assert!(lineage.verify_transformation(0, output));
```

### SR 11-7 Model Inventory

```rust
use leviathan_compliance::sr117::{ModelInventoryEntry, ModelType, ModelRiskRating};
use chrono::Utc;

let model = ModelInventoryEntry {
    model_id: "MDL-001".to_string(),
    model_name: "Credit Risk PD Model".to_string(),
    model_type: ModelType::CreditRisk,
    model_owner: "John Doe".to_string(),
    business_unit: "Credit Risk".to_string(),
    description: "Probability of Default estimation".to_string(),
    risk_rating: ModelRiskRating::High,
    validation_status: ValidationStatus::Validated,
    last_validation_date: Some(Utc::now()),
    next_validation_due: Utc::now() + chrono::Duration::days(365),
    validation_frequency_months: 12,
    material_changes_since_validation: vec![],
    known_limitations: vec!["Limited data for low-default portfolios".to_string()],
    compensating_controls: vec!["Expert override process".to_string()],
};

if model.requires_revalidation() {
    println!("Model validation overdue!");
}
```

### GDPR Data Subject Requests

```rust
use leviathan_compliance::gdpr::{DataSubjectRequest, DataSubjectRight};
use uuid::Uuid;

// Create access request
let mut request = DataSubjectRequest::new(
    Uuid::new_v4(), // subject_id
    DataSubjectRight::Access,
);

println!("Must respond within {} days", request.days_until_due());

// Check if overdue
if request.is_overdue() {
    eprintln!("GDPR deadline violation!");
}
```

### Control Testing

```rust
use leviathan_compliance::controls::{ControlLibrary, ControlTest, ControlException};
use uuid::Uuid;

// Get MFA control
let mfa_control = ControlLibrary::mfa_control();

// Test the control
let mut test = ControlTest::new(
    mfa_control.control_id.clone(),
    "Security Auditor".to_string(),
    100, // sample size
);

// Add exceptions found during testing
test.add_exception(ControlException {
    exception_id: Uuid::new_v4(),
    description: "Service account without MFA".to_string(),
    severity: Severity::High,
    root_cause: Some("Legacy system integration".to_string()),
    remediation: Some("Implement API key rotation".to_string()),
    remediation_due: Some(Utc::now() + chrono::Duration::days(30)),
});

// Calculate effectiveness
test.calculate_effectiveness();
println!("Control effectiveness: {:?}", test.effectiveness);
```

### Gap Analysis

```rust
use leviathan_compliance::{ComplianceValidator, ComplianceFramework};

let validator = ComplianceValidator::new();

// Perform gap analysis
let gaps = validator.gap_analysis(ComplianceFramework::FFIEC)?;

println!("Total gaps identified: {}", gaps.total_gaps);

for gap in &gaps.gaps {
    println!("  - {} ({:?}): {}",
        gap.requirement_id,
        gap.gap_type,
        gap.recommendation
    );
}
```

## Pre-defined Requirements

### FFIEC Requirements

- **FFIEC-IS-001**: Access Control (Multi-factor authentication, least privilege)
- **FFIEC-IS-002**: Data Encryption (AES-256 at rest, TLS 1.2+ in transit)
- **FFIEC-AM-001**: Audit Logging (Comprehensive logging with integrity protection)
- **FFIEC-BCP-001**: Business Continuity (DR plans, RTO/RPO, annual testing)
- **FFIEC-DA-001**: Secure SDLC (Code review, security testing, change management)
- **FFIEC-OT-001**: Third-Party Risk (Vendor assessments, SOC 2 reports)

### BCBS 239 Principles

- **Principle 2**: Data Architecture and IT Infrastructure
- **Principle 3**: Accuracy and Integrity
- **Principle 4**: Completeness
- **Principle 5**: Timeliness
- **Principle 6**: Adaptability

### SR 11-7 Components

- **SR117-INV-001**: Model Inventory
- **SR117-VAL-001**: Independent Validation
- **SR117-MON-001**: Ongoing Monitoring
- **SR117-GOV-001**: Model Governance

### GDPR Articles

- **Article 6**: Lawful Processing (Legal basis required)
- **Articles 12-22**: Data Subject Rights (Access, erasure, portability)
- **Article 25**: Data Protection by Design

## Pre-defined Controls

All controls include testing procedures and framework mappings:

- **AC-001**: Multi-Factor Authentication
- **DC-001**: Data Encryption at Rest
- **CM-001**: Change Management Process
- **DQ-001**: Data Quality Validation (BCBS 239)
- **MV-001**: Independent Model Validation (SR 11-7)
- **AL-001**: Comprehensive Audit Logging

## Evidence Types

The framework supports multiple evidence types:

- **Documentation**: Policies, procedures, standards
- **AuditLog**: System logs and audit trails
- **TestResult**: Validation and testing outputs
- **CodeReview**: Security scan results
- **DataLineage**: Provenance and transformation tracking
- **AccessControl**: Permission records
- **CryptographicProof**: Cryptographic attestations
- **Attestation**: Third-party certifications

All evidence is cryptographically hashed using Blake3 for integrity verification.

## Compliance Report Structure

Reports include:

- **Framework**: Which regulatory framework
- **Status**: Compliant, PartiallyCompliant, NonCompliant, NotAssessed
- **Findings**: Detailed compliance gaps
- **Evidence**: Cryptographically verified artifacts
- **Metrics**: Compliance score, finding counts by severity
- **Signature**: Ed25519 digital signature for non-repudiation

## Testing

Run the comprehensive test suite:

```bash
cargo test
```

Run specific framework tests:

```bash
cargo test ffiec
cargo test bcbs239
cargo test sr117
cargo test gdpr
```

## Production Readiness

This crate is designed for actual banking compliance:

- Real regulatory requirement IDs referenced
- Industry-standard cryptography (Ed25519, Blake3)
- Comprehensive audit trails
- Automated evidence collection
- Gap analysis and remediation tracking
- Control testing procedures
- Data lineage verification (BCBS 239)
- Model risk management (SR 11-7)
- GDPR data subject rights handling

## Integration Example

```rust
use leviathan_compliance::{
    ComplianceValidator,
    ComplianceFramework,
    Evidence,
    EvidenceType,
    controls::ControlLibrary,
};

async fn comprehensive_compliance_check() -> Result<(), Box<dyn std::error::Error>> {
    let mut validator = ComplianceValidator::new();

    // 1. Collect evidence from systems
    let access_logs = fetch_access_logs().await?;
    let evidence = Evidence::new(
        EvidenceType::AuditLog,
        "90-day access logs".to_string(),
        &access_logs,
        "automated_collector".to_string(),
    );
    validator.add_evidence("FFIEC-IS-001".to_string(), evidence);

    // 2. Test controls
    let controls = ControlLibrary::controls_for_framework(ComplianceFramework::FFIEC);
    for control in controls {
        let test = test_control(&control).await?;
        // Store test results...
    }

    // 3. Validate all frameworks
    for framework in [
        ComplianceFramework::FFIEC,
        ComplianceFramework::BCBS239,
        ComplianceFramework::SR117,
        ComplianceFramework::GDPR,
    ] {
        let result = validator.validate_framework(framework)?;

        if result.compliance_score < 0.95 {
            alert_compliance_team(&result).await?;
        }
    }

    // 4. Generate executive report
    let report = validator.generate_report(
        ComplianceFramework::FFIEC,
        "Northern Trust".to_string(),
        "Chief Compliance Officer".to_string(),
    )?;

    // 5. Archive with signature
    archive_compliance_report(report).await?;

    Ok(())
}
```

## License

MIT OR Apache-2.0

## References

- [FFIEC IT Examination Handbook](https://ithandbook.ffiec.gov/)
- [BCBS 239 Principles](https://www.bis.org/publ/bcbs239.htm)
- [SR 11-7 Guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [GDPR Official Text](https://gdpr-info.eu/)
