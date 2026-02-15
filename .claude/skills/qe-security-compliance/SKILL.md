---
name: "QE Security Compliance"
description: "Security auditing, vulnerability scanning, and compliance validation for OWASP, SOC2, GDPR, and other standards."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-security-compliance.yaml

---

# QE Security Compliance

## Purpose

Guide the use of v3's security and compliance testing capabilities including SAST/DAST scanning, vulnerability detection, compliance auditing, and security gate enforcement.

## Activation

- When performing security audits
- When scanning for vulnerabilities
- When validating compliance
- When checking dependencies
- When setting up security gates

## Quick Start

```bash
# Full security scan
aqe security scan --scope src/ --checks all

# Vulnerability check
aqe security vulns --dependencies --severity critical,high

# Compliance audit
aqe security compliance --standard soc2 --output report.html

# OWASP check
aqe security owasp --top-10 --scope src/
```

## Agent Workflow

```typescript
// Security audit
Task("Security audit", `
  Perform comprehensive security audit:
  - SAST scan for code vulnerabilities
  - Dependency vulnerability check
  - Secret detection in code and configs
  - OWASP Top 10 validation
  Generate security report with remediation steps.
`, "qe-security-auditor")

// Compliance validation
Task("SOC2 compliance check", `
  Validate SOC2 compliance requirements:
  - Access control verification
  - Encryption validation
  - Audit logging check
  - Data retention compliance
  Generate compliance evidence report.
`, "qe-compliance-checker")
```

## Security Operations

### 1. SAST Scanning

```typescript
await securityScanner.staticAnalysis({
  scope: 'src/**/*.ts',
  checks: [
    'sql-injection',
    'xss',
    'command-injection',
    'path-traversal',
    'insecure-crypto',
    'hardcoded-secrets'
  ],
  rules: 'owasp-top-10',
  severity: ['critical', 'high', 'medium']
});
```

### 2. Dependency Scanning

```typescript
await securityScanner.dependencyCheck({
  sources: ['package.json', 'package-lock.json'],
  checks: {
    knownVulnerabilities: true,
    outdatedPackages: true,
    licenseCompliance: true,
    supplyChainRisk: true
  },
  severity: ['critical', 'high'],
  autoFix: {
    enabled: true,
    dryRun: false
  }
});
```

### 3. Compliance Audit

```typescript
await complianceChecker.audit({
  standards: ['SOC2', 'GDPR', 'HIPAA'],
  scope: {
    code: 'src/',
    configs: 'config/',
    infrastructure: 'terraform/'
  },
  output: {
    gaps: true,
    evidence: true,
    recommendations: true
  }
});
```

### 4. Secret Detection

```typescript
await securityScanner.detectSecrets({
  scope: ['.', 'config/', '.env*'],
  patterns: [
    'api-keys',
    'passwords',
    'tokens',
    'private-keys',
    'connection-strings'
  ],
  exclude: ['*.test.ts', 'mocks/'],
  action: {
    onDetect: 'block',
    notify: ['security-team']
  }
});
```

## OWASP Top 10 Coverage

```yaml
owasp_2021:
  A01_broken_access_control:
    checks: [privilege-escalation, idor, cors-misconfiguration]
    automated: true

  A02_cryptographic_failures:
    checks: [weak-encryption, missing-encryption, key-management]
    automated: true

  A03_injection:
    checks: [sql, nosql, command, xss, ldap]
    automated: true

  A04_insecure_design:
    checks: [threat-modeling, secure-patterns]
    automated: partial

  A05_security_misconfiguration:
    checks: [default-credentials, unnecessary-features]
    automated: true

  A06_vulnerable_components:
    checks: [outdated-deps, known-cves]
    automated: true

  A07_auth_failures:
    checks: [weak-passwords, session-issues]
    automated: true

  A08_software_data_integrity:
    checks: [insecure-deserialization, cicd-security]
    automated: partial

  A09_logging_monitoring:
    checks: [insufficient-logging, missing-alerts]
    automated: partial

  A10_ssrf:
    checks: [server-side-request-forgery]
    automated: true
```

## Security Report

```typescript
interface SecurityReport {
  summary: {
    score: number;  // 0-100
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  vulnerabilities: {
    id: string;
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    location: string;
    description: string;
    remediation: string;
    cwe: string;
    owasp: string;
  }[];
  dependencies: {
    vulnerable: number;
    outdated: number;
    details: DependencyVuln[];
  };
  compliance: {
    standard: string;
    status: 'compliant' | 'non-compliant' | 'partial';
    gaps: ComplianceGap[];
    evidence: Evidence[];
  }[];
  secrets: {
    detected: number;
    locations: SecretLocation[];
  };
}
```

## Security Gates

```yaml
security_gates:
  block_merge:
    - critical_vulnerabilities > 0
    - high_vulnerabilities > 2
    - secrets_detected > 0
    - compliance_failures > 0

  warn:
    - medium_vulnerabilities > 5
    - outdated_dependencies > 10

  enforce:
    - signed_commits: required
    - code_review: required
    - security_scan: required
```

## Compliance Standards

| Standard | Scope | Auto-Check |
|----------|-------|------------|
| SOC2 | Security controls | Partial |
| GDPR | Data privacy | Partial |
| HIPAA | Health data | Partial |
| PCI-DSS | Payment data | Yes |
| ISO 27001 | InfoSec | Partial |

## Coordination

**Primary Agents**: qe-security-auditor, qe-security-scanner, qe-compliance-checker
**Coordinator**: qe-security-coordinator
**Related Skills**: qe-quality-assessment, qe-contract-testing
