# Security Compliance Domain Shard

**Domain**: security-compliance
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Zero Critical Vulnerabilities**: Code changes affecting auth, security, or sensitive data MUST have zero critical vulnerabilities before deployment; no exceptions without security team sign-off.

2. **SAST Before Merge**: Static Application Security Testing (SAST) MUST complete before any merge to protected branches; bypass is a constitutional violation.

3. **Secret Detection Mandatory**: All code MUST pass secret scanning; detected secrets MUST be revoked immediately and commits purged from history.

4. **Compliance Evidence Required**: Compliance claims MUST be backed by evidence (audit logs, scan results, attestations); unsubstantiated claims are prohibited.

5. **DAST in Staging**: Dynamic Application Security Testing (DAST) MUST run against staging environments before production deployment.

6. **Vulnerability Triage SLA**: Critical vulnerabilities MUST be triaged within 24 hours; high within 72 hours. SLA violations escalate automatically.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Security Score | 0.8 | 0.95 | < 0.6 |
| SAST Pass Rate | 1.0 (critical) | 0.95 (all) | < 0.9 |
| Secret Detection | 0 secrets | 0 secrets | > 0 secrets |
| Compliance Score | 0.8 | 0.95 | < 0.7 |
| Vulnerability SLA | 100% | 100% | < 90% |
| DAST Coverage | 0.7 | 0.9 | < 0.5 |

---

## Invariants

```
INVARIANT zero_critical_vulnerabilities:
  FOR ALL change IN security_sensitive_changes:
    IF change.affects_auth OR change.affects_security OR change.affects_sensitive_data THEN
      change.critical_vulnerabilities = 0
```

```
INVARIANT sast_before_merge:
  FOR ALL merge IN protected_branch_merges:
    EXISTS sast_scan WHERE
      sast_scan.commit = merge.source_commit AND
      sast_scan.status = 'passed' AND
      sast_scan.timestamp < merge.timestamp
```

```
INVARIANT secret_detection:
  FOR ALL commit IN commits:
    commit.detected_secrets = 0 OR
    (commit.secrets_revoked = true AND commit.purged_from_history = true)
```

```
INVARIANT compliance_evidence:
  FOR ALL claim IN compliance_claims:
    EXISTS evidence WHERE
      evidence.claim_id = claim.id AND
      evidence.type IN ['audit_log', 'scan_result', 'attestation'] AND
      evidence.verified = true
```

```
INVARIANT vulnerability_sla:
  FOR ALL vuln IN vulnerabilities:
    IF vuln.severity = 'critical' THEN
      (NOW() - vuln.detected_at) < 24_HOURS OR vuln.triaged = true
    IF vuln.severity = 'high' THEN
      (NOW() - vuln.detected_at) < 72_HOURS OR vuln.triaged = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/security-compliance/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Security Scanner Service | `services/security-scanner.ts` | SAST/DAST orchestration |
| Security Auditor Service | `services/security-auditor.ts` | Security posture assessment |
| Compliance Validator Service | `services/compliance-validator.ts` | Standards validation |
| Scanners | `services/scanners/` | Individual scanner implementations |
| Security Compliance Coordinator | `coordinator.ts` | Workflow orchestration |

**Supported Standards**: OWASP Top 10, PCI-DSS, HIPAA, SOC 2, GDPR (via ComplianceValidatorService).

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-security-scanner` | Full scanning, vulnerability detection |
| **Secondary** | `qe-security-auditor` | Posture assessment, triage |
| **Secondary** | `qe-compliance-validator` | Standards validation, evidence collection |
| **Support** | `qe-code-analyst` | Provide code paths for analysis |
| **Approval** | `security-team` (human) | Critical vulnerability override |

**Forbidden Agents**: Non-security agents MUST NOT override security findings or bypass security gates.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Critical vulnerability in auth code | CRITICAL | Block deployment, escalate to security team immediately |
| Secret detected in commit | CRITICAL | Revoke secret, purge history, escalate |
| SAST bypass attempted | CRITICAL | Block merge, escalate to Queen Coordinator |
| Compliance score < 0.7 | CRITICAL | Block deployment, initiate gap analysis |
| Vulnerability SLA breach | HIGH | Escalate to Queen Coordinator |
| DAST coverage < 0.5 | HIGH | Block production deploy, escalate |
| High vulnerability untriaged > 72h | HIGH | Auto-escalate to security team |
| Unsubstantiated compliance claim | MEDIUM | Block claim, request evidence |

---

## Memory Namespace

- **Namespace**: `qe-patterns/security-compliance`
- **Retention**: 365 days (compliance requirement)
- **Contradiction Check**: Enabled
- **Audit Trail**: Required for all findings

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `code-intelligence` | Input | Receive code paths for scanning |
| `quality-assessment` | Output | Report security score |
| `requirements-validation` | Bidirectional | Security requirements validation |
| `test-execution` | Input | Receive auth test results |
| `learning-optimization` | Output | Share vulnerability patterns |

---

## Vulnerability Severity Classification

| Severity | CVSS Score | SLA | Examples |
|----------|------------|-----|----------|
| Critical | 9.0 - 10.0 | 24h | RCE, Auth bypass, SQL injection in auth |
| High | 7.0 - 8.9 | 72h | XSS, CSRF, Privilege escalation |
| Medium | 4.0 - 6.9 | 7 days | Information disclosure, DoS |
| Low | 0.1 - 3.9 | 30 days | Minor info leak, verbose errors |

---

## Compliance Evidence Schema

```typescript
interface ComplianceEvidence {
  id: string;
  claimId: string;
  standard: 'OWASP' | 'PCI-DSS' | 'HIPAA' | 'SOC2' | 'GDPR';
  control: string;
  type: 'audit_log' | 'scan_result' | 'attestation' | 'policy_document';
  collectedAt: Date;
  expiresAt: Date;
  verified: boolean;
  verifier: string;
  artifacts: {
    name: string;
    hash: string;
    location: string;
  }[];
}
```

---

## Secret Detection Categories

| Category | Action | Example |
|----------|--------|---------|
| API Keys | Revoke immediately | AWS, GCP, Azure keys |
| Passwords | Force rotation | Database, service passwords |
| Tokens | Invalidate | JWT, OAuth tokens |
| Private Keys | Regenerate | SSH, SSL certificates |
| Connection Strings | Rotate credentials | Database URLs with embedded passwords |

---

*This shard is enforced by @claude-flow/guidance governance system.*
