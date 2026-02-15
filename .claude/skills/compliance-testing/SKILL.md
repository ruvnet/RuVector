---
name: compliance-testing
description: "Regulatory compliance testing for GDPR, CCPA, HIPAA, SOC2, PCI-DSS and industry-specific regulations. Use when ensuring legal compliance, preparing for audits, or handling sensitive data."
category: specialized-testing
priority: high
tokenEstimate: 900
agents: [qe-security-scanner, qe-test-executor, qe-quality-gate]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [compliance, gdpr, hipaa, pci-dss, ccpa, soc2, privacy, audit]

# ADR-056 Validation Stack Configuration
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/compliance-testing.yaml
---

# Compliance Testing

<default_to_action>
When validating regulatory compliance:
1. IDENTIFY applicable regulations (GDPR, HIPAA, PCI-DSS, etc.)
2. MAP requirements to testable controls
3. TEST data rights (access, erasure, portability)
4. VERIFY encryption and access logging
5. GENERATE audit-ready reports with evidence

**Quick Compliance Checklist:**
- Data subject rights work (access, delete, export)
- PII is encrypted at rest and in transit
- Access to sensitive data is logged
- Consent is tracked with timestamps
- Payment card data not stored (only tokenized)

**Critical Success Factors:**
- Non-compliance = €20M or 4% revenue (GDPR)
- Audit trail everything
- Test continuously, not just before audits
</default_to_action>

## Quick Reference Card

### When to Use
- Legal compliance requirements
- Before security audits
- Handling PII/PHI/PCI data
- Entering new markets (EU, CA, healthcare)

### Major Regulations
| Regulation | Scope | Key Focus |
|------------|-------|-----------|
| **GDPR** | EU data | Privacy rights, consent |
| **CCPA** | California | Consumer data rights |
| **HIPAA** | Healthcare | PHI protection |
| **PCI-DSS** | Payments | Card data security |
| **SOC2** | SaaS | Security controls |

### Penalties
| Regulation | Maximum Fine |
|------------|--------------|
| **GDPR** | €20M or 4% revenue |
| **HIPAA** | $1.5M per violation |
| **PCI-DSS** | $100k/month |
| **CCPA** | $7,500 per violation |

---

## GDPR Compliance Testing

```javascript
// Test data subject rights
test('user can request their data', async () => {
  const response = await api.post('/data-export', { userId });

  expect(response.status).toBe(200);
  expect(response.data.downloadUrl).toBeDefined();

  const data = await downloadFile(response.data.downloadUrl);
  expect(data).toHaveProperty('profile');
  expect(data).toHaveProperty('orders');
});

test('user can delete their account', async () => {
  await api.delete(`/users/${userId}`);

  // All personal data deleted
  expect(await db.users.findOne({ id: userId })).toBeNull();
  expect(await db.orders.find({ userId })).toHaveLength(0);

  // Audit log retained (legal requirement)
  expect(await db.auditLogs.find({ userId })).toBeDefined();
});

test('consent is tracked', async () => {
  await api.post('/consent', {
    userId, type: 'marketing', granted: true,
    timestamp: new Date(), ipAddress: '192.168.1.1'
  });

  const consent = await db.consents.findOne({ userId, type: 'marketing' });
  expect(consent.timestamp).toBeDefined();
  expect(consent.ipAddress).toBeDefined();
});
```

---

## HIPAA Compliance Testing

```javascript
// Test PHI security
test('PHI is encrypted at rest', async () => {
  const patient = await db.patients.create({
    ssn: '123-45-6789',
    medicalHistory: 'Diabetes'
  });

  const raw = await db.raw('SELECT * FROM patients WHERE id = ?', patient.id);
  expect(raw.ssn).not.toBe('123-45-6789'); // Should be encrypted
});

test('access to PHI is logged', async () => {
  await api.get('/patients/123', {
    headers: { 'User-Id': 'doctor456' }
  });

  const auditLog = await db.auditLogs.findOne({
    resourceType: 'patient',
    resourceId: '123',
    userId: 'doctor456'
  });

  expect(auditLog.action).toBe('read');
  expect(auditLog.timestamp).toBeDefined();
});
```

---

## PCI-DSS Compliance Testing

```javascript
// Test payment card handling
test('credit card numbers not stored', async () => {
  await api.post('/payment', {
    cardNumber: '4242424242424242',
    expiry: '12/25', cvv: '123'
  });

  const payment = await db.payments.findOne({ /* ... */ });
  expect(payment.cardNumber).toBeUndefined();
  expect(payment.last4).toBe('4242'); // Only last 4
  expect(payment.tokenId).toBeDefined(); // Token from gateway
});

test('CVV never stored', async () => {
  const payments = await db.raw('SELECT * FROM payments');
  const hasCVV = payments.some(p =>
    JSON.stringify(p).toLowerCase().includes('cvv')
  );
  expect(hasCVV).toBe(false);
});
```

---

## Agent-Driven Compliance

```typescript
// Comprehensive compliance validation
await Task("Compliance Validation", {
  regulations: ['GDPR', 'PCI-DSS'],
  scope: 'full-application',
  generateAuditReport: true
}, "qe-security-scanner");

// Returns:
// {
//   gdpr: { compliant: true, controls: 12, passed: 12 },
//   pciDss: { compliant: false, controls: 8, passed: 7 },
//   violations: [{ control: 'card-storage', severity: 'critical' }],
//   auditReport: 'compliance-audit-2025-12-02.pdf'
// }
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/compliance-testing/
├── regulations/*        - Regulation requirements
├── controls/*           - Control test results
├── audit-reports/*      - Generated audit reports
└── violations/*         - Compliance violations
```

### Fleet Coordination
```typescript
const complianceFleet = await FleetManager.coordinate({
  strategy: 'compliance-validation',
  agents: [
    'qe-security-scanner',   // Scan for vulnerabilities
    'qe-test-executor',      // Execute compliance tests
    'qe-quality-gate'        // Block non-compliant releases
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [security-testing](../security-testing/) - Security vulnerabilities
- [test-data-management](../test-data-management/) - PII handling
- [accessibility-testing](../accessibility-testing/) - Legal requirements

---

## Remember

**Compliance is mandatory, not optional.** Fines are severe: GDPR up to €20M or 4% of revenue, HIPAA up to $1.5M per violation. But beyond fines, non-compliance damages reputation and user trust.

**Audit trail everything.** Every access to sensitive data, every consent, every deletion must be logged with timestamps and user IDs.

**With Agents:** Agents validate compliance requirements continuously, detect violations early, and generate audit-ready reports. Catch compliance issues in development, not in audits.
