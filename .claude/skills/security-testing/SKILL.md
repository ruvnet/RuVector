---
name: security-testing
description: "Test for security vulnerabilities using OWASP principles. Use when conducting security audits, testing auth, or implementing security practices."
category: specialized-testing
priority: critical
tokenEstimate: 1200
agents: [qe-security-scanner, qe-api-contract-validator, qe-quality-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [security, owasp, sast, dast, vulnerabilities, auth, injection]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/security-testing.yaml
---

# Security Testing

<default_to_action>
When testing security or conducting audits:
1. TEST OWASP Top 10 vulnerabilities systematically
2. VALIDATE authentication and authorization on every endpoint
3. SCAN dependencies for known vulnerabilities (npm audit)
4. CHECK for injection attacks (SQL, XSS, command)
5. VERIFY secrets aren't exposed in code/logs

**Quick Security Checks:**
- Access control → Test horizontal/vertical privilege escalation
- Crypto → Verify password hashing, HTTPS, no sensitive data exposed
- Injection → Test SQL injection, XSS, command injection
- Auth → Test weak passwords, session fixation, MFA enforcement
- Config → Check error messages don't leak info

**Critical Success Factors:**
- Think like an attacker, build like a defender
- Security is built in, not added at the end
- Test continuously in CI/CD, not just before release
</default_to_action>

## Quick Reference Card

### When to Use
- Security audits and penetration testing
- Testing authentication/authorization
- Validating input sanitization
- Reviewing security configuration

### OWASP Top 10 (2021)
| # | Vulnerability | Key Test |
|---|---------------|----------|
| 1 | Broken Access Control | User A accessing User B's data |
| 2 | Cryptographic Failures | Plaintext passwords, HTTP |
| 3 | Injection | SQL/XSS/command injection |
| 4 | Insecure Design | Rate limiting, session timeout |
| 5 | Security Misconfiguration | Verbose errors, exposed /admin |
| 6 | Vulnerable Components | npm audit, outdated packages |
| 7 | Auth Failures | Weak passwords, no MFA |
| 8 | Integrity Failures | Unsigned updates, malware |
| 9 | Logging Failures | No audit trail for breaches |
| 10 | SSRF | Server fetching internal URLs |

### Tools
| Type | Tool | Purpose |
|------|------|---------|
| SAST | SonarQube, Semgrep | Static code analysis |
| DAST | OWASP ZAP, Burp | Dynamic scanning |
| Deps | npm audit, Snyk | Dependency vulnerabilities |
| Secrets | git-secrets, TruffleHog | Secret scanning |

### Agent Coordination
- `qe-security-scanner`: Multi-layer SAST/DAST scanning
- `qe-api-contract-validator`: API security testing
- `qe-quality-analyzer`: Security code review

---

## Key Vulnerability Tests

### 1. Broken Access Control
```javascript
// Horizontal escalation - User A accessing User B's data
test('user cannot access another user\'s order', async () => {
  const userAToken = await login('userA');
  const userBOrder = await createOrder('userB');

  const response = await api.get(`/orders/${userBOrder.id}`, {
    headers: { Authorization: `Bearer ${userAToken}` }
  });
  expect(response.status).toBe(403);
});

// Vertical escalation - Regular user accessing admin
test('regular user cannot access admin', async () => {
  const userToken = await login('regularUser');
  expect((await api.get('/admin/users', {
    headers: { Authorization: `Bearer ${userToken}` }
  })).status).toBe(403);
});
```

### 2. Injection Attacks
```javascript
// SQL Injection
test('prevents SQL injection', async () => {
  const malicious = "' OR '1'='1";
  const response = await api.get(`/products?search=${malicious}`);
  expect(response.body.length).toBeLessThan(100); // Not all products
});

// XSS
test('sanitizes HTML output', async () => {
  const xss = '<script>alert("XSS")</script>';
  await api.post('/comments', { text: xss });

  const html = (await api.get('/comments')).body;
  expect(html).toContain('&lt;script&gt;');
  expect(html).not.toContain('<script>');
});
```

### 3. Cryptographic Failures
```javascript
test('passwords are hashed', async () => {
  await db.users.create({ email: 'test@example.com', password: 'MyPassword123' });
  const user = await db.users.findByEmail('test@example.com');

  expect(user.password).not.toBe('MyPassword123');
  expect(user.password).toMatch(/^\$2[aby]\$\d{2}\$/); // bcrypt
});

test('no sensitive data in API response', async () => {
  const response = await api.get('/users/me');
  expect(response.body).not.toHaveProperty('password');
  expect(response.body).not.toHaveProperty('ssn');
});
```

### 4. Security Misconfiguration
```javascript
test('errors don\'t leak sensitive info', async () => {
  const response = await api.post('/login', { email: 'nonexistent@test.com', password: 'wrong' });
  expect(response.body.error).toBe('Invalid credentials'); // Generic message
});

test('sensitive endpoints not exposed', async () => {
  const endpoints = ['/debug', '/.env', '/.git', '/admin'];
  for (let ep of endpoints) {
    expect((await fetch(`https://example.com${ep}`)).status).not.toBe(200);
  }
});
```

### 5. Rate Limiting
```javascript
test('rate limiting prevents brute force', async () => {
  const responses = [];
  for (let i = 0; i < 20; i++) {
    responses.push(await api.post('/login', { email: 'test@example.com', password: 'wrong' }));
  }
  expect(responses.filter(r => r.status === 429).length).toBeGreaterThan(0);
});
```

---

## Security Checklist

### Authentication
- [ ] Strong password requirements (12+ chars)
- [ ] Password hashing (bcrypt, scrypt, Argon2)
- [ ] MFA for sensitive operations
- [ ] Account lockout after failed attempts
- [ ] Session ID changes after login
- [ ] Session timeout

### Authorization
- [ ] Check authorization on every request
- [ ] Least privilege principle
- [ ] No horizontal escalation
- [ ] No vertical escalation

### Data Protection
- [ ] HTTPS everywhere
- [ ] Encrypted at rest
- [ ] Secrets not in code/logs
- [ ] PII compliance (GDPR)

### Input Validation
- [ ] Server-side validation
- [ ] Parameterized queries (no SQL injection)
- [ ] Output encoding (no XSS)
- [ ] Rate limiting

---

## CI/CD Integration

```yaml
# GitHub Actions
security-checks:
  steps:
    - name: Dependency audit
      run: npm audit --audit-level=high

    - name: SAST scan
      run: npm run sast

    - name: Secret scan
      uses: trufflesecurity/trufflehog@main

    - name: DAST scan
      if: github.ref == 'refs/heads/main'
      run: docker run owasp/zap2docker-stable zap-baseline.py -t https://staging.example.com
```

**Pre-commit hooks:**
```bash
#!/bin/sh
git-secrets --scan
npm run lint:security
```

---

## Agent-Assisted Security Testing

```typescript
// Comprehensive multi-layer scan
await Task("Security Scan", {
  target: 'src/',
  layers: { sast: true, dast: true, dependencies: true, secrets: true },
  severity: ['critical', 'high', 'medium']
}, "qe-security-scanner");

// OWASP Top 10 testing
await Task("OWASP Scan", {
  categories: ['broken-access-control', 'injection', 'cryptographic-failures'],
  depth: 'comprehensive'
}, "qe-security-scanner");

// Validate fix
await Task("Validate Fix", {
  vulnerability: 'CVE-2024-12345',
  expectedResolution: 'upgrade package to v2.0.0',
  retestAfterFix: true
}, "qe-security-scanner");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/security/
├── scans/*           - Scan results
├── vulnerabilities/* - Found vulnerabilities
├── fixes/*           - Remediation tracking
└── compliance/*      - Compliance status
```

### Fleet Coordination
```typescript
const securityFleet = await FleetManager.coordinate({
  strategy: 'security-testing',
  agents: [
    'qe-security-scanner',
    'qe-api-contract-validator',
    'qe-quality-analyzer',
    'qe-deployment-readiness'
  ],
  topology: 'parallel'
});
```

---

## Common Mistakes

### ❌ Security by Obscurity
Hiding admin at `/super-secret-admin` → **Use proper auth**

### ❌ Client-Side Validation Only
JavaScript validation can be bypassed → **Always validate server-side**

### ❌ Trusting User Input
Assuming input is safe → **Sanitize, validate, escape all input**

### ❌ Hardcoded Secrets
API keys in code → **Environment variables, secret management**

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - Security with agents
- [api-testing-patterns](../api-testing-patterns/) - API security testing
- [compliance-testing](../compliance-testing/) - GDPR, HIPAA, SOC2

---

## Remember

**Think like an attacker:** What would you try to break? Test that.
**Build like a defender:** Assume input is malicious until proven otherwise.
**Test continuously:** Security testing is ongoing, not one-time.

**With Agents:** Agents automate vulnerability scanning, track remediation, and validate fixes. Use agents to maintain security posture at scale.
