# Security Policy

## Reporting a Vulnerability

We take the security of RuVector and the π collective (`pi.ruv.io`) seriously and
appreciate responsible disclosure.

**Please do not open a public issue for security vulnerabilities.** Public issues
disclose the flaw before a fix is available and put users at risk.

Instead, report privately through either channel:

1. **GitHub Private Vulnerability Reporting (preferred).**
   Use the **"Report a vulnerability"** button under the repository's
   [Security tab](https://github.com/ruvnet/ruvector/security/advisories/new).
   This opens a private advisory visible only to you and the maintainers.

2. **Email.** Send details to **ruv@ruv.net**. Encrypt with the maintainer's
   public key on request. Use a subject line beginning with `[SECURITY]`.

### What to include

A useful report contains:

- The affected component and version/commit (e.g. a crate under `crates/`, an
  npm package under `npm/packages/`, or a `pi.ruv.io` endpoint).
- A clear description of the impact (what an attacker can do).
- A minimal, reproducible proof-of-concept, or precise steps to reproduce.
- Any relevant logs, payloads, or configuration.

The more concrete and reproducible the report, the faster we can verify and fix it.

### Our commitment

- We aim to **acknowledge** your report within **3 business days**.
- We will provide an initial **assessment** within **10 business days**.
- We will keep you informed of remediation progress and coordinate a disclosure
  timeline with you. We are happy to credit reporters who wish to be named.

## Scope

In scope:

- The RuVector Rust crates (`crates/`) and published npm packages
  (`npm/packages/`).
- The `pi.ruv.io` / mcp-brain-server API and its data-handling
  (provenance, differential privacy, witness chains).
- Cryptographic, memory-safety, authentication, authorization, injection,
  data-poisoning, and denial-of-service issues.

Out of scope:

- Reports without a demonstrable security impact or a reproducible
  proof-of-concept.
- Findings in third-party dependencies — please report those upstream (we will
  still help coordinate if RuVector is affected).
- Social engineering, physical attacks, and volumetric DDoS.

## Supported versions

Security fixes are applied to the latest released version of each package and
crate. We do not backport fixes to older majors unless a fix is trivial and the
older line is still in wide use.

Thank you for helping keep RuVector and its users safe.
