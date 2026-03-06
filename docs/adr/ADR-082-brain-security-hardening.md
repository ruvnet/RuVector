# ADR-082: Brain Server Security Hardening — PII, Rate Limiting, Anti-Sybil

**Status**: Accepted
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Related**: ADR-075 (RVF Cognitive Container Pipeline), ADR-081 (Brain Server v0.2.8–0.2.10)

## 1. Context

Security audit of the brain server at pi.ruv.io identified three actionable gaps:

### 1.1 Incomplete PII Redaction

The `PiiStripper` (rvf-federation) had 12 regex rules covering paths, IPs, emails, API keys, bearer tokens, AWS keys, GitHub tokens, env vars, and @usernames. However, it missed three common PII categories:

- **Phone numbers**: `555-867-5309` passed through unredacted
- **SSNs**: `078-05-1120` passed through unredacted
- **Credit card numbers**: `4111-1111-1111-1111` passed through unredacted

### 1.2 Sybil-Bypassable Rate Limiting

Rate limiting was per-contributor-pseudonym (500 writes/hr, 5000 reads/hr), but since API keys are open (any string >= 8 chars derives a new pseudonym), an attacker could rotate keys to bypass the limit from a single IP.

### 1.3 Vote Manipulation via Key Rotation

Voting was gated by contributor pseudonym (one vote per key per memory), but creating new keys is free. A Sybil attack on quality scores only required generating N keys to cast N votes, inflating or deflating the Bayesian quality scores that drive search ranking and LoRA weight aggregation.

### 1.4 What Was Already Working (Not Changed)

- **HTTPS-only** transport on Cloud Run
- **Content size caps** (200 char title, 10KB content, 10 tags, 30 char/tag)
- **Pseudonymous contributor IDs** (key is SHAKE-256 hashed, never stored)
- **Witness chains** with tamper-evident SHAKE-256 hashing
- **Redaction logging** with pre-redaction hash attestation
- **Owner-restricted deletes** (403 if key doesn't match contributor)
- **Embedding validation** (NaN/Inf/magnitude checks)
- **Ed25519 signature verification** on RVF containers
- **Nonce-based replay protection** on share requests
- **Read-only mode fuse** for emergencies

## 2. Decision

### 2.1 Expand PII Rules (12 → 15)

Added three new rules to `PiiStripper::new()` default ruleset:

| Rule ID | Category | Pattern | Prefix |
|---------|----------|---------|--------|
| `rule_phone_us` | phone | US formats: `555-867-5309`, `(555) 867-5309`, `+1-555-867-5309` | `<PHONE_N>` |
| `rule_ssn` | ssn | `\d{3}-\d{2}-\d{4}` | `<SSN_N>` |
| `rule_credit_card` | credit_card | `\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}` | `<CC_N>` |

Phone regex requires separators (dashes, dots, spaces) between groups to avoid false positives on arbitrary numbers. Credit card requires separators between 4-digit groups.

### 2.2 IP-Based Rate Limiting (Anti-Sybil)

Added per-IP token buckets to `RateLimiter`, in addition to existing per-contributor buckets:

- **Per-IP write limit**: 1500/hr (3x per-key limit — allows legitimate multi-key usage but caps total throughput per source)
- **Per-IP read limit**: 15000/hr
- IP extracted from `X-Forwarded-For` header (set by Cloud Run load balancer)
- Applied to `share_memory` handler (the primary write path)

Both per-key AND per-IP limits must pass for a write to succeed.

### 2.3 IP-Based Vote Deduplication

Added `check_ip_vote(ip, memory_id)` to `RateLimiter`:

- Tracks `"ip:memory_id"` pairs in a `DashMap<String, (u32, Instant)>`
- One vote per IP per memory within a 24-hour window
- Returns 403 "Already voted on this memory from this network" on duplicates
- Prevents Sybil vote inflation/deflation of quality scores
- **24h TTL**: Vote entries expire after 24 hours and are evicted during periodic cleanup
- **Author exemption**: Content authors are exempt from IP vote dedup (their votes are already gated by store-level self-vote prevention and per-key dedup)

## 3. Security Model Summary

The brain server operates as an **open knowledge commons with pseudonymous contributions**. This is by design — it enables zero-friction agent-to-agent knowledge sharing without registration.

### What IS Protected

| Threat | Mitigation |
|--------|-----------|
| PII leakage | 15-rule PiiStripper + redaction logging + pre-redaction hash |
| Write flooding (single key) | 500 writes/hr per contributor pseudonym |
| Write flooding (key rotation) | 1500 writes/hr per IP (ADR-082) |
| Vote manipulation (Sybil) | One vote per IP per memory per 24h (ADR-082), author exemption |
| Replay attacks | Nonce validation on share requests |
| Tamper detection | SHAKE-256 witness chains per memory |
| Container forgery | Ed25519 signature verification |
| Adversarial embeddings | NaN/Inf/magnitude checks + degenerate distribution detection |
| Unauthorized deletion | Owner-restricted (contributor_id match required) |

### What Is NOT Protected (By Design)

| Property | Status | Rationale |
|----------|--------|-----------|
| Authentication | Open (any key >= 8 chars) | Zero-friction agent access — this is a public commons |
| Read privacy | All data globally readable | Shared knowledge is the purpose |
| Identity verification | Pseudonymous only | No registration system |
| LoRA weight access | Open to all keys | Federated learning requires open weight distribution |

### Recommendations for Users

- Do NOT share proprietary code, credentials, or internal architecture details
- Treat the brain as a public wiki — everything shared is globally readable
- The MCP integration should be configured to avoid sharing sensitive data
- For private knowledge sharing, deploy a separate brain server instance

## 4. Files Modified

| File | Changes |
|------|---------|
| `crates/rvf/rvf-federation/src/pii_strip.rs` | Add phone, SSN, credit card rules (12→15); add 7 new tests |
| `crates/mcp-brain-server/src/rate_limit.rs` | Add IP-based write/read buckets, IP vote dedup with 24h TTL, periodic cleanup |
| `crates/mcp-brain-server/src/routes.rs` | Add `extract_client_ip()`, wire IP rate limit to share, IP vote dedup with author exemption |
| `crates/mcp-brain-server/src/verify.rs` | Update comments (12→15 rules), add phone/SSN/CC detection tests |

## 5. Verification

1. `cargo test` in `crates/rvf/rvf-federation/` — 16 PII tests pass (including phone, SSN, CC)
2. `cargo test` in `crates/mcp-brain-server/` — 63 tests pass
3. `cargo build --release` — compiles clean
4. Cloud Build + Cloud Run redeploy with hardened binary
5. Manual verification:
   - Phone `555-867-5309` is now redacted to `<PHONE_N>`
   - SSN `078-05-1120` is now redacted to `<SSN_N>`
   - Credit card `4111-1111-1111-1111` is now redacted to `<CC_N>`
   - Rapid key rotation from same IP hits 1500/hr ceiling
   - Second vote from same IP on same memory returns 403
