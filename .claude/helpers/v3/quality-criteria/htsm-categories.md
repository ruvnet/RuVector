# HTSM v6.3 Quality Criteria Categories

James Bach's Heuristic Test Strategy Model (HTSM) v6.3 Quality Criteria framework.

## 1. Capability
**Can it perform the required functions?**

| Subcategory | Focus |
|-------------|-------|
| Sufficiency | Does it do what it's supposed to? |
| Correctness | Does it do it correctly? |

**Priority Indicators:**
- P0: Core business functionality
- P1: Important features
- P2: Secondary features
- P3: Nice-to-have features

## 2. Reliability
**Will it work well and resist failure?**

| Subcategory | Focus |
|-------------|-------|
| Robustness | Can it handle adverse conditions? |
| Error Handling | Does it handle errors gracefully? |
| Data Integrity | Is data protected from corruption? |
| Safety | Does it avoid dangerous behaviors? |

**Cannot be omitted** - All systems can fail.

## 3. Usability
**How easy is it for real users?**

| Subcategory | Focus |
|-------------|-------|
| Learnability | How quickly can users learn? |
| Operability | How easy to operate day-to-day? |
| Accessibility | Can users with disabilities use it? |

## 4. Charisma
**How appealing is the product?**

| Subcategory | Focus |
|-------------|-------|
| Aesthetics | Is it visually pleasing? |
| Uniqueness | Does it stand out? |
| Entrancement | Does it engage users? |
| Image | Does it project the right brand? |

**Note:** "Brand guidelines handled separately" is NOT a valid omission reason. Charisma is about UX testing, not brand documentation.

## 5. Security
**How well protected against unauthorized use?**

| Subcategory | Focus |
|-------------|-------|
| Authentication | Who is using it? |
| Authorization | What are they allowed to do? |
| Privacy | Is personal data protected? |
| Security Holes | Are there vulnerabilities? |

**Cannot be omitted** - Every system has attack surface.

## 6. Scalability
**How well does deployment scale?**

| Subcategory | Focus |
|-------------|-------|
| Load Handling | Behavior under increased demand |
| Resource Efficiency | Resource usage at scale |

## 7. Compatibility
**Works with external components?**

| Subcategory | Focus |
|-------------|-------|
| Application | Works with other applications? |
| OS | Works with target operating systems? |
| Hardware | Works with target hardware? |
| Backward | Works with previous versions? |
| Product Footprint | Resource requirements acceptable? |

## 8. Performance
**How speedy and responsive?**

| Subcategory | Focus |
|-------------|-------|
| Response Time | Under various conditions |
| Throughput | Data processing capacity |
| Efficiency | Resource utilization |

**Cannot be omitted** - Every system has response time.

## 9. Installability
**How easily installed?**

| Subcategory | Focus |
|-------------|-------|
| System Requirements | Clear and achievable? |
| Configuration | Easy to configure? |
| Uninstallation | Clean removal? |
| Upgrades/Patches | Easy to update? |
| Administration | Easy to administer? |

**Valid omission:** Pure SaaS/browser-based with no client installation.

## 10. Development
**How well can we create/test/modify?**

| Subcategory | Focus |
|-------------|-------|
| Supportability | Easy to support? |
| Testability | Easy to test? |
| Maintainability | Easy to maintain? |
| Portability | Easy to port? |
| Localizability | Easy to localize? |

**Cannot be omitted** - Always applies to software.

---

## Priority Assignment Guide

| Priority | Definition | Example |
|----------|------------|---------|
| **P0 (Critical)** | Failure causes immediate business/user harm | Payment failures, data breaches |
| **P1 (High)** | Critical to core user value proposition | Core features not working |
| **P2 (Medium)** | Affects satisfaction but not blocking | Secondary features |
| **P3 (Low)** | Nice-to-have improvements | Polish, edge case optimization |

## Valid vs Invalid Omission Reasons

| Category | Valid Omission | Invalid Omission |
|----------|----------------|------------------|
| Installability | "Pure SaaS, no client installation" | "Handled by ops team" |
| Charisma | "CLI tool, visual design N/A" | "Brand guidelines separate" |
| Compatibility | "Single-platform by contract" | "Will test on main browsers" |
| Development | **NEVER** | "Team is experienced" |
| Security | **NEVER** | "Internal system only" |
