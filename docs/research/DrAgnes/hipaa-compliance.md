# DrAgnes HIPAA Compliance Strategy

**Status**: Research & Planning
**Date**: 2026-03-21

## Overview

DrAgnes operates at the intersection of medical imaging, AI classification, and collective intelligence. This document defines the comprehensive strategy for HIPAA compliance, FDA considerations, and privacy engineering that ensures patient data is protected at every layer while still enabling practice-adaptive and collective learning.

## Regulatory Framework

### HIPAA (Health Insurance Portability and Accountability Act)

DrAgnes must comply with:
- **Privacy Rule** (45 CFR 164.500-534): Governs use and disclosure of PHI
- **Security Rule** (45 CFR 164.302-318): Technical, administrative, and physical safeguards
- **Breach Notification Rule** (45 CFR 164.400-414): Notification within 60 days
- **HITECH Act**: Enhanced penalties, breach notification to HHS for 500+ records

### FDA Considerations

DrAgnes functions as a Clinical Decision Support (CDS) tool. Under FDA guidance on Clinical Decision Support Software (2022 final guidance):

**Criteria for Non-Device CDS (all four must be met)**:
1. Not intended to acquire, process, or analyze a medical image -- **DrAgnes processes dermoscopic images, so this criterion is NOT met**
2. Displays/analyzes but does not replace clinician judgment
3. Intended for healthcare professionals
4. Provides basis for understanding the recommendation

**Conclusion**: DrAgnes likely falls under FDA regulation as a Software as a Medical Device (SaMD). The classification depends on the intended use:
- **Class II (510(k))**: If positioned as an aid to dermatologists (not standalone diagnosis)
- **Class III (PMA)**: If positioned as a screening/diagnostic tool for non-specialists

**Recommended Regulatory Path**: Class II 510(k) with predicate device comparison to 3Derm (DEN200069, FDA-cleared AI for skin cancer detection). Position DrAgnes as a clinical decision support tool that assists qualified dermatologists.

### FDA 21 CFR 820 (Quality System Regulation)

If pursuing FDA clearance:
- **Design Controls** (820.30): Design input, output, review, verification, validation
- **Software Validation** (820.70(i)): Per FDA guidance on General Principles of Software Validation
- **SOUP Documentation**: Software of Unknown Provenance (MobileNetV3 architecture, pre-trained weights)
- **Risk Management**: ISO 14971 risk analysis for AI/ML components
- **Post-Market Surveillance**: Monitoring model performance drift in production

## PHI Handling Architecture

### What Constitutes PHI in DrAgnes

| Data Element | PHI? | Handling |
|-------------|------|----------|
| Dermoscopic image (raw) | Yes (biometric) | Never leaves device. Stored in IndexedDB, encrypted |
| Patient name | Yes | Never stored in DrAgnes. Linked via EHR only |
| Date of birth | Yes | Converted to age decade (30s, 40s, ...) before any processing |
| MRN / Chart number | Yes | Never stored. External reference only via EHR integration |
| Classification result | Potentially | De-identified before brain submission |
| CNN embedding (576-dim) | No* | Non-invertible. Cannot reconstruct image from embedding |
| ABCDE scores | No* | Aggregated metrics, not identifiable |
| Body location | Potentially | Generalized to category (trunk, extremity, head) |
| Fitzpatrick skin type | No | Population-level demographic, not individually identifying |
| GPS coordinates | Yes | Stripped from EXIF before any processing |
| Device serial number | Yes (indirect) | Stripped from EXIF metadata |
| Clinician notes (free text) | Yes | NLP-based PII detection before any storage/sharing |

*When combined, these elements could potentially be re-identifying. k-anonymity (k>=5) is enforced on all combinations.

### The "No Raw Image" Principle

The foundational privacy guarantee of DrAgnes:

```
RAW IMAGE ──▶ CNN ──▶ EMBEDDING ──▶ BRAIN
    │                      │
    │                      └── Non-invertible: cannot reconstruct image
    │                          from 576-dim float vector
    │
    └── NEVER LEAVES DEVICE
        - Stored in IndexedDB (encrypted)
        - Processed locally (WASM CNN)
        - Displayed locally only
        - Deleted per retention policy
```

**Mathematical basis for non-invertibility**: MobileNetV3 Small maps a 224x224x3 = 150,528-dimensional input to a 576-dimensional embedding. This is a 261:1 dimensionality reduction. The mapping is many-to-one (infinite input images map to the same embedding). No computational technique can invert this mapping to recover the original image.

### PII Stripping Pipeline

Leverages the existing brain server's redaction infrastructure:

```
Input Record
    │
    ▼
Stage 1: EXIF Sanitization
    ├── Remove GPS coordinates
    ├── Remove device serial number
    ├── Remove camera make/model (keep DermLite type only)
    ├── Remove software version strings
    └── Remove timestamp (replace with date-only, bucketed to week)
    │
    ▼
Stage 2: Demographic Generalization
    ├── Age → decade bucket (20, 30, 40, ...)
    ├── Body location → category (head, trunk, upper_extremity, lower_extremity)
    ├── Gender → removed (not clinically necessary for classification)
    └── Ethnicity → Fitzpatrick scale only (I-VI)
    │
    ▼
Stage 3: Free Text Scrubbing
    ├── Named entity recognition (NER) for person names
    ├── Pattern matching for MRN, SSN, phone, email, address
    ├── Date normalization (remove exact dates, keep relative)
    └── Organization name redaction
    │
    ▼
Stage 4: k-Anonymity Enforcement
    ├── Group by (Fitzpatrick, age_decade, body_location_category)
    ├── Suppress groups with fewer than k=5 members
    └── Generalize further if needed to achieve k-anonymity
    │
    ▼
Stage 5: Differential Privacy
    ├── Laplace noise to continuous values (epsilon=1.0)
    ├── Randomized response for binary features
    └── Privacy budget tracking (per practice, per epoch)
    │
    ▼
Clean Record (ready for brain submission)
```

### Differential Privacy Implementation

**Mechanism**: Laplace mechanism with epsilon=1.0 (matching brain server's current configuration).

```
For each continuous value v with sensitivity Δ:
    v_noisy = v + Laplace(0, Δ/epsilon)

For embeddings (576-dim vector):
    Each dimension independently noised
    Sensitivity calibrated per-dimension from training data
    epsilon budget split across dimensions: epsilon_per_dim = epsilon / sqrt(576) ≈ 0.042
```

**Privacy Budget Tracking**:
- Each practice has an annual privacy budget (epsilon_total = 10.0)
- Each brain contribution costs epsilon=1.0
- Budget resets annually
- When budget exhausted, contributions are aggregated locally until reset
- Brain server tracks global dp_budget_used (currently 1.0)

### Witness Chain Audit Trail

Every DrAgnes classification carries a cryptographic provenance chain:

```
Witness Chain Structure:
    [0..31]  = Previous witness hash (or zeros for genesis)
    [32..63] = SHAKE-256(
                  model_version ||
                  brain_epoch ||
                  input_embedding_hash ||
                  classification_output ||
                  clinician_id_hash ||
                  timestamp
               )
    [64..N]  = Chain continuation
```

**Audit capabilities**:
- Verify which model version produced a classification
- Verify the brain state at classification time
- Detect if a classification has been tampered with
- Reconstruct the full decision chain for regulatory review
- Prove temporal ordering of classifications

## Technical Safeguards (Security Rule)

### Access Controls (164.312(a))

| Control | Implementation |
|---------|---------------|
| Unique user identification | OAuth 2.0 with Google Identity Platform |
| Emergency access | Break-glass procedure with audit logging |
| Automatic logoff | 15-minute session timeout, token refresh required |
| Encryption | AES-256-GCM at rest, TLS 1.3 in transit |
| Role-based access | Admin, Clinician, Technician, Viewer roles |
| Multi-factor authentication | Required for all clinician accounts |

### Audit Controls (164.312(b))

| Audit Event | Data Captured |
|-------------|--------------|
| Image capture | Timestamp, device, user, body location |
| Classification run | Timestamp, model version, brain epoch, user |
| Brain contribution | Timestamp, de-identification confirmation, witness hash |
| Brain search | Timestamp, query type, result count |
| Record access | Timestamp, user, record ID, access type |
| Export | Timestamp, user, data scope, format |
| Failed login | Timestamp, user identifier, IP, reason |

**Retention**: Audit logs retained for 6 years (HIPAA minimum) in append-only Cloud Logging with CMEK encryption.

### Integrity Controls (164.312(c))

- All data at rest uses AES-256-GCM with Google Cloud CMEK
- All witness chains are append-only (SHAKE-256, tamper-evident)
- Database writes use Firestore transactions (ACID)
- Model weight integrity verified via SHA-256 checksums before inference
- WASM module integrity verified via Subresource Integrity (SRI) hashes

### Transmission Security (164.312(e))

- TLS 1.3 required for all connections (no fallback)
- Certificate pinning for mobile PWA
- HSTS with 1-year max-age and preloading
- Perfect forward secrecy (ECDHE)
- Brain sync uses authenticated encryption (witness chain verification)

## Administrative Safeguards

### Business Associate Agreement (BAA)

**Required BAAs**:
| Entity | Role | BAA Status |
|--------|------|-----------|
| Google Cloud Platform | Infrastructure provider | Google Cloud BAA available (standard) |
| DermLite / 3Gen Inc. | Hardware manufacturer | Not required (no PHI exchange) |
| Practice using DrAgnes | Covered entity | BAA with DrAgnes operator required |
| PubMed / NCBI | Literature source | Not required (public data) |

**Google Cloud BAA Coverage**:
Google Cloud's BAA covers Cloud Run, Firestore, GCS, Pub/Sub, Cloud Logging, Secret Manager, and Cloud KMS -- all services used by DrAgnes.

### Workforce Training

- All personnel with access to DrAgnes infrastructure must complete HIPAA training annually
- Security awareness training quarterly
- Incident response drills semi-annually
- Role-specific training for developers handling PHI-adjacent code

### Incident Response Plan

```
Incident Detection
    │
    ├── Automated: Cloud Monitoring alerts, anomaly detection
    ├── Manual: User reports, security team discovery
    │
    ▼
Assessment (within 1 hour)
    ├── Determine if PHI was involved
    ├── Classify severity (1-4)
    ├── Identify affected individuals
    │
    ▼
Containment (within 4 hours)
    ├── Isolate affected systems
    ├── Revoke compromised credentials
    ├── Preserve forensic evidence
    │
    ▼
Notification (within 60 days per HIPAA)
    ├── Individual notification if PHI compromised
    ├── HHS notification if 500+ individuals affected (within 60 days)
    ├── Media notification if 500+ in single state
    ├── State attorney general notification (varies by state)
    │
    ▼
Remediation
    ├── Root cause analysis
    ├── System hardening
    ├── Policy updates
    └── Post-incident review
```

### Data Retention Policy

| Data Type | Retention | Location | Justification |
|-----------|----------|----------|---------------|
| Raw dermoscopic images | Per practice policy (default 7 years) | Device only (IndexedDB) | Clinical record retention |
| CNN embeddings (local) | Same as images | Device only | Tied to image lifecycle |
| Brain contributions | Indefinite (de-identified) | GCS / Firestore | Research value, non-PHI |
| Audit logs | 6 years | Cloud Logging | HIPAA minimum |
| Model weights | Indefinite | GCS | Reproducibility |
| Classification results | Per practice policy | Device + Firestore | Clinical record |
| Clinician feedback | Indefinite (de-identified) | Firestore | Model improvement |

## Risk Assessment

### HIPAA Risk Analysis (164.308(a)(1))

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Raw image exfiltration | Low | Critical | Images never leave device; no upload API exists |
| Re-identification from embeddings | Very Low | High | 261:1 dimensionality reduction; k-anonymity; DP noise |
| Model inversion attack | Very Low | High | MobileNetV3 is many-to-one; DP noise prevents gradient-based inversion |
| Insider threat (developer) | Low | High | No production access to PHI; all PHI stays on device |
| Cloud infrastructure breach | Low | Medium | Only de-identified data in cloud; CMEK encryption |
| Man-in-the-middle | Very Low | High | TLS 1.3 + certificate pinning |
| Malicious model update | Low | High | Model checksums + witness chain verification |
| Session hijacking | Low | Medium | Short session timeout; MFA; secure cookies |

### FDA Risk Analysis (ISO 14971)

| Hazard | Severity | Probability | Risk Level | Mitigation |
|--------|----------|------------|------------|------------|
| False negative (missed melanoma) | Critical | Medium | High | >95% sensitivity target; always recommend dermatologist review |
| False positive (unnecessary biopsy) | Moderate | Medium | Medium | >85% specificity; clinical decision support, not standalone |
| Model drift (accuracy degradation) | Serious | Low | Medium | Brain drift monitoring; automated retraining triggers |
| Bias against skin types | Serious | Medium | High | Fitzpatrick-stratified evaluation; diverse training data |
| System unavailability | Minor | Low | Low | Offline-first architecture; no dependency on connectivity |

## International Considerations

While DrAgnes targets US deployment first, the architecture supports international compliance:

| Regulation | Region | Key Requirement | DrAgnes Approach |
|-----------|--------|-----------------|-----------------|
| GDPR | EU | Data minimization, right to erasure | Embeddings are non-invertible; erasure of device data trivial |
| PIPEDA | Canada | Consent, purpose limitation | Explicit consent workflow; purpose-bound data processing |
| LGPD | Brazil | Data protection officer, consent | DPO appointment; consent management |
| POPIA | South Africa | Processing limitation | Minimal data collection; de-identification |
| MDR 2017/745 | EU | Medical device regulation | CE marking pathway if EU deployment |
| PMDA | Japan | Pharmaceutical and medical device regulation | J-PMDA approval pathway |

## Compliance Monitoring

### Continuous Compliance Dashboard

```
DrAgnes Compliance Dashboard
    │
    ├── Privacy Budget Status
    │       ├── Per-practice epsilon consumption
    │       ├── Global DP budget (currently 1.0 used)
    │       └── Budget exhaustion forecast
    │
    ├── Access Audit
    │       ├── Login frequency by role
    │       ├── Failed login attempts
    │       ├── Anomalous access patterns
    │       └── Break-glass usage
    │
    ├── Data Flow Verification
    │       ├── Confirmation: zero raw images in cloud
    │       ├── PII stripping success rate (target: 100%)
    │       ├── k-anonymity compliance rate
    │       └── Witness chain integrity checks
    │
    ├── Model Governance
    │       ├── Current model version across practices
    │       ├── Drift detection alerts
    │       ├── Fairness metrics by Fitzpatrick type
    │       └── Sensitivity/specificity by subgroup
    │
    └── Incident Tracker
            ├── Open incidents
            ├── Time to resolution
            ├── Breach notification status
            └── Corrective action tracking
```
