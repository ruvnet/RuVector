# ADR-007: Differential Privacy and Epsilon-Budget Management

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-13 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-004-rvf-format, ADR-003-mcp-protocol |

## 1. Context

### 1.1 The Collective Learning Dilemma

The Pi Brain (shared collective intelligence) enables agents to:
- Share learned patterns and knowledge
- Benefit from collective experience
- Build on each other's discoveries

However, this creates privacy risks:
- Individual contributions may leak sensitive information
- Query patterns could reveal user behavior
- Aggregated knowledge might expose private data

### 1.2 What is Differential Privacy?

Differential Privacy (DP) is a mathematical framework guaranteeing that:

> The inclusion or exclusion of any single record has minimal effect on the output of a query.

Formally, a mechanism M satisfies (epsilon, delta)-differential privacy if:

```
P[M(D) in S] <= exp(epsilon) * P[M(D') in S] + delta
```

For all datasets D, D' differing in one record, and all outputs S.

### 1.3 Privacy Budget Concepts

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| **Epsilon (e)** | Privacy loss | Lower = more private |
| **Delta (d)** | Failure probability | Probability of catastrophic leak |
| **Sensitivity** | Delta_f | Maximum change from one record |

**Epsilon Interpretation:**
- e = 0: Perfect privacy (useless output)
- e = 0.1: Very strong privacy
- e = 1: Strong privacy (recommended)
- e = 10: Weak privacy (nearly raw data)

## 2. Decision

### 2.1 Privacy-Preserving Brain with Budget Management

We integrate differential privacy at multiple levels:

```
              Raw Contribution
                    |
                    v
            +---------------+
            | PII Stripping |  <- Remove identifying info
            +---------------+
                    |
                    v
            +---------------+
            | Embedding DP  |  <- Add noise to embeddings
            +---------------+
                    |
                    v
            +---------------+
            | Aggregation   |  <- Secure aggregation
            +---------------+
                    |
                    v
            +---------------+
            | Budget Mgmt   |  <- Track privacy spend
            +---------------+
                    |
                    v
              Shared Brain
```

### 2.2 Privacy Mechanisms

#### 2.2.1 Gaussian Mechanism for Embeddings

Add calibrated Gaussian noise to embeddings before sharing:

```rust
pub struct GaussianMechanism {
    epsilon: f32,
    delta: f32,
    sensitivity: f32,  // L2 sensitivity of embeddings
}

impl GaussianMechanism {
    /// Compute noise scale for (epsilon, delta)-DP
    /// sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    pub fn noise_scale(&self) -> f32 {
        let ln_term = (1.25 / self.delta).ln();
        (2.0 * ln_term).sqrt() * self.sensitivity / self.epsilon
    }

    /// Apply DP noise to embedding
    pub fn privatize(&self, embedding: &mut [f32], rng: &mut impl Rng) {
        let sigma = self.noise_scale();
        let noise_dist = Normal::new(0.0, sigma).unwrap();

        for x in embedding.iter_mut() {
            *x += noise_dist.sample(rng);
        }

        // Renormalize to unit sphere (for cosine similarity)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }
}
```

#### 2.2.2 Contribution Clipping

Bound the influence of any single contribution:

```rust
pub struct ContributionClipper {
    max_norm: f32,      // Maximum L2 norm
    max_count: usize,   // Maximum contributions per contributor
}

impl ContributionClipper {
    /// Clip embedding to bounded L2 norm
    pub fn clip(&self, embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > self.max_norm {
            let scale = self.max_norm / norm;
            for x in embedding.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Check contribution quota
    pub fn check_quota(&self, contributor: &str, store: &QuotaStore) -> bool {
        store.get_count(contributor) < self.max_count
    }
}
```

#### 2.2.3 PII Detection and Stripping

Detect and remove personally identifiable information before embedding:

```rust
pub struct PiiDetector {
    patterns: Vec<Regex>,
    sensitive_keywords: HashSet<String>,
}

impl PiiDetector {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                // Email addresses
                Regex::new(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b").unwrap(),
                // Social Security Numbers (US)
                Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
                // Credit card numbers
                Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap(),
                // IP addresses
                Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
                // Phone numbers (various formats)
                Regex::new(r"\b(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b").unwrap(),
                // API keys (common patterns)
                Regex::new(r"\b(sk|pk|api|key|token)[_-]?[a-zA-Z0-9]{16,}\b").unwrap(),
            ],
            sensitive_keywords: ["password", "secret", "api_key", "token", "credential"]
                .iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Detect PII in text content
    pub fn detect(&self, text: &str) -> Vec<PiiMatch> {
        let mut matches = Vec::new();

        for (i, pattern) in self.patterns.iter().enumerate() {
            for m in pattern.find_iter(text) {
                matches.push(PiiMatch {
                    pii_type: PiiType::from_index(i),
                    start: m.start(),
                    end: m.end(),
                    content: m.as_str().to_string(),
                });
            }
        }

        matches
    }

    /// Strip PII from text
    pub fn strip(&self, text: &str) -> String {
        let mut result = text.to_string();

        for pattern in &self.patterns {
            result = pattern.replace_all(&result, "[REDACTED]").to_string();
        }

        result
    }
}

#[derive(Debug, Clone)]
pub enum PiiType {
    Email,
    Ssn,
    CreditCard,
    IpAddress,
    PhoneNumber,
    ApiKey,
    Unknown,
}
```

### 2.3 Privacy Budget Accountant

#### 2.3.1 Budget Tracking

Track cumulative privacy loss using composition theorems:

```rust
pub struct PrivacyBudgetAccountant {
    total_epsilon: f32,
    total_delta: f32,
    spent_epsilon: AtomicF32,
    spent_delta: AtomicF32,
    composition_type: CompositionType,
    contributor_budgets: DashMap<String, ContributorBudget>,
}

pub enum CompositionType {
    /// Simple composition: epsilon adds linearly
    Basic,
    /// Advanced composition: sqrt(n) * epsilon growth
    Advanced,
    /// Optimal composition using Renyi DP
    RDP,
}

impl PrivacyBudgetAccountant {
    pub fn new(epsilon: f32, delta: f32) -> Self {
        Self {
            total_epsilon: epsilon,
            total_delta: delta,
            spent_epsilon: AtomicF32::new(0.0),
            spent_delta: AtomicF32::new(0.0),
            composition_type: CompositionType::Advanced,
            contributor_budgets: DashMap::new(),
        }
    }

    /// Request budget for an operation
    pub fn request(&self, epsilon: f32, delta: f32) -> Result<PrivacyGrant, PrivacyError> {
        let current_epsilon = self.spent_epsilon.load(Ordering::Acquire);
        let current_delta = self.spent_delta.load(Ordering::Acquire);

        let (new_epsilon, new_delta) = match self.composition_type {
            CompositionType::Basic => {
                // Linear composition: total = sum of individual
                (current_epsilon + epsilon, current_delta + delta)
            }
            CompositionType::Advanced => {
                // Advanced composition theorem
                // e_total <= sqrt(2k * ln(1/d')) * e + k * e * (e^e - 1)
                let k = (current_epsilon / epsilon.max(0.01)) as f32 + 1.0;
                let composed = (2.0 * k * (1.0 / delta).ln()).sqrt() * epsilon;
                (composed, delta)
            }
            CompositionType::RDP => {
                // Renyi DP composition (more sophisticated)
                self.rdp_compose(current_epsilon, epsilon)
                    .unwrap_or((current_epsilon + epsilon, current_delta + delta))
            }
        };

        if new_epsilon > self.total_epsilon || new_delta > self.total_delta {
            return Err(PrivacyError::BudgetExhausted {
                requested: (epsilon, delta),
                remaining: (self.total_epsilon - current_epsilon,
                           self.total_delta - current_delta),
            });
        }

        self.spent_epsilon.store(new_epsilon, Ordering::Release);
        self.spent_delta.store(new_delta, Ordering::Release);

        Ok(PrivacyGrant {
            epsilon,
            delta,
            grant_id: Uuid::new_v4(),
            granted_at: Utc::now(),
        })
    }

    /// Get remaining budget
    pub fn remaining(&self) -> (f32, f32) {
        (
            self.total_epsilon - self.spent_epsilon.load(Ordering::Acquire),
            self.total_delta - self.spent_delta.load(Ordering::Acquire),
        )
    }
}
```

#### 2.3.2 Per-Contributor Budgets

Each contributor has their own privacy budget to limit data exposure:

```rust
pub struct ContributorBudget {
    contributor_id: String,
    daily_epsilon: f32,
    daily_delta: f32,
    spent_today: (f32, f32),
    last_reset: DateTime<Utc>,
}

impl PrivacyBudgetAccountant {
    /// Request budget on behalf of a contributor
    pub fn request_for(
        &self,
        contributor: &str,
        epsilon: f32,
        delta: f32,
    ) -> Result<PrivacyGrant, PrivacyError> {
        let mut budget = self.contributor_budgets
            .entry(contributor.to_string())
            .or_insert_with(|| ContributorBudget {
                contributor_id: contributor.to_string(),
                daily_epsilon: 1.0,  // Default daily budget
                daily_delta: 1e-6,
                spent_today: (0.0, 0.0),
                last_reset: Utc::now(),
            });

        // Reset if new day
        if budget.last_reset.date_naive() != Utc::now().date_naive() {
            budget.spent_today = (0.0, 0.0);
            budget.last_reset = Utc::now();
        }

        if budget.spent_today.0 + epsilon > budget.daily_epsilon {
            return Err(PrivacyError::ContributorBudgetExhausted {
                contributor: contributor.to_string(),
            });
        }

        budget.spent_today.0 += epsilon;
        budget.spent_today.1 += delta;

        // Also deduct from global budget
        self.request(epsilon, delta)
    }
}
```

### 2.4 Secure Aggregation

For federated learning scenarios, use secure aggregation to hide individual contributions:

```rust
pub struct SecureAggregator {
    threshold: usize,        // Minimum participants
    dimension: usize,        // Vector dimension
}

impl SecureAggregator {
    /// Generate pairwise masks for participant
    pub fn generate_masks(
        &self,
        participant_id: &str,
        other_participants: &[String],
        round: u64,
    ) -> HashMap<String, Vec<f32>> {
        let mut masks = HashMap::new();

        for other in other_participants {
            // PRF-based mask generation (deterministic, cancels out)
            let seed = self.derive_shared_seed(participant_id, other, round);
            let mut rng = ChaCha20Rng::from_seed(seed);

            let mask: Vec<f32> = (0..self.dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            // For participant pairs (A, B), A adds mask, B subtracts it
            let sign = if participant_id < other { 1.0 } else { -1.0 };
            masks.insert(other.clone(), mask.iter().map(|m| m * sign).collect());
        }

        masks
    }

    /// Apply masks to contribution
    pub fn mask_contribution(
        &self,
        contribution: &[f32],
        masks: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let mut masked = contribution.to_vec();

        for mask in masks.values() {
            for (m, mask_val) in masked.iter_mut().zip(mask.iter()) {
                *m += mask_val;
            }
        }

        masked
    }

    /// Aggregate masked contributions (masks cancel out)
    pub fn aggregate(&self, masked_contributions: &[Vec<f32>]) -> Vec<f32> {
        let n = masked_contributions.len();
        let dim = masked_contributions[0].len();

        let mut sum = vec![0.0f32; dim];
        for contribution in masked_contributions {
            for (s, c) in sum.iter_mut().zip(contribution) {
                *s += c;
            }
        }

        // Average
        for s in &mut sum {
            *s /= n as f32;
        }

        sum
    }
}
```

## 3. Privacy-Utility Tradeoffs

### 3.1 Noise Calibration

| Epsilon | Noise Level | Utility Impact | Use Case |
|---------|-------------|----------------|----------|
| 0.1 | Very High | ~30% accuracy loss | Highly sensitive (medical) |
| 0.5 | High | ~15% accuracy loss | Sensitive (financial) |
| 1.0 | Moderate | ~10% accuracy loss | **Default** |
| 3.0 | Low | ~3% accuracy loss | Less sensitive |
| 10.0 | Minimal | ~0.5% accuracy loss | Public data |

### 3.2 Utility Preservation Techniques

1. **Subsampling**: Apply DP to random subset (privacy amplification)
   - If you sample q fraction, epsilon_effective = q * epsilon

2. **Local Hashing**: Randomized response before aggregation

3. **Adaptive Clipping**: Learn optimal clip threshold from public data

4. **Gradient Compression**: Communicate only top-k gradients

## 4. Implementation

### 4.1 Private Brain API

```rust
pub struct PrivateBrain {
    brain: Brain,
    privacy: PrivacyBudgetAccountant,
    pii_detector: PiiDetector,
    gaussian: GaussianMechanism,
    clipper: ContributionClipper,
}

impl PrivateBrain {
    /// Share knowledge with privacy guarantees
    pub fn share(&self, contribution: Contribution) -> Result<ShareResult, BrainError> {
        // 1. Check PII
        let pii_matches = self.pii_detector.detect(&contribution.content);
        if !pii_matches.is_empty() {
            return Err(BrainError::PiiDetected { matches: pii_matches });
        }

        // 2. Request privacy budget
        let grant = self.privacy.request_for(
            &contribution.contributor,
            contribution.requested_epsilon(),
            contribution.requested_delta(),
        )?;

        // 3. Generate embedding
        let mut embedding = self.brain.embed(&contribution.content)?;

        // 4. Clip contribution
        self.clipper.clip(&mut embedding);

        // 5. Privatize embedding
        self.gaussian.privatize(&mut embedding, &mut thread_rng());

        // 6. Store with privatized embedding
        self.brain.store(contribution.with_embedding(embedding))
    }

    /// Search (no privacy cost for public aggregated data)
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Memory>, BrainError> {
        self.brain.search(query, limit)
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self, contributor: &str) -> (f32, f32) {
        self.privacy.contributor_budgets
            .get(contributor)
            .map(|b| (b.daily_epsilon - b.spent_today.0, b.daily_delta - b.spent_today.1))
            .unwrap_or((1.0, 1e-6))
    }
}
```

## 5. Consequences

### 5.1 Benefits

1. **Formal Guarantees**: Mathematical privacy bounds
2. **Composability**: Budget accounting enables multiple operations
3. **Compliance**: Helps meet GDPR, CCPA requirements
4. **Trust**: Contributors confident their data is protected
5. **Transparency**: Clear privacy budget reporting

### 5.2 Costs

1. **Utility Loss**: Noise reduces embedding quality
2. **Complexity**: Budget management adds overhead
3. **Cold Start**: Need minimum contributors for aggregation
4. **False Positives**: PII detection may over-redact

### 5.3 Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| No privacy | Unacceptable for sensitive data |
| k-anonymity | No composition, weak guarantees |
| Trusted server | Single point of failure |
| Homomorphic encryption | Too slow for embeddings |

## 6. Privacy-Utility Curve

```
Utility
  |
  |****
  |    ****
  |        ****
  |            ****
  |                ****
  |                    ****
  +--------------------------> Privacy (epsilon)
  0.1   1     5     10
```

## 7. Related Decisions

- **ADR-004-rvf-format**: RVF files include privacy metadata
- **ADR-003-mcp-protocol**: Brain tools respect privacy budgets
- **ADR-006-sona-adaptation**: SONA feedback respects privacy

## 8. References

1. Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy."
2. Abadi, M. et al. (2016). "Deep Learning with Differential Privacy." CCS.
3. Mironov, I. (2017). "Renyi Differential Privacy." CSF.
4. Implementation: `/crates/pi-brain/src/privacy/`

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-13 | Architecture Team | Initial decision record |
