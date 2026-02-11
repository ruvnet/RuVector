# ADR-028: Security, Privacy & Compliance Architecture for RuVector DNA Analyzer

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: RuVector Security Architecture Team
**Deciders**: Architecture Review Board, Security Team, Privacy Officer
**Technical Area**: Genomic Data Security, Differential Privacy, Homomorphic Encryption, Zero-Knowledge Proofs, Post-Quantum Cryptography, Confidential Computing, Compliance
**Parent ADRs**: ADR-001 (Core Architecture), ADR-007 (Security Review), ADR-012 (Security Remediation), ADR-CE-008 (Multi-Tenant Isolation), ADR-CE-017 (Unified Audit Trail), ADR-DB-010 (Delta Security Model)

---

## Context and Problem Statement

The RuVector DNA Analyzer operates on genomic data -- the most sensitive category of personal information in existence. Unlike passwords, credit card numbers, or social security numbers, genomic data is **immutable**: a compromised genome cannot be rotated, revoked, or regenerated. A breach of genomic data constitutes permanent, irrevocable exposure of an individual's most intimate biological identity.

Furthermore, genomic data is **inherently shared**. An individual's genome encodes information about biological relatives: approximately 50% shared with parents and children, 25% with grandparents and grandchildren, 12.5% with first cousins. A single breach therefore cascades across an entire family tree, including individuals who never consented to analysis.

Published research has demonstrated that as few as **75 single-nucleotide polymorphisms (SNPs)** suffice to re-identify an individual from an ostensibly anonymized dataset (Gymrek et al., 2013; Erlich & Narayanan, 2014). Traditional de-identification techniques are therefore insufficient. The system must employ cryptographic and information-theoretic guarantees rather than relying on statistical anonymization alone.

### Regulatory Landscape

Genomic data falls under overlapping and sometimes conflicting regulatory frameworks:

| Regulation | Jurisdiction | Key Requirement | Genomic Specificity |
|------------|-------------|-----------------|---------------------|
| HIPAA | United States | PHI safeguards | Genomic data is PHI when linked to a covered entity |
| GINA | United States | Non-discrimination | Prohibits use of genetic information in health insurance and employment |
| GDPR Article 9 | European Union | Special category data | Explicit consent required; right to erasure applies |
| FDA 21 CFR Part 11 | United States | Electronic records | Applies when genomic analysis supports clinical decisions |
| ISO 27001 / 27701 | International | Information security / Privacy | Framework for ISMS and PIMS |
| California CCPA/CPRA | California | Consumer privacy rights | Genetic data classified as sensitive personal information |

### Threat Actor Taxonomy

| Actor | Motivation | Capability | Primary Targets |
|-------|-----------|------------|-----------------|
| Nation-state | Population-level intelligence, bioweapons | Advanced persistent threat, supply-chain compromise | Entire genomic databases, ancestry correlations |
| Insurance actuaries | Risk discrimination | Legal or semi-legal data acquisition, linkage attacks | Disease predisposition variants, pharmacogenomic markers |
| Law enforcement | Forensic identification | Familial DNA searching, compelled disclosure | STR profiles, Y-chromosome haplotypes, mitochondrial sequences |
| Employers | Workforce risk assessment | GINA violations through third-party data brokers | Late-onset disease genes (BRCA, Huntington's HTT, APOE e4) |
| Criminal extortion | Blackmail | Breach-and-threaten | Ancestry secrets, disease predispositions, paternity |
| Academic competitors | Priority, intellectual credit | Reconstruction attacks on published summary statistics | Rare variant frequencies, novel associations |

---

## 1. Threat Model for Genomic Data

### 1.1 Fundamental Properties of Genomic Threats

**Immutability.** Genomic data cannot be changed after compromise. The threat horizon is the lifetime of the individual plus the lifetimes of all descendants for whom the data remains predictive. Conservatively, a single breach has a **multi-generational impact window of 100+ years**.

**Re-identification surface.** Homer et al. (2008) demonstrated that the presence of an individual in a genome-wide association study (GWAS) cohort can be inferred from aggregate allele frequency statistics. Subsequent work has reduced the threshold to 75 independent SNPs for high-confidence re-identification. The RuVector HNSW index, by design, enables rapid nearest-neighbor retrieval -- the same property that makes it useful for genomic similarity search also makes it a powerful re-identification engine if access controls fail.

**Familial transitivity.** Compromising one genome partially compromises all biological relatives. The system must treat familial linkage as a first-class security concern, not an afterthought.

### 1.2 Attack Vectors Specific to RuVector

**Side-channel timing attacks on HNSW search.** The hierarchical navigable small world graph traverses different numbers of nodes depending on the query vector's proximity to existing data points. An attacker who can measure query latency with sufficient precision (microsecond-level, achievable via network timing) can infer whether a query genome is "near" entries in the database, enabling membership inference. Mitigation requires constant-time traversal padding or oblivious RAM (ORAM) techniques at the index layer.

**Model inversion on embeddings.** If the RuVector embedding model maps genomic sequences to dense vectors, an adversary with access to the embedding space can attempt to reconstruct the original sequence. For genomic data, even partial reconstruction (recovering the values of clinically significant SNPs) constitutes a severe breach. The relationship between the embedding function `E: Genome -> R^d` and the underlying genotype must be analyzed under the lens of membership inference and attribute inference attacks.

**Delta replay and injection.** Per ADR-DB-010, delta-based updates introduce the risk of replaying old deltas or injecting crafted deltas. In the genomic context, a replayed delta could revert a patient's corrected variant annotation, while an injected delta could alter clinical-grade variant calls.

**Embedding linkage across datasets.** If the same embedding model is used across multiple institutions, an adversary with access to embeddings from two datasets can perform linkage attacks, matching individuals across ostensibly separate cohorts. This is the vector-space analog of a record linkage attack.

### 1.3 Attack Surface Diagram

```
                        EXTERNAL BOUNDARY
                              |
          +-------------------+-------------------+
          |                   |                   |
    [API Gateway]      [WASM Client]      [MCP Interface]
          |                   |                   |
          +-------------------+-------------------+
                              |
                    [Authentication Layer]
                    [Claims Evaluator (ADR-010)]
                              |
          +-------------------+-------------------+
          |                   |                   |
    [Query Engine]     [Embedding Engine]  [Variant Caller]
    - HNSW traversal   - Model inference   - VCF processing
    - Timing leaks     - Inversion risk    - Path traversal
          |                   |                   |
          +-------------------+-------------------+
                              |
                    [Storage Layer]
                    - Encrypted vectors
                    - Delta chain integrity
                    - Key management
                              |
                    [Audit Trail (ADR-CE-017)]
                    - Hash-chained witnesses
                    - Tamper-evident log
```

---

## 2. Differential Privacy for Genomic Queries

### 2.1 Privacy Model

All population-level frequency queries (allele frequency, genotype frequency, haplotype frequency) must satisfy **(epsilon, delta)-differential privacy**. The system guarantees that the output of any query changes by at most a bounded amount whether or not any single individual's genome is included in the dataset.

**Definition.** A randomized mechanism M satisfies (epsilon, delta)-differential privacy if for all datasets D1 and D2 differing in one individual's record, and for all sets S of possible outputs:

```
Pr[M(D1) in S] <= exp(epsilon) * Pr[M(D2) in S] + delta
```

### 2.2 Noise Calibration for Genomic Queries

The existing `DifferentialPrivacy` implementation in `crates/ruvector-dag/src/qudag/crypto/differential_privacy.rs` provides Laplace and Gaussian mechanisms. For the DNA Analyzer, we extend this with genomic-specific calibration.

**Allele frequency queries.** For a biallelic SNP in a cohort of N individuals (2N chromosomes), the sensitivity of the allele frequency estimator is `Delta_f = 1/(2N)` (adding or removing one individual changes the count by at most 1 out of 2N alleles). The Laplace mechanism adds noise with scale `b = Delta_f / epsilon = 1 / (2N * epsilon)`.

```rust
/// Genomic-specific differential privacy for allele frequency queries.
pub struct GenomicDpConfig {
    /// Privacy budget per query
    pub epsilon: f64,
    /// Failure probability
    pub delta: f64,
    /// Cohort size (number of individuals)
    pub cohort_size: usize,
    /// Ploidy (2 for diploid organisms)
    pub ploidy: usize,
}

impl GenomicDpConfig {
    /// Sensitivity of allele frequency for a single biallelic locus.
    /// Adding/removing one individual changes allele count by at most `ploidy`
    /// out of `cohort_size * ploidy` total alleles.
    pub fn allele_frequency_sensitivity(&self) -> f64 {
        self.ploidy as f64 / (self.cohort_size * self.ploidy) as f64
    }

    /// Laplace noise scale for allele frequency queries.
    pub fn laplace_scale(&self) -> f64 {
        self.allele_frequency_sensitivity() / self.epsilon
    }
}
```

**Multi-SNP queries.** For queries spanning k correlated SNPs, we apply the composition theorem. Under basic composition, the total privacy loss is `k * epsilon`. Under advanced composition (Dwork, Roth, & Vadhan, 2010), for k queries each satisfying (epsilon, delta)-DP:

```
Total epsilon' = sqrt(2k * ln(1/delta')) * epsilon + k * epsilon * (exp(epsilon) - 1)
```

The existing `advanced_privacy_loss` method in the codebase implements this correctly.

### 2.3 Privacy Budget Accounting

Each dataset and each user maintains a **privacy budget ledger**. Every query consumes a portion of the budget. When the budget is exhausted, further queries are denied.

```rust
pub struct PrivacyBudgetLedger {
    /// Maximum total epsilon allowed
    pub max_epsilon: f64,
    /// Maximum total delta allowed
    pub max_delta: f64,
    /// Running total of epsilon consumed (advanced composition)
    pub consumed_epsilon: f64,
    /// Running total of delta consumed
    pub consumed_delta: f64,
    /// Per-query log for audit
    pub query_log: Vec<PrivacyExpenditure>,
}

pub struct PrivacyExpenditure {
    pub timestamp: u64,
    pub query_hash: [u8; 32],
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub mechanism: PrivacyMechanism,
    pub requester_id: String,
}

pub enum PrivacyMechanism {
    Laplace,
    Gaussian,
    ExponentialMechanism,
    SparseVector,
    /// Renyi DP with alpha parameter
    Renyi { alpha: f64 },
}
```

**Budget policy.** Default budget: epsilon_total = 10.0, delta_total = 1e-5 per dataset per calendar year. Clinical queries (pharmacogenomic lookups) draw from a separate, more generous budget since the patient has consented to their own data use.

### 2.4 Secure Multi-Party Computation for Cross-Institutional Studies

For federated GWAS across multiple institutions, no institution should reveal its raw genotype data. We specify a two-phase protocol:

**Phase 1: Secure aggregation of allele counts.** Each institution i holds counts (a_i, n_i) for allele a at locus L in cohort of size n_i. Institutions use additive secret sharing: each institution splits its count into k shares (one per other institution plus one for itself), distributes shares, and all institutions sum their received shares locally. The result is the global sum without any institution learning another's individual count.

**Phase 2: Differentially private release.** The aggregated count is perturbed with calibrated noise before release. The sensitivity is 1 (one institution contributing one additional individual changes the global count by at most 1 allele per locus for diploid).

**Protocol specification.**

```
Protocol: DP-SecureAlleleAggregation
Participants: Institutions I_1, ..., I_k
Input: Each I_j holds (count_j, total_j) for a locus
Output: Noisy global allele frequency

1. Each I_j generates random shares s_j1, ..., s_jk such that
   sum(s_j1..s_jk) = count_j  (modular arithmetic over Z_p, p > N_total)
2. I_j sends s_ji to I_i for all i != j, retains s_jj
3. Each I_i computes local_sum_i = sum over j of s_ji
4. All institutions broadcast local_sum_i
5. Global count = sum(local_sum_i) mod p
6. Trusted noise oracle (or threshold decryption) adds Laplace(1/epsilon) noise
7. Release noisy global frequency = (global_count + noise) / N_total
```

### 2.5 Renyi Differential Privacy (RDP) for Tight Composition

The basic and advanced composition theorems described in Section 2.2 provide correct but **pessimistic** bounds on cumulative privacy loss. For workloads involving hundreds or thousands of queries (typical in GWAS, where each SNP constitutes a separate query), this pessimism translates to premature budget exhaustion and reduced research utility. Renyi Differential Privacy (RDP) provides substantially tighter accounting.

**Definition.** A randomized mechanism M satisfies (alpha, epsilon)-Renyi Differential Privacy if for all adjacent datasets D1, D2:

```
D_alpha(M(D1) || M(D2)) <= epsilon
```

where the Renyi divergence of order alpha (alpha > 1) is:

```
D_alpha(P || Q) = 1/(alpha - 1) * log( E_Q[ (P(x)/Q(x))^alpha ] )
```

RDP interpolates between max-divergence (alpha -> infinity, equivalent to pure epsilon-DP) and KL-divergence (alpha -> 1). The key advantage is the **composition theorem**: composing k mechanisms with individual RDP guarantees (alpha, epsilon_1), ..., (alpha, epsilon_k) yields a composed guarantee of (alpha, sum(epsilon_i)). Converting back to (epsilon, delta)-DP:

```
epsilon(delta) = min over alpha>1 of { sum(epsilon_i(alpha)) + log(1/delta) / (alpha - 1) }
```

This optimization over alpha yields bounds that are **O(sqrt(k * log(1/delta)))** rather than the **O(k)** of naive sequential composition, enabling approximately **3x more queries** for the same total privacy budget when k >= 100.

#### 2.5.1 Gaussian Differential Privacy (GDP)

GDP parameterizes privacy through a single scalar mu, representing the standard deviation shift in a hypothesis testing framework. A mechanism satisfies mu-GDP if its privacy profile is at least as good as the hypothesis test between N(0,1) and N(mu,1).

**Central limit theorem for DP.** Under GDP, the composition of n mechanisms with individual mu_1, ..., mu_n converges to sqrt(sum(mu_i^2))-GDP. This provides closed-form composition without numerical optimization.

**For genomic queries.** Releasing allele frequency tables under GDP guarantees enables approximately 3x more queries at the same privacy level compared to (epsilon, delta)-DP with advanced composition, because GDP eliminates the slack terms in the conversion between RDP and (epsilon, delta)-DP.

#### 2.5.2 f-Differential Privacy (f-DP)

The f-DP framework generalizes all previous DP definitions by characterizing privacy as a **tradeoff function** between type I and type II errors in a hypothesis test distinguishing adjacent datasets:

```
T(M, D1, D2) = { (alpha, beta) : exists test phi with
                  E[phi(M(D1))] <= alpha and E[1-phi(M(D2))] <= beta }
```

The mechanism M satisfies f-DP for tradeoff function f if for all adjacent D1, D2, the type II error beta >= f(alpha) for all achievable type I error alpha. This is the **tightest possible** characterization of a mechanism's privacy guarantee, and subsumes epsilon-DP, (epsilon,delta)-DP, RDP, and GDP as special cases.

**Practical benefit for RuVector.** The PRV (Privacy Random Variable) accountant computes **exact** privacy loss under composition by convolving the privacy loss random variables of individual mechanisms. Unlike RDP, which requires optimizing over the order alpha, PRV accounting gives exact (non-pessimistic) bounds. For the RuVector workload of mixed Laplace and Gaussian mechanisms across allele frequency, genotype frequency, and haplotype frequency queries, PRV accounting recovers 15-25% additional budget compared to RDP.

#### 2.5.3 Implementation: RenyiPrivacyAccountant

The existing `PrivacyBudgetLedger` (Section 2.3) is extended with a `RenyiPrivacyAccountant` that replaces naive sequential composition with RDP-based accounting.

```rust
/// Renyi Differential Privacy accountant with PRV-based exact composition.
/// Replaces naive sequential composition for tighter privacy budget tracking.
pub struct RenyiPrivacyAccountant {
    /// Accumulated RDP epsilon values at each alpha in the discretized grid
    pub rdp_epsilons: Vec<(f64, f64)>,  // (alpha, epsilon) pairs
    /// Alpha grid for RDP optimization (typically 1.1, 1.5, 2, 3, ..., 256)
    pub alpha_grid: Vec<f64>,
    /// Target delta for conversion from RDP to (epsilon, delta)-DP
    pub target_delta: f64,
    /// PRV accountant for exact composition (when available)
    pub prv_accountant: Option<PrvAccountant>,
    /// Total number of compositions performed
    pub composition_count: usize,
}

/// Privacy Random Variable accountant for exact, non-pessimistic bounds.
pub struct PrvAccountant {
    /// Discretized privacy loss distributions for each mechanism
    pub loss_distributions: Vec<DiscretizedPld>,
    /// Grid resolution for numerical convolution
    pub grid_resolution: f64,
    /// Truncation bound for numerical stability
    pub truncation_bound: f64,
}

impl RenyiPrivacyAccountant {
    /// Record a Gaussian mechanism application with given sensitivity and sigma.
    /// RDP guarantee for Gaussian mechanism: epsilon(alpha) = alpha * (sensitivity^2) / (2 * sigma^2)
    pub fn record_gaussian(&mut self, sensitivity: f64, sigma: f64) {
        for (alpha, eps) in self.rdp_epsilons.iter_mut() {
            *eps += *alpha * sensitivity.powi(2) / (2.0 * sigma.powi(2));
        }
        self.composition_count += 1;
    }

    /// Convert accumulated RDP guarantee to (epsilon, delta)-DP.
    /// Returns the tightest epsilon by optimizing over the alpha grid.
    pub fn get_epsilon(&self) -> f64 {
        self.rdp_epsilons
            .iter()
            .map(|(alpha, eps)| eps + (1.0 / self.target_delta).ln() / (alpha - 1.0))
            .fold(f64::INFINITY, f64::min)
    }

    /// Remaining budget: max_epsilon minus consumed epsilon.
    pub fn remaining_budget(&self, max_epsilon: f64) -> f64 {
        max_epsilon - self.get_epsilon()
    }
}
```

**Migration path.** The `PrivacyBudgetLedger` retains its existing `consumed_epsilon` field for backward compatibility. A new `rdp_accountant: Option<RenyiPrivacyAccountant>` field is added. When the RDP accountant is present, budget decisions use `rdp_accountant.get_epsilon()` instead of `consumed_epsilon`. The `PrivacyMechanism::Renyi { alpha }` variant already exists in the codebase and is extended to support the full alpha grid.

**References.** Mironov (2017) -- Renyi Differential Privacy. Dong et al. (2022) -- Gaussian Differential Privacy. Balle et al. (2020) -- Hypothesis Testing Interpretations and the f-DP Framework.

---

## 3. Homomorphic Encryption for Secure Analysis

### 3.1 Scheme Selection

We adopt the **CKKS** (Cheon-Kim-Kim-Song) scheme for operations on encrypted genomic vectors. CKKS supports approximate arithmetic on encrypted real-valued vectors, which maps directly to the vector operations required by the RuVector core engine.

**Rationale for CKKS over BFV/BGV.** Genomic similarity computations (cosine distance, dot product) operate on floating-point vectors. CKKS natively encodes real numbers with bounded precision, avoiding the integer-encoding overhead of BFV/BGV schemes. The approximate nature of CKKS (results carry a small additive error) is acceptable for similarity search where ranking, not exact distances, determines results.

### 3.2 Target Operations and Performance Bounds

| Operation | Plaintext Baseline | Encrypted Target | Max Overhead |
|-----------|-------------------|------------------|-------------|
| Cosine similarity (384-dim) | 0.5us | 5us | 10x |
| HNSW distance comparison | 0.3us | 2.4us | 8x |
| Variant genotype lookup | 0.1us | 1.0us | 10x |
| Batch embedding (1000 vectors) | 50ms | 400ms | 8x |
| Allele frequency aggregation | 1ms | 8ms | 8x |

**Parameter selection.** For 128-bit security with CKKS:

```
Ring dimension (N):     2^15 = 32768
Coefficient modulus:    log(Q) = 438 bits (chain of 14 primes)
Scaling factor:         2^40
Max multiplicative depth: 12
Key-switching method:   Hybrid (decomposition base = 2^60)
```

These parameters provide sufficient depth for a single HNSW layer traversal (involving ~log(N) distance comparisons, each requiring one multiplication and one addition on encrypted data).

### 3.3 Selective Encryption Architecture

Full homomorphic encryption of the entire genome is computationally prohibitive for interactive queries. Instead, we employ **selective encryption** with a three-tier classification:

| Tier | Classification | Encryption | Examples |
|------|---------------|------------|---------|
| **Tier 1: Sensitive** | Clinically actionable, high discrimination risk | Full CKKS encryption | BRCA1/2, APOE, HTT, CFTR, HLA region |
| **Tier 2: Moderate** | Research-relevant, moderate re-identification risk | Encrypted at rest, decrypted in TEE for computation | Common GWAS hits, pharmacogenomic loci (CYP2D6, CYP2C19) |
| **Tier 3: Reference** | Low sensitivity, publicly catalogued variants | Cleartext with integrity protection (HMAC) | Synonymous variants, intergenic SNPs with >5% MAF |

The tier assignment is driven by a policy engine that considers:
- ClinVar clinical significance classification
- Allele frequency (rare variants are more identifying)
- Gene-disease association strength (OMIM, ClinGen)
- Regulatory classification under GINA protected categories

```rust
pub enum EncryptionTier {
    /// Full CKKS homomorphic encryption. All computation on ciphertext.
    Sensitive,
    /// Encrypted at rest. Decrypted only inside TEE for computation.
    Moderate,
    /// Cleartext with HMAC integrity verification.
    Reference,
}

pub struct GenomicRegionPolicy {
    pub chromosome: u8,
    pub start_position: u64,
    pub end_position: u64,
    pub tier: EncryptionTier,
    pub justification: &'static str,
    pub regulatory_references: Vec<RegulatoryReference>,
}

pub enum RegulatoryReference {
    Gina { section: &'static str },
    Hipaa { standard: &'static str },
    Gdpr { article: u32 },
    Fda21Cfr11 { section: &'static str },
}
```

### 3.4 Key Management

Encryption keys follow a hierarchy rooted in a hardware security module (HSM) or, where HSM is unavailable, a software-based key derivation chain using HKDF-SHA256.

```
Master Key (HSM-resident, never exported)
  |
  +-- Dataset Encryption Key (DEK) -- per-dataset, AES-256-GCM for at-rest
  |     |
  |     +-- Region Key -- per-genomic-region, derived via HKDF
  |
  +-- CKKS Public Key (for homomorphic operations)
  |
  +-- CKKS Secret Key (HSM-resident, used only for final decryption)
  |
  +-- CKKS Evaluation Keys (galois keys, relinearization keys -- public)
```

**Key rotation.** DEKs rotate on a 90-day cycle. On rotation, existing ciphertexts are re-encrypted under the new key via a background migration process. The old key is retained in a "decrypt-only" state for 180 days to handle in-flight operations, then destroyed. This mechanism also supports **cryptographic deletion** (see Section 5.5).

### 3.5 Fully Homomorphic Encryption with TFHE Bootstrapping

While CKKS (Section 3.1) excels at approximate arithmetic over encrypted real-valued vectors, certain genomic operations require **exact Boolean or integer computation** on encrypted data. The TFHE (Torus Fully Homomorphic Encryption) scheme addresses this through **programmable bootstrapping**, enabling the evaluation of arbitrary functions on encrypted data with no limit on circuit depth.

#### 3.5.1 TFHE Scheme Overview

TFHE represents ciphertexts as elements of the real torus T = R/Z and leverages a bootstrapping procedure that simultaneously reduces noise and evaluates an arbitrary lookup table (LUT). Unlike leveled HE schemes (CKKS, BFV) that require pre-determined multiplicative depth, TFHE bootstrapping resets the noise after every gate evaluation, enabling unbounded computation depth.

**Gate bootstrapping.** A single Boolean gate (AND, OR, XOR, NAND, etc.) is evaluated on encrypted bits in approximately **10ms** on a modern CPU. Each gate evaluation includes a full bootstrap, producing a fresh ciphertext with minimal noise regardless of how many prior operations have been performed.

**Programmable bootstrapping.** Rather than evaluating a single Boolean gate, programmable bootstrapping evaluates an arbitrary function f: Z_p -> Z_p encoded as a lookup table during the bootstrap operation. This enables multi-valued logic, integer comparisons, and threshold functions in a single bootstrapping step.

#### 3.5.2 Genomic Applications of TFHE

| Operation | Description | TFHE Approach | Performance |
|-----------|-------------|---------------|-------------|
| Variant comparison | Check if encrypted genotype matches a specific allele | Single gate bootstrap per bit | ~30ms for diploid genotype (3 bits) |
| Allele frequency statistics | Compute allele counts across encrypted genomes | Encrypted integer addition with carry | ~500ms per SNP per 1000 individuals |
| Pathogenicity predicate | Evaluate if a variant is pathogenic per ClinVar | LUT bootstrap with ClinVar classification table | ~50ms per variant |
| Cross-institutional GWAS | Compute chi-squared statistics on encrypted genotypes | Multi-bit arithmetic with programmable bootstrap | ~2 hours for 100K SNPs across 3 institutions |

**Performance with GPU acceleration.** The Zama Concrete library provides GPU-accelerated TFHE operations achieving approximately **1 million gate evaluations per second** on an NVIDIA A100 GPU. This brings the overhead for cross-institutional GWAS from impractical to feasible: a 100-1000x overhead versus plaintext computation is acceptable when the alternative is not performing the study at all due to privacy constraints.

#### 3.5.3 TFHE Pipeline for Cross-Institutional Genomic Analysis

```
Protocol: TFHE-FederatedGenomicCompute

Participants: Institutions I_1, ..., I_k; Coordinator C
Setup: All institutions agree on TFHE parameters and share evaluation keys

1. Encrypt: Each I_j encrypts its genotype matrix G_j under its local TFHE key
   - Each genotype encoded as 2-bit value: 00=hom_ref, 01=het, 10=hom_alt, 11=missing
   - Ciphertext size: ~2 KB per encrypted genotype (vs 2 bits plaintext)

2. Key-switch: Institutions perform distributed key-switching so all ciphertexts
   are under a common evaluation key (no single party holds the decryption key)

3. Compute: Coordinator evaluates GWAS statistics on encrypted data:
   a. Allele counting: homomorphic addition of encrypted genotype values
   b. Chi-squared statistic: programmable bootstrap for squaring and division
   c. P-value threshold: comparison gate bootstrap for significance test

4. Result: Encrypted summary statistics (allele frequencies, chi-squared, p-values)

5. Threshold decrypt: k-of-n threshold decryption reveals only results passing
   significance threshold; non-significant results are never decrypted
```

#### 3.5.4 Implementation: TfheGenomicCompute

```rust
/// TFHE-based genomic computation engine for exact operations on encrypted genotypes.
/// Wraps Concrete/TFHE-rs library with genomic-domain-specific abstractions.
pub struct TfheGenomicCompute {
    /// TFHE parameter set (controls security level and performance)
    pub params: TfheParams,
    /// Client key (held by data owner; never shared)
    pub client_key: Option<TfheClientKey>,
    /// Server key (evaluation key; can be shared with compute nodes)
    pub server_key: TfheServerKey,
    /// Bootstrapping key for programmable bootstrapping
    pub bootstrap_key: TfheBootstrapKey,
    /// GPU acceleration configuration
    pub gpu_config: Option<GpuAccelConfig>,
}

pub struct TfheParams {
    /// LWE dimension (security parameter)
    pub lwe_dimension: usize,        // 742 for 128-bit security
    /// GLWE dimension
    pub glwe_dimension: usize,       // 1
    /// Polynomial size for bootstrapping
    pub polynomial_size: usize,      // 2048
    /// Base log for decomposition
    pub base_log: usize,             // 23
    /// Decomposition level
    pub level: usize,                // 1
    /// Standard deviation of encryption noise
    pub noise_std_dev: f64,          // 2^{-25}
}

pub struct GpuAccelConfig {
    /// Enable GPU acceleration via CUDA
    pub enabled: bool,
    /// Target gate evaluations per second
    pub target_throughput: u64,       // ~1_000_000 on A100
    /// Maximum GPU memory allocation (bytes)
    pub max_gpu_memory: usize,
}

impl TfheGenomicCompute {
    /// Encrypt a diploid genotype (0, 1, or 2 copies of alt allele) as TFHE ciphertext.
    pub fn encrypt_genotype(&self, genotype: u8) -> TfheCiphertext {
        assert!(genotype <= 2, "Invalid diploid genotype");
        // Encode as 2-bit FheUint2 via client key
        unimplemented!("Delegate to TFHE-rs FheUint2::encrypt")
    }

    /// Compute allele frequency across a vector of encrypted genotypes.
    /// Returns encrypted count (sum of alt alleles) without decryption.
    pub fn encrypted_allele_count(&self, genotypes: &[TfheCiphertext]) -> TfheCiphertext {
        // Homomorphic addition chain with bootstrapping every 8 additions
        // to control noise growth. O(n) additions, O(n/8) bootstraps.
        unimplemented!("Homomorphic addition with periodic bootstrap")
    }

    /// Evaluate ClinVar pathogenicity lookup table via programmable bootstrapping.
    /// Input: encrypted variant classification code. Output: encrypted pathogenicity flag.
    pub fn pathogenicity_predicate(&self, variant_code: &TfheCiphertext) -> TfheCiphertext {
        // Programmable bootstrap with LUT encoding ClinVar 5-tier classification
        // into binary pathogenic/non-pathogenic
        unimplemented!("Programmable bootstrap with ClinVar LUT")
    }
}
```

**Key sizes and ciphertext expansion.** TFHE ciphertexts for a single encrypted bit are approximately 1 KB (LWE dimension 742 * 8 bytes). For a diploid genotype (2 bits), the ciphertext is ~2 KB, representing a 8,000x expansion over plaintext. The server (evaluation) key is approximately 30 MB. These sizes are manageable for targeted analyses (specific loci or gene panels) but prohibitive for whole-genome encrypted computation, reinforcing the tiered encryption strategy of Section 3.3: TFHE is applied selectively to Tier 1 (Sensitive) loci where exact computation on encrypted data is mandatory.

**References.** Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2020). "TFHE: Fast Fully Homomorphic Encryption over the Torus." *Journal of Cryptology*. Zama Concrete Library documentation.

---

## 4. Zero-Knowledge Proofs for Genomic Attestation

### 4.1 Motivation

Clinical and social scenarios require proving properties of a genome without revealing the genome itself:

- A patient proves compatibility with a prescribed drug (pharmacogenomics) without revealing their CYP2D6 metabolizer status to the pharmacist's information system.
- A couple proves genetic compatibility (absence of shared recessive disease carrier status) without disclosing individual genotypes to each other or a third party.
- An individual proves membership in a specific ancestry cluster for a research study without revealing full ancestry composition.

### 4.2 Construction: zk-SNARK for Genotype Predicates

We define a general-purpose **genotype predicate circuit** that proves statements of the form:

```
"I possess a genotype at locus L that satisfies predicate P,
 committed under Pedersen commitment C."
```

The circuit operates over the BLS12-381 curve (chosen for its efficient pairing operations and compatibility with the Groth16 proof system).

**Circuit definition (R1CS).**

```
Public inputs:  commitment C, predicate hash H(P), locus identifier L
Private inputs: genotype g (encoded as {0, 1, 2} for {homozygous ref, het, homozygous alt}),
                blinding factor r

Constraints:
1. C == g * G + r * H                    (Pedersen commitment opens correctly)
2. g in {0, 1, 2}                         (valid diploid genotype)
3. P(g) == 1                              (predicate satisfied)
```

**Predicate examples.**

| Use Case | Predicate P(g) | Statement Proven |
|----------|---------------|------------------|
| Pharmacogenomic safety | g != 2 at CYP2D6*4 | "I am not a CYP2D6 poor metabolizer" |
| Carrier screening | g1 + g2 < 4 for both partners | "We do not both carry two copies of the same recessive allele" |
| Ancestry membership | embedding(genome) in cluster C | "My ancestry falls within cluster C" |
| Disease risk threshold | risk_score(genotypes) < T | "My polygenic risk score is below threshold T" |

### 4.3 Proof Parameters and Performance

| Parameter | Value |
|-----------|-------|
| Curve | BLS12-381 |
| Proof system | Groth16 (succinct, constant-size proofs) |
| Proof size | 192 bytes (3 group elements) |
| Verification time | <5ms |
| Proving time (single locus) | <500ms |
| Proving time (polygenic, 100 loci) | <10s |
| Trusted setup | Powers of Tau ceremony + circuit-specific phase 2 |

**Implementation note.** The existing ZK proof infrastructure in the codebase (see `/home/user/ruvector/docs/security/zk_security_audit_report.md`) has been audited and found to contain critical vulnerabilities in its proof-of-concept implementation. For genomic attestation, we mandate the use of production-grade libraries:

- **arkworks-rs** (ark-groth16, ark-bls12-381) for proof generation and verification
- **merlin** for Fiat-Shamir transcript management
- **curve25519-dalek** for Pedersen commitments where Ristretto255 suffices
- **subtle** crate for all constant-time operations

The custom hash function, fake bulletproof verification, and blinding-in-commitment-struct patterns identified in the audit report are explicitly prohibited in the genomic security module.

### 4.4 Genomic Attestation Protocol

```
Protocol: ZK-GenomicAttestation

Setup (once per predicate class):
  1. Define R1CS circuit for predicate P
  2. Run trusted setup (Powers of Tau + Phase 2)
  3. Publish verification key VK; retain proving key PK

Prove (per attestation request):
  1. Patient retrieves encrypted genotype from RuVector
  2. Decryption occurs inside TEE (or client-side)
  3. Patient computes Pedersen commitment C = g*G + r*H
  4. Patient generates Groth16 proof pi using PK, (C, H(P), L), (g, r)
  5. Patient sends (C, pi, L, H(P)) to verifier

Verify:
  1. Verifier checks pi against VK with public inputs (C, H(P), L)
  2. If valid: predicate P holds for the committed genotype
  3. Verifier learns NOTHING about g beyond P(g) == true
```

### 4.5 Universal and Recursive Proofs with Plonk/Halo2

The Groth16 system (Section 4.3) produces the smallest proofs (192 bytes) with the fastest verification (<5ms) but requires a **per-circuit trusted setup**. Every time a new genomic predicate class is defined, a new trusted setup ceremony must be conducted. This is operationally burdensome and introduces a recurring trust assumption. Plonk and Halo2 eliminate these limitations.

#### 4.5.1 Plonk: Universal Structured Reference String

Plonk introduces a **universal, updatable structured reference string (SRS)**: a single trusted setup supports all circuits up to a maximum size. Adding a new genomic predicate (e.g., a new pharmacogenomic interaction check) requires only circuit compilation against the existing SRS, not a new ceremony.

**Key properties.**

| Property | Groth16 | Plonk | Halo2 |
|----------|---------|-------|-------|
| Trusted setup | Per-circuit | Universal (one-time) | **None** (transparent) |
| Proof size | 192 bytes | ~400 bytes | ~1.5 KB |
| Verification time | ~3ms | ~5ms | ~10ms |
| Proving time (2^20 constraints) | ~3s | ~5s | ~8s |
| Updatable SRS | No | Yes | N/A |
| Recursive composition | Complex | Possible | **Native** |

**Plonk arithmetization.** Plonk uses a custom gate constraint system over a univariate polynomial commitment scheme (Kate/KZG commitments over BLS12-381). The arithmetic circuit for a genotype predicate check with ~2^20 constraints (sufficient for polygenic risk score computation over 100 loci with lookup tables) compiles to a Plonk circuit with:

- **Proof generation:** ~5 seconds on a modern 8-core CPU
- **Proof verification:** ~5ms (dominated by 2 pairing operations + polynomial evaluation)
- **SRS size:** ~100 MB for circuits up to 2^20 gates (one-time download, reused across all predicates)

#### 4.5.2 Halo2: Recursive Proofs Without Trusted Setup

Halo2 eliminates the trusted setup entirely by replacing KZG polynomial commitments with an **Inner Product Argument (IPA)** commitment scheme. The IPA requires only a random group element as a public parameter (transparent setup), removing all trust assumptions.

**Recursive proof composition.** Halo2's defining capability is efficient recursive proof composition: a proof can verify another proof as part of its circuit. For genomic attestation, this enables:

1. **Multi-stage analysis proofs.** A variant calling pipeline involves multiple stages (alignment, calling, annotation, interpretation). Each stage generates a Halo2 proof. The final proof **recursively verifies all intermediate proofs**, producing a single compact attestation that the entire pipeline executed correctly on the committed input.

2. **Aggregated institutional proofs.** In a federated GWAS, each institution generates a Halo2 proof that its local computation was correct. A recursive aggregator composes these into a single proof, which any verifier can check without interacting with individual institutions.

3. **Incremental attestation.** As new variants are added to a patient's genome over time (e.g., from successive sequencing runs), each update generates an incremental proof that is recursively composed with the previous attestation, maintaining a continuously valid aggregate proof.

**Concrete performance for genomic predicates.**

```
Circuit: "Patient carries a pathogenic BRCA1 variant" (without revealing which one)
- Predicate: EXISTS v in BRCA1_pathogenic_set WHERE genotype(v) in {1,2}
- BRCA1 pathogenic variant set: ~4,000 variants (ClinVar)
- Circuit size: ~2^18 constraints (lookup table for variant set membership)
- Proof generation: ~2 seconds (single-threaded), ~0.5 seconds (8 threads)
- Proof verification: ~5ms
- Proof size: ~1.5 KB (IPA commitment)
- Recursive aggregation of 10 proofs: ~3 seconds, output ~1.5 KB
```

#### 4.5.3 Implementation: Extending Groth16 with Plonk Backend

```rust
/// Proof system backend selector for genomic attestation.
/// Extends existing Groth16 infrastructure with Plonk and Halo2 backends.
pub enum ZkBackend {
    /// Groth16: smallest proofs, fastest verification, per-circuit trusted setup.
    /// Use for: high-frequency verification (millions of checks), fixed predicate sets.
    Groth16 {
        proving_key: Groth16ProvingKey,
        verification_key: Groth16VerificationKey,
    },
    /// Plonk: universal SRS, no per-circuit setup, moderate proof size.
    /// Use for: evolving predicate library, rapid deployment of new checks.
    Plonk {
        srs: PlonkUniversalSrs,    // ~100 MB, shared across all circuits
        circuit_key: PlonkCircuitKey,
    },
    /// Halo2: no trusted setup, recursive composition, largest proofs.
    /// Use for: multi-stage pipelines, federated proofs, maximum trust minimization.
    Halo2 {
        params: Halo2Params,       // Transparent setup parameters
        circuit: Halo2Circuit,
    },
}

/// Genomic attestation service supporting multiple proof backends.
pub struct GenomicAttestationService {
    /// Active backend (selected based on use case)
    pub backend: ZkBackend,
    /// Predicate registry mapping predicate IDs to compiled circuits
    pub predicate_registry: HashMap<String, CompiledPredicate>,
    /// Proof cache for recently verified proofs (avoid re-verification)
    pub proof_cache: LruCache<[u8; 32], VerificationResult>,
    /// Metrics: proof generation times, verification counts, cache hit rates
    pub metrics: AttestationMetrics,
}

impl GenomicAttestationService {
    /// Generate a proof for a genomic predicate using the active backend.
    pub fn prove(
        &self,
        predicate_id: &str,
        private_genotype: &GenotypeData,
        blinding: &BlindingFactor,
    ) -> Result<GenomicProof, ProofError> {
        let predicate = self.predicate_registry.get(predicate_id)
            .ok_or(ProofError::UnknownPredicate)?;
        match &self.backend {
            ZkBackend::Groth16 { proving_key, .. } => {
                // Existing Groth16 path
                unimplemented!("Delegate to ark-groth16")
            }
            ZkBackend::Plonk { srs, circuit_key } => {
                // Plonk proving with universal SRS
                unimplemented!("Delegate to plonk library")
            }
            ZkBackend::Halo2 { params, circuit } => {
                // Halo2 proving with transparent setup
                unimplemented!("Delegate to halo2 library")
            }
        }
    }

    /// Recursively aggregate multiple proofs into a single compact proof.
    /// Only available with Halo2 backend.
    pub fn recursive_aggregate(
        &self,
        proofs: &[GenomicProof],
    ) -> Result<GenomicProof, ProofError> {
        match &self.backend {
            ZkBackend::Halo2 { .. } => {
                // Recursive Halo2 aggregation
                unimplemented!("Recursive IPA verification circuit")
            }
            _ => Err(ProofError::RecursionNotSupported),
        }
    }
}
```

**Migration strategy.** The existing Groth16 infrastructure remains the default for high-frequency, fixed-predicate attestations (pharmacogenomic safety checks). New predicates are deployed against the Plonk universal SRS by default. For federated and multi-stage workflows, Halo2 is used. The `ZkBackend` enum allows runtime selection based on the attestation use case.

**References.** Gabizon, A., Williamson, Z., & Ciobotaru, O. (2019). "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge." *IACR ePrint*. Bowe, S., Grigg, J., & Hopwood, D. (2019). "Halo: Recursive Proof Composition without a Trusted Setup." *IACR ePrint*.

---

## 5. Access Control and Audit

### 5.1 Claims-Based Authorization for Genomic Data

Extending the existing claims system (per the claims-authorizer agent specification), genomic access control introduces domain-specific claim types.

**Role hierarchy.**

```
Level 4: Patient (data subject)
  - Full access to own genomic data
  - Can grant/revoke access to others
  - Can request erasure
  - Can export in standard formats (VCF, FHIR)

Level 3: Clinician (treating physician)
  - Access to patient's clinically relevant variants (with consent)
  - Cannot access raw sequence data without explicit authorization
  - Time-limited access tokens (expire with episode of care)

Level 2: Researcher (IRB-approved)
  - Access to differentially private aggregate statistics
  - No individual-level data without explicit consent + IRB approval
  - Budget-limited queries (see Section 2.3)

Level 1: Analyst (institutional)
  - Access to pre-computed, anonymized summary reports
  - No query capability against individual records
  - Read-only access to published results
```

### 5.2 Fine-Grained Access Specifications

Access control operates at three granularity levels.

```rust
pub enum GenomicAccessScope {
    /// Access to a specific gene (e.g., "BRCA1")
    Gene { gene_symbol: String },
    /// Access to a specific variant (e.g., rs1234567)
    Variant { rsid: String },
    /// Access to a genomic region (e.g., chr17:41196312-41277500)
    Region {
        chromosome: u8,
        start: u64,
        end: u64,
    },
    /// Access to a functional category (e.g., all pharmacogenomic variants)
    Category { category: GenomicCategory },
    /// Access to aggregate statistics only (no individual genotypes)
    AggregateOnly,
}

pub enum GenomicCategory {
    Pharmacogenomic,
    CancerPredisposition,
    CardiovascularRisk,
    CarrierScreening,
    Ancestry,
    Forensic,    // Highly restricted: STR profiles, Y-haplogroups
}
```

**Policy example.** A pharmacist checking drug interactions receives a claim of the form:

```yaml
claim:
  role: clinician
  scope: genomic:category:pharmacogenomic
  patient_id: "patient-uuid"
  valid_from: "2026-02-11T00:00:00Z"
  valid_until: "2026-02-11T23:59:59Z"
  access_level: predicate_only    # Can verify ZK proof, cannot see raw genotype
  audit_required: true
```

### 5.3 Immutable Audit Log

All genomic data access events are recorded in a hash-chained, append-only audit log extending the UnifiedWitnessLog (ADR-CE-017).

```rust
pub struct GenomicAccessWitness {
    /// Sequential event ID
    pub event_id: u64,
    /// Hash of previous witness (chain integrity)
    pub prev_hash: [u8; 32],
    /// Timestamp (from trusted time source)
    pub timestamp: u64,
    /// Who accessed the data
    pub accessor: AccessorIdentity,
    /// What was accessed
    pub resource: GenomicAccessScope,
    /// Access type
    pub action: AccessAction,
    /// Authorization decision
    pub decision: AuthorizationDecision,
    /// Claims presented
    pub claims_presented: Vec<String>,
    /// Consent reference (if applicable)
    pub consent_id: Option<String>,
    /// Hash of this record
    pub self_hash: [u8; 32],
}

pub enum AccessAction {
    Read,
    Query { query_hash: [u8; 32] },
    Export { format: ExportFormat },
    ZkProofGeneration { predicate_hash: [u8; 32] },
    AggregateQuery { epsilon_spent: f64 },
    Deletion { scope: DeletionScope },
}
```

The audit log is stored in a separate, write-only partition with independent backup. The hash chain provides tamper evidence: any modification to a historical record breaks the chain, detectable in O(n) time via sequential verification or O(log n) via Merkle tree indexing.

### 5.4 Consent Management

GDPR Article 7 requires freely given, specific, informed, and unambiguous consent. For genomic data (Article 9), explicit consent is mandatory.

```rust
pub struct GenomicConsent {
    pub consent_id: String,
    pub patient_id: String,
    pub granted_to: Vec<ConsentRecipient>,
    pub scope: Vec<GenomicAccessScope>,
    pub purpose: ConsentPurpose,
    pub granted_at: u64,
    pub expires_at: Option<u64>,
    pub revocable: bool,          // Must be true for GDPR compliance
    pub revoked_at: Option<u64>,
    pub signature: ConsentSignature,
}

pub enum ConsentPurpose {
    ClinicalCare,
    Pharmacogenomics,
    ResearchSpecific { study_id: String, irb_approval: String },
    ResearchBroad,      // Requires re-consent for each new study under GDPR
    CarrierScreening,
    AncestryAnalysis,
}
```

### 5.5 GDPR Article 17: Right to Erasure via Cryptographic Deletion

Physical deletion of genomic data from all backups, replicas, and derived products is operationally difficult. Instead, we implement **cryptographic deletion**: the data remains encrypted, but the encryption key is irrevocably destroyed, rendering the ciphertext computationally indistinguishable from random noise.

**Protocol.**

```
CryptographicDeletion(patient_id):
  1. Identify all DEKs associated with patient_id
  2. Re-verify erasure request (consent, identity verification)
  3. Record deletion request in audit log (this record is RETAINED)
  4. Destroy all copies of the patient-specific DEK:
     a. HSM: invoke key destruction command with audit witness
     b. All replicas: send key revocation via authenticated channel
     c. Key escrow (if any): destroy escrowed copy
  5. Mark all ciphertext blocks as "cryptographically deleted" in metadata
  6. Record completion in audit log with HSM attestation of key destruction
  7. Retain: audit log entries, anonymized aggregate statistics (already DP-protected)
  8. Return erasure confirmation with audit trail hash
```

**Key isolation requirement.** Each patient's genomic data must be encrypted under a patient-specific key (or a key derivable only with patient-specific material). Shared encryption keys across patients would make individual erasure impossible without re-encrypting all other patients' data.

---

## 6. Compliance Framework Mapping

### 6.1 HIPAA Compliance

| HIPAA Requirement | Implementation | Section |
|-------------------|---------------|---------|
| 164.312(a)(1) Access control | Claims-based RBAC with genomic scopes | 5.1, 5.2 |
| 164.312(b) Audit controls | Hash-chained GenomicAccessWitness log | 5.3 |
| 164.312(c)(1) Integrity | Signed deltas (ADR-DB-010), HMAC on Tier 3 data | 3.3 |
| 164.312(d) Authentication | mTLS, JWT with claims, TEE attestation | 5.1 |
| 164.312(e)(1) Transmission security | TLS 1.3 minimum, CKKS for computation | 3.1, 7.1 |
| 164.530(c) Minimum necessary | Fine-grained per-gene/variant access scopes | 5.2 |
| 164.524 Right of access | Patient-level export in VCF/FHIR formats | 5.1 |

### 6.2 GINA Compliance

| GINA Provision | Implementation |
|----------------|---------------|
| Title I: Health insurance non-discrimination | Genetic data inaccessible to insurance claims processors; enforced via claims system with no `insurance_underwriting` scope |
| Title II: Employment non-discrimination | Employer roles excluded from all genomic access scopes; audit log flags any access attempt from employer-classified entities |
| Forensic exemption (GINA does not cover law enforcement) | All law enforcement access requires court order; separate audit trail; data subject notified unless court orders otherwise |

### 6.3 GDPR Compliance (Article 9 -- Special Category Data)

| GDPR Requirement | Implementation | Section |
|-------------------|---------------|---------|
| Article 9(2)(a) Explicit consent | GenomicConsent with purpose limitation | 5.4 |
| Article 17 Right to erasure | Cryptographic deletion via key destruction | 5.5 |
| Article 20 Data portability | VCF export with patient's own key | 5.1 |
| Article 25 Data protection by design | Selective encryption, differential privacy by default | 2, 3.3 |
| Article 32 Security of processing | TEE, CKKS, HNSW timing protection | 3, 7 |
| Article 35 DPIA requirement | Mandatory before any new genomic processing activity | Operational |
| Recital 51 Special category processing | All genomic data classified as Article 9 by default | Architecture-wide |

### 6.4 FDA 21 CFR Part 11

When the DNA Analyzer supports clinical decision-making (e.g., pharmacogenomic dosing recommendations, variant pathogenicity classification used in diagnosis):

| Requirement | Implementation |
|-------------|---------------|
| 11.10(a) Validation | Validated variant calling pipeline with known-truth benchmarks (Genome in a Bottle) |
| 11.10(b) Accurate copies | Cryptographic hash verification on all data copies |
| 11.10(c) Record protection | Encryption at rest (Tier 1/2), integrity protection (all tiers) |
| 11.10(d) System access control | Claims-based access with role hierarchy |
| 11.10(e) Audit trail | GenomicAccessWitness with hash chain |
| 11.10(g) Authority checks | Consent verification before any write operation |
| 11.50 Electronic signatures | Ed25519 signatures on clinical reports, linked to identity |
| 11.70 Signature binding | Signature covers document hash; any modification invalidates |

### 6.5 ISO 27001 / 27701 Controls

| Control | Description | Genomic Implementation |
|---------|-------------|----------------------|
| A.8.2 Information classification | Three-tier classification (Sensitive/Moderate/Reference) | 3.3 |
| A.9.4 System and application access control | Claims-based RBAC with genomic scopes | 5.1 |
| A.10.1 Cryptographic controls | CKKS for computation, AES-256-GCM at rest, TLS 1.3 in transit | 3 |
| A.12.4 Logging and monitoring | Hash-chained audit trail with tamper detection | 5.3 |
| A.18.1 Compliance with legal requirements | Compliance mapping table (this section) | 6 |
| 27701-7.2.2 Identifying purposes | ConsentPurpose enum with explicit purpose limitation | 5.4 |
| 27701-7.3.4 Providing mechanism to withdraw consent | Consent revocation with cryptographic deletion | 5.4, 5.5 |

---

## 7. Secure Computation Pipeline

### 7.1 Trusted Execution Environments

All genomic computations that require access to plaintext genotype data occur inside a **trusted execution environment** (TEE). The TEE provides hardware-enforced memory encryption and attestation.

**Supported TEE platforms.**

| Platform | Technology | Use Case |
|----------|-----------|----------|
| Intel | TDX (Trust Domain Extensions) | Cloud-based variant calling, batch processing |
| Intel (legacy) | SGX (Software Guard Extensions) | Enclave-based key management, small computations |
| ARM | TrustZone + CCA | Edge/mobile genomic analysis |
| AMD | SEV-SNP | Cloud VMs with encrypted memory |

### 7.2 Memory Encryption for Vector Operations

Within the TEE, the RuVector HNSW index operates on decrypted vectors. The TEE's memory encryption engine (e.g., Intel TME or AMD SME) ensures that even a physical memory dump by a hypervisor or co-tenant yields only ciphertext.

**HNSW timing attack mitigation.** Inside the TEE, we additionally implement:

1. **Constant-iteration traversal.** Each HNSW layer search always visits exactly `ef_construction` nodes, with dummy comparisons for nodes that would not normally be visited. This prevents timing-based inference about query proximity to stored vectors.

2. **Oblivious memory access.** For Tier 1 (Sensitive) vectors, memory access patterns are made data-independent via Path ORAM. The overhead is O(log^2 N) per access, but applies only to the ~2% of vectors classified as Tier 1.

```rust
pub struct ObliviousHnswConfig {
    /// Enable constant-time traversal (pads to max iterations)
    pub constant_time_search: bool,
    /// Enable Path ORAM for sensitive-tier vectors
    pub oram_for_sensitive: bool,
    /// Maximum ORAM block count (determines tree height)
    pub oram_capacity: usize,
    /// Enable dummy distance computations
    pub dummy_comparisons: bool,
}
```

### 7.3 Attestation Chain for Result Provenance

Every computation result carries an **attestation chain** proving that:

1. The computation occurred inside a genuine TEE (hardware attestation quote).
2. The TEE was running the expected software version (measurement/hash of loaded code).
3. The input data was integrity-verified (hash of input ciphertexts).
4. The output was produced by the attested code from the attested inputs.

```rust
pub struct ComputationAttestation {
    /// TEE platform attestation quote (Intel TDX report or SGX quote)
    pub tee_quote: Vec<u8>,
    /// Hash of the binary loaded into the TEE
    pub code_measurement: [u8; 48],
    /// Hash of all input ciphertexts
    pub input_hash: [u8; 32],
    /// Hash of the computation output
    pub output_hash: [u8; 32],
    /// Timestamp from TEE-internal trusted clock
    pub timestamp: u64,
    /// Signature over the above fields using TEE-bound signing key
    pub signature: Vec<u8>,
    /// Certificate chain linking TEE signing key to platform root of trust
    pub certificate_chain: Vec<Vec<u8>>,
}
```

For clinical-grade results (FDA 21 CFR Part 11), the attestation chain serves as the electronic record's integrity proof. The certificate chain is rooted in the hardware manufacturer's root certificate (Intel, AMD, or ARM), providing a hardware root of trust independent of the software operator.

### 7.4 End-to-End Secure Pipeline

```
Patient Device          Cloud TEE                    Clinician
     |                      |                            |
     |-- Encrypted genome ->|                            |
     |                      |-- TEE attestation -------->|
     |                      |                            |-- Verify attestation
     |                      |<- Attested public key -----|
     |                      |                            |
     |                      | [Inside TEE:]              |
     |                      |   Decrypt genome           |
     |                      |   Run variant calling      |
     |                      |   Generate ZK proof        |
     |                      |   Re-encrypt results       |
     |                      |   Sign attestation         |
     |                      |                            |
     |<- Encrypted result --|                            |
     |<- Attestation -------|-- Attestation copy ------->|
     |<- ZK proof ----------|-- ZK proof copy ---------->|
     |                      |                            |
     |                      | [TEE scrubs memory]        |
     |                      |                            |
```

---

## 8. Post-Quantum Cryptography

### 8.1 Quantum Threat Assessment for Genomic Data

**Mosca's theorem** (Mosca, 2018) formalizes the urgency of post-quantum migration: if the time required to migrate a cryptographic system (`migration_time`) plus the number of years the data must remain secure (`shelf_life`) exceeds the time until a cryptographically relevant quantum computer exists (`quantum_threat_time`), then migration must begin immediately:

```
IF  migration_time + shelf_life > quantum_threat_time
THEN  start migration NOW
```

For genomic data, the parameters are stark:

| Parameter | Estimate | Rationale |
|-----------|----------|-----------|
| `migration_time` | 5-10 years | Enterprise-wide cryptographic migration across key management, protocols, and stored ciphertexts |
| `shelf_life` | **100+ years** | Genomic data is immutable and relevant across generations (Section 1.1) |
| `quantum_threat_time` | 10-30 years | Conservative estimates for fault-tolerant quantum computers capable of running Shor's algorithm |
| **Urgency** | **IMMEDIATE** | Even the most optimistic quantum timeline (30 years) is exceeded by shelf_life + migration_time (105-110 years) |

The conclusion is unambiguous: **post-quantum migration for genomic data is the most urgent case in all of information security.** Every RSA, ECDSA, and ECDH ciphertext or signature protecting genomic data today is subject to "harvest now, decrypt later" attacks by adversaries (particularly nation-states, per the threat actor taxonomy in the Context section) who can store encrypted genomic data today and decrypt it when quantum computers become available.

### 8.2 NIST Post-Quantum Standards

NIST finalized three post-quantum cryptographic standards in 2024. RuVector adopts all three for defense-in-depth:

| Standard | NIST Designation | Algorithm | Type | Primary Use in RuVector |
|----------|-----------------|-----------|------|------------------------|
| FIPS 203 | ML-KEM | CRYSTALS-Kyber | Lattice-based key encapsulation | Key exchange for all data-in-transit and key encapsulation for data-at-rest |
| FIPS 204 | ML-DSA | CRYSTALS-Dilithium | Lattice-based digital signature | Signing attestations, audit log entries, clinical reports, delta chains |
| FIPS 205 | SLH-DSA | SPHINCS+ | Hash-based digital signature | Ultra-long-term signatures (100+ year verification horizon), root certificates |

**Key sizes and performance.**

| Algorithm | Security Level | Public Key | Secret Key | Ciphertext/Signature | Operations/sec (CPU) |
|-----------|---------------|------------|------------|---------------------|---------------------|
| Kyber768 (ML-KEM-768) | NIST Level 3 (~AES-192) | 1,184 bytes | 2,400 bytes | 1,088 bytes (ciphertext) | ~50,000 encaps/sec |
| Dilithium3 (ML-DSA-65) | NIST Level 3 | 1,952 bytes | 4,000 bytes | 3,293 bytes (signature) | ~10,000 sign/sec |
| SPHINCS+-SHA256-192f | NIST Level 3 | 48 bytes | 96 bytes | 35,664 bytes (signature) | ~50 sign/sec |

**SPHINCS+ trade-off.** SPHINCS+ signatures are large (35 KB) and slow to generate, but their security rests **solely on the collision resistance of hash functions** (no lattice assumptions). This makes SPHINCS+ the most conservative choice for artifacts that must remain verifiable for 100+ years. RuVector uses SPHINCS+ for:
- Root CA certificates in the key hierarchy
- Long-term archival signatures on sealed genomic records
- Attestation chain root signatures

For high-frequency operations (API authentication, delta signing), Dilithium provides a practical balance of speed and post-quantum security.

### 8.3 Hybrid Classical/Post-Quantum Key Exchange

During the transition period, RuVector implements **hybrid key exchange** combining a classical algorithm with a post-quantum algorithm. A hybrid ciphertext is secure if **either** algorithm remains unbroken. This provides defense-in-depth against both:
- Undiscovered vulnerabilities in the new post-quantum algorithms
- Quantum attacks on the classical algorithms

**Hybrid construction: X25519 + Kyber768.**

```
HybridKeyExchange(client, server):
  1. Client generates X25519 keypair (ecdh_pk, ecdh_sk)
  2. Client generates Kyber768 keypair (kyber_pk, kyber_sk)
  3. Client sends (ecdh_pk || kyber_pk) to server

  4. Server generates X25519 keypair, computes ecdh_ss = X25519(server_sk, ecdh_pk)
  5. Server encapsulates against kyber_pk: (kyber_ct, kyber_ss) = Kyber.Encaps(kyber_pk)
  6. Server computes combined_ss = HKDF-SHA384(ecdh_ss || kyber_ss, context="RuVector-PQ-Hybrid")
  7. Server sends (server_ecdh_pk || kyber_ct) to client

  8. Client computes ecdh_ss = X25519(ecdh_sk, server_ecdh_pk)
  9. Client decapsulates: kyber_ss = Kyber.Decaps(kyber_sk, kyber_ct)
  10. Client computes combined_ss = HKDF-SHA384(ecdh_ss || kyber_ss, context="RuVector-PQ-Hybrid")

  Both parties now share combined_ss, which is secure if either X25519 or Kyber768 holds.
```

**Wire overhead.** The hybrid key exchange adds 1,184 bytes (Kyber768 public key) to the client hello and 1,088 bytes (ciphertext) to the server response, for a total overhead of ~2.3 KB per handshake. This is negligible compared to genomic data payloads (a single VCF file is typically megabytes).

### 8.4 Post-Quantum Key Hierarchy

The existing key hierarchy (Section 3.4) is extended with post-quantum algorithms:

```
Master Key (HSM-resident, never exported)
  |
  +-- [PQ] Root CA Signing Key (SPHINCS+-SHA256-192f)
  |     |   Ultra-long-term root of trust; 100+ year verification horizon
  |     |
  |     +-- [PQ] Intermediate CA Signing Key (Dilithium3)
  |           |   5-year rotation; signs operational certificates
  |           |
  |           +-- [PQ] Node Signing Key (Dilithium3)
  |           |     Per-node, 90-day rotation; signs attestations, deltas
  |           |
  |           +-- [PQ] Audit Log Signing Key (Dilithium3)
  |                 Dedicated key for GenomicAccessWitness signatures
  |
  +-- [PQ] Key Encapsulation Key (Kyber768)
  |     Per-session key agreement; combined with X25519 in hybrid mode
  |
  +-- [HYBRID] Dataset Encryption Key (DEK)
  |     AES-256-GCM, encapsulated under Kyber768 + X25519 hybrid
  |     |
  |     +-- Region Key -- per-genomic-region, derived via HKDF-SHA384
  |
  +-- CKKS Public Key (for homomorphic operations)
  |     (CKKS security based on RLWE -- already lattice-based, inherently PQ-resistant)
  |
  +-- CKKS Secret Key (HSM-resident, used only for final decryption)
  |
  +-- CKKS Evaluation Keys (galois keys, relinearization keys -- public)
```

**Note on CKKS.** The CKKS homomorphic encryption scheme is based on the Ring Learning With Errors (RLWE) problem, which is the same hardness assumption underlying CRYSTALS-Kyber. CKKS is therefore **already post-quantum resistant** at its current parameter settings (ring dimension 2^15 provides >128-bit post-quantum security). No migration is needed for the homomorphic encryption layer.

### 8.5 Implementation Plan

```rust
/// Post-quantum cryptographic primitive configuration for RuVector.
pub struct PostQuantumConfig {
    /// Key encapsulation mechanism (for key exchange and key wrapping)
    pub kem: PqKem,
    /// Digital signature algorithm (for signing)
    pub signature: PqSignature,
    /// Whether to use hybrid mode (classical + PQ)
    pub hybrid_mode: bool,
    /// Classical algorithm for hybrid mode
    pub classical_kem: Option<ClassicalKem>,
    pub classical_signature: Option<ClassicalSignature>,
}

pub enum PqKem {
    /// FIPS 203: ML-KEM-768 (CRYSTALS-Kyber)
    /// Public key: 1,184 bytes, Ciphertext: 1,088 bytes
    MlKem768,
    /// FIPS 203: ML-KEM-1024 (for highest security tier)
    /// Public key: 1,568 bytes, Ciphertext: 1,568 bytes
    MlKem1024,
}

pub enum PqSignature {
    /// FIPS 204: ML-DSA-65 (CRYSTALS-Dilithium3)
    /// Public key: 1,952 bytes, Signature: 3,293 bytes
    MlDsa65,
    /// FIPS 205: SLH-DSA-SHA2-192f (SPHINCS+)
    /// Public key: 48 bytes, Signature: 35,664 bytes
    /// Use ONLY for long-term archival signatures
    SlhDsa192f,
}

pub enum ClassicalKem {
    X25519,
}

pub enum ClassicalSignature {
    Ed25519,
}

/// Hybrid key encapsulation: combines classical and post-quantum KEM.
pub struct HybridKemOutput {
    /// Combined shared secret: HKDF(classical_ss || pq_ss)
    pub shared_secret: [u8; 48],  // 384-bit output from HKDF-SHA384
    /// Concatenated ciphertexts: classical_ct || pq_ct
    pub ciphertext: Vec<u8>,
    /// Metadata for decapsulation
    pub kem_algorithm: String,
}
```

**Migration schedule.** Phase 1 (immediate): deploy hybrid X25519+Kyber768 for all new TLS connections and key encapsulations. Phase 2 (within 6 months): replace all Ed25519 operational signatures with Dilithium3. Phase 3 (within 12 months): issue new root CA certificate signed with SPHINCS+. Phase 4 (within 18 months): re-encapsulate all existing DEKs under Kyber768 hybrid and destroy classical-only key material.

**References.** NIST FIPS 203 (2024) -- Module-Lattice-Based Key-Encapsulation Mechanism Standard. NIST FIPS 204 (2024) -- Module-Lattice-Based Digital Signature Standard. NIST FIPS 205 (2024) -- Stateless Hash-Based Digital Signature Standard. Mosca, M. (2018). "Cybersecurity in an Era with Quantum Computers: Will We Be Ready?" *IEEE Security & Privacy*.

---

## 9. Secure Multi-Party Computation with SPDZ Protocol

### 9.1 Motivation: Beyond Additive Secret Sharing

The secure aggregation protocol in Section 2.4 uses additive secret sharing, which provides security against **semi-honest** (honest-but-curious) adversaries: institutions that follow the protocol correctly but attempt to learn additional information from the messages they receive. However, semi-honest security is insufficient for high-stakes genomic computation, where an adversary may deviate from the protocol (e.g., sending malformed shares to bias the result, or colluding with other institutions to reconstruct private inputs).

The **SPDZ protocol** (Damgard et al., 2012) provides security against **active (malicious) adversaries** who can arbitrarily deviate from the protocol specification. SPDZ achieves this through information-theoretic MACs (Message Authentication Codes) on secret-shared values, detecting any manipulation of shares during computation.

### 9.2 SPDZ Protocol Overview

SPDZ operates in two phases:

**Offline (preprocessing) phase.** Before the actual computation, parties generate **correlated randomness** in the form of:
- **Beaver multiplication triples:** random shares of (a, b, c) where c = a * b. These enable multiplication of secret-shared values in constant rounds during the online phase.
- **Authenticated shares:** each share carries a MAC under a global key alpha (itself secret-shared). Any modification to a share invalidates the MAC, which is detected during output reconstruction.

The offline phase is independent of the input data and can be precomputed. The MASCOT protocol (Keller et al., 2018) generates these triples efficiently using oblivious transfer, achieving ~1 million triples per second per pair of parties over a 10 Gbps network.

**Online phase.** Given the preprocessed correlated randomness:
- **Addition** of secret-shared values is free: parties locally add their shares. Communication cost: O(0).
- **Multiplication** consumes one Beaver triple and requires one round of communication (each party broadcasts one field element). Communication cost: O(n) per multiplication, where n is the number of parties.
- **Output reconstruction** involves opening the MAC check. If any party has deviated, the MAC check fails with overwhelming probability (2^{-sec_param}), and the protocol aborts.

### 9.3 SPDZ for Multi-Institutional Genomic Analysis

**Target computation: federated GWAS.** Each of k institutions holds genotype data for its cohort. The goal is to compute genome-wide association statistics (allele frequencies, odds ratios, chi-squared statistics, meta-analysis z-scores) without any institution revealing individual-level genotype data.

**Performance characteristics for genomic workloads.**

| Configuration | 3 institutions | 5 institutions | 10 institutions |
|---------------|---------------|----------------|-----------------|
| GWAS (100K SNPs, 100K individuals total) | ~2 hours | ~3.5 hours | ~8 hours |
| Allele frequency computation (1M SNPs) | ~30 minutes | ~50 minutes | ~2 hours |
| Beaver triple generation (offline, per M triples) | ~1 second | ~2 seconds | ~5 seconds |
| Communication per SNP (online phase) | ~128 bytes | ~256 bytes | ~640 bytes |
| Network requirement | 1 Gbps sufficient | 1 Gbps sufficient | 10 Gbps recommended |

**Protocol: SPDZ-GWAS.**

```
Protocol: SPDZ-FederatedGWAS
Participants: Institutions I_1, ..., I_k
Security: Active security against up to k-1 corrupted parties
Network: Authenticated channels between all pairs (mTLS)

Offline Phase (precompute, can run weeks before study):
  1. Generate Beaver triples via MASCOT OT protocol
     - Need ~3 triples per SNP per statistical test
     - For 100K SNPs: ~300K triples total
  2. Generate random authenticated shares for input masking
  3. Store correlated randomness locally at each institution

Online Phase:
  1. Input sharing: Each I_j secret-shares its genotype vector [g_j]
     using authenticated additive shares: [g_j] = (share_1, ..., share_k, mac_1, ..., mac_k)
  2. Allele counting: compute [sum_alleles] = [g_1] + ... + [g_k] (local addition, free)
  3. Frequency estimation: compute [freq] = [sum_alleles] * [1/N_total] (one multiplication, one triple)
  4. Chi-squared statistic:
     a. [observed] = [allele_count_case] (shared addition)
     b. [expected] = [total] * [case_freq] (one multiplication)
     c. [chi2_term] = ([observed] - [expected])^2 / [expected] (2 multiplications)
  5. Significance testing: compare [chi2] against public threshold
  6. Output: open only SNPs exceeding significance threshold (selective opening)
  7. MAC check: verify all opened values; abort if any MAC fails

Post-computation:
  8. Apply differential privacy noise to opened results (Section 2)
  9. Publish DP-protected significant associations
```

### 9.4 Combining SPDZ with Differential Privacy

SPDZ guarantees that individual genotype inputs remain hidden during computation. Differential privacy guarantees that the output statistics do not reveal information about any individual. Together, they provide **defense-in-depth**: even if the SPDZ protocol implementation has a subtle bug that leaks partial information, the DP noise provides a second independent layer of protection.

The combination is applied at the output stage: after SPDZ computes the exact (secret-shared) result, the parties jointly add DP noise using a distributed noise generation protocol (each party generates a share of the noise, and the combined noise has the correct Laplace or Gaussian distribution). The noise-added result is then opened (MAC-checked), while the exact result is never reconstructed.

### 9.5 Implementation: SpdZComputeCluster

```rust
/// SPDZ-based secure multi-party computation cluster for genomic analysis.
/// Provides active security against malicious adversaries.
pub struct SpdZComputeCluster {
    /// Number of participating institutions
    pub num_parties: usize,
    /// This institution's party index (0-indexed)
    pub party_index: usize,
    /// Security parameter (MAC key length in bits)
    pub security_parameter: usize,  // 128
    /// Pre-generated Beaver triples (offline phase output)
    pub beaver_triples: BeaverTripleStore,
    /// Global MAC key share (this party's share of alpha)
    pub mac_key_share: FieldElement,
    /// Network configuration for inter-party communication
    pub network: SpdZNetworkConfig,
    /// Differential privacy integration
    pub dp_config: Option<GenomicDpConfig>,
}

pub struct BeaverTripleStore {
    /// Pre-generated triples: (a_share, b_share, c_share, mac_a, mac_b, mac_c)
    pub triples: Vec<AuthenticatedTriple>,
    /// Number of unconsumed triples remaining
    pub remaining: usize,
    /// Alert threshold: warn when remaining triples drop below this
    pub low_watermark: usize,
}

pub struct SpdZNetworkConfig {
    /// Endpoints for all parties (mTLS-authenticated)
    pub party_endpoints: Vec<String>,
    /// Minimum network bandwidth (bytes/sec) required for target performance
    pub min_bandwidth: u64,
    /// Maximum round-trip latency (ms) for acceptable performance
    pub max_rtt_ms: u64,
}

impl SpdZComputeCluster {
    /// Secret-share a local genotype vector with authentication.
    pub fn share_input(&self, genotypes: &[u8]) -> Vec<AuthenticatedShare> {
        // Generate random shares and compute MACs
        unimplemented!("SPDZ authenticated input sharing")
    }

    /// Compute allele frequency across all institutions (online phase).
    /// Consumes Beaver triples for the division operation.
    pub fn federated_allele_frequency(
        &mut self,
        local_shares: &[AuthenticatedShare],
        total_individuals: usize,
    ) -> Result<Vec<AuthenticatedShare>, SpdZError> {
        // Addition is free (local), division requires triples
        unimplemented!("SPDZ online computation")
    }

    /// Open a secret-shared result with MAC verification.
    /// Returns None if MAC check fails (indicating adversarial behavior).
    pub fn verified_open(&self, shares: &[AuthenticatedShare]) -> Option<Vec<FieldElement>> {
        // Reconstruct value and verify MAC; abort on failure
        unimplemented!("SPDZ MAC-checked opening")
    }
}
```

**References.** Damgard, I., Pastro, V., Smart, N., & Warinschi, S. (2012). "Multiparty Computation from Somewhat Homomorphic Encryption." *CRYPTO*. Keller, M., Orsini, E., & Scholl, P. (2018). "MASCOT: Faster Malicious Arithmetic Secure Computation with Oblivious Transfer." *ACM CCS*.

---

## 10. Confidential Computing with TEE Attestation

### 10.1 Extended TEE Architecture

Section 7.1 introduced TEE platforms for genomic computation. This section specifies the **attestation-first architecture** required for trustworthy confidential computing: no genomic data enters a TEE until the TEE has been remotely attested, and every result carries a cryptographic proof of its provenance within an attested TEE.

### 10.2 Hardware Platforms and Security Guarantees

| Platform | Isolation Mechanism | Memory Encryption | Attestation Root | TCB Size | Max Enclave Memory |
|----------|-------------------|-------------------|-----------------|----------|-------------------|
| Intel SGX | Process-level enclaves | AES-128-CTR (MEE) | Intel Attestation Service (IAS) / DCAP | ~100 KB | 256 MB (SGX1), 1 TB (SGX2) |
| Intel TDX | VM-level trust domains | AES-128-XTS (MKTME) | Intel Trust Authority (ITA) | ~500 KB | No limit (full VM) |
| AMD SEV-SNP | VM-level encrypted VMs | AES-128-XTS per-page | AMD Key Distribution Service (KDS) | ~200 KB | No limit (full VM) |
| ARM CCA | Realm-level isolation | AES-XTS (RME) | ARM CCA Attestation Service | ~150 KB | Configurable |

**Performance overhead of memory encryption.**

| Operation | Plaintext | TEE (AES-XTS encrypted memory) | Overhead |
|-----------|-----------|-------------------------------|----------|
| Sequential memory read (1 GB) | 1.2s | 1.26s | ~5% |
| Random memory access (1M lookups) | 45ms | 48ms | ~7% |
| HNSW search (1M vectors, ef=200) | 2.1ms | 2.3ms | ~10% |
| Variant calling (30x WGS) | 4.2 hours | 4.4 hours | ~5% |

The ~5% overhead of hardware memory encryption is negligible compared to the cryptographic overhead of homomorphic encryption (8-10x, Section 3.2), making TEEs the preferred execution environment for Tier 2 data (encrypted at rest, decrypted inside TEE).

### 10.3 Remote Attestation Flow

Remote attestation enables a data owner (e.g., a patient or institution) to cryptographically verify that their genomic data will be processed by the expected code, on genuine hardware, inside a properly configured TEE, **before** sending any data.

```
Attestation Flow: TeeExecutionContext

Data Owner                     Compute Node (TEE)              Attestation Verifier
    |                               |                                   |
    |-- 1. Request attestation ---->|                                   |
    |                               |-- 2. Generate attestation ------->|
    |                               |   report (signed by TEE HW key)   |
    |                               |                                   |
    |                               |   Report contains:                |
    |                               |   - Platform identity (FMSPC)     |
    |                               |   - TCB version (microcode, FW)   |
    |                               |   - Code measurement (MRENCLAVE   |
    |                               |     or TD report)                 |
    |                               |   - User data (TEE public key)    |
    |                               |   - HW signature (ECDSA/RSA       |
    |                               |     from platform key)            |
    |                               |                                   |
    |<----- 3. Attestation report --+<-- 4. Verification result --------|
    |                               |   (checks against manufacturer's  |
    |                               |    root of trust, TCB status,     |
    |                               |    known-good code measurements)  |
    |                               |                                   |
    |-- 5. IF verified: send encrypted genome (under TEE public key) -->|
    |                               |                                   |
    |                               |-- 6. Decrypt inside TEE,          |
    |                               |      process, re-encrypt result   |
    |                               |      Sign result with TEE key     |
    |                               |                                   |
    |<----- 7. Encrypted result + signed attestation ------------------|
```

### 10.4 Practical Deployment: AMD SEV-SNP for Variant Calling

AMD SEV-SNP is the most practical TEE platform for genomic workloads because it encrypts an **entire VM** without requiring code modifications. The existing RuVector variant calling pipeline can run unmodified inside an SEV-SNP VM.

**Deployment configuration.**

```
AMD SEV-SNP VM for RuVector Genomic Pipeline:
  - vCPUs: 32 (AMD EPYC 9004 series)
  - Memory: 256 GB (encrypted, per-page tweaking)
  - Attestation: SEV-SNP attestation report verified against AMD KDS
  - Guest policy: no debugging, no migration, no key sharing
  - Memory encryption: AES-128-XTS with per-page ASID-based tweaking
  - Measured boot: entire VM image measured into SNP launch digest
  - Network: encrypted channels only (TLS 1.3 + hybrid PQ, Section 8.3)
```

**Pipeline execution inside SEV-SNP.**

1. VM boots with measured image containing RuVector variant calling pipeline
2. Remote attestation report generated and verified by data owner
3. Data owner sends encrypted genome (AES-256-GCM under key derived from attestation handshake)
4. Inside VM: decrypt genome, run BWA-MEM2 alignment, GATK HaplotypeCaller, VEP annotation
5. Results encrypted under data owner's key, signed with TEE-bound Dilithium3 key
6. VM scrubs all plaintext genomic data from memory (explicit memzero + page table flush)
7. Encrypted results + attestation chain returned to data owner

### 10.5 Sealed Storage for Persistent TEE State

TEEs provide **sealed storage**: data encrypted under a key derived from the platform identity and code measurement. Sealed data can only be decrypted by the same code running on the same platform (or a platform with the same measurement, depending on sealing policy).

For RuVector, sealed storage protects:
- CKKS secret keys that must persist across TEE restarts
- Privacy budget ledger state (to prevent budget reset attacks)
- Cached attestation verification results

```rust
/// TEE execution context with attestation and sealed storage.
pub struct TeeExecutionContext {
    /// TEE platform type
    pub platform: TeePlatform,
    /// Attestation report (generated by hardware)
    pub attestation_report: AttestationReport,
    /// TEE-bound signing key (Dilithium3, generated inside TEE)
    pub signing_key: PqSigningKey,
    /// Sealed storage handle (platform-specific)
    pub sealed_storage: SealedStorageHandle,
    /// Code measurement (hash of loaded binary)
    pub measurement: [u8; 48],
    /// Policy: what operations are allowed in this TEE context
    pub policy: TeePolicy,
}

pub struct AttestationReport {
    /// Raw platform attestation quote
    pub quote: Vec<u8>,
    /// Platform identity (vendor-specific)
    pub platform_id: Vec<u8>,
    /// TCB (Trusted Computing Base) version components
    pub tcb_version: TcbVersion,
    /// User-supplied data bound into the quote (typically a public key hash)
    pub user_data: [u8; 64],
    /// Verification timestamp
    pub verified_at: u64,
    /// Verification result from attestation service
    pub verification_status: AttestationStatus,
}

pub enum AttestationStatus {
    /// Attestation verified, TCB is up-to-date
    UpToDate,
    /// Attestation verified, but TCB needs update (SW-hardening available)
    SwHardeningNeeded,
    /// Attestation verified, but TCB is out-of-date (configuration needed)
    ConfigurationNeeded,
    /// Attestation verification failed
    Failed { reason: String },
}

pub struct TeePolicy {
    /// Allow debugging (MUST be false for production genomic workloads)
    pub allow_debug: bool,
    /// Allow VM migration (MUST be false to prevent key extraction)
    pub allow_migration: bool,
    /// Minimum TCB version required
    pub min_tcb_version: TcbVersion,
    /// Allowed code measurements (whitelist of known-good binaries)
    pub allowed_measurements: Vec<[u8; 48]>,
    /// Maximum plaintext data retention time (seconds) before mandatory scrub
    pub max_plaintext_retention_secs: u64,
}
```

---

## 11. Genomic Data Sovereignty and Federated Governance

### 11.1 Principle: Data Never Leaves Jurisdiction

Genomic data sovereignty mandates that **data never crosses jurisdictional boundaries**; instead, computation travels to the data. This is not merely a policy preference but a legal requirement under multiple regulatory frameworks:

- **GDPR Chapter V** (Articles 44-49): transfers of personal data to third countries require adequacy decisions, Standard Contractual Clauses (SCCs), or Binding Corporate Rules (BCRs).
- **China's Personal Information Protection Law (PIPL)**: genetic data classified as sensitive personal information; cross-border transfer requires security assessment by the Cyberspace Administration.
- **Brazil's LGPD**: genetic data is sensitive; international transfers require explicit consent or adequacy finding.
- **India's Digital Personal Data Protection Act (2023)**: genetic data may be restricted from transfer to specified jurisdictions.

For multi-national genomic studies, the federated architecture ensures each institution's RuVector node processes its local data and shares only **aggregated, encrypted, or differentially private results** across borders.

### 11.2 Federated Analytics Architecture

```
Jurisdiction A (EU)          Jurisdiction B (US)          Jurisdiction C (JP)
+-------------------+        +-------------------+        +-------------------+
| RuVector Node A   |        | RuVector Node B   |        | RuVector Node C   |
|                   |        |                   |        |                   |
| [Local Genomes]   |        | [Local Genomes]   |        | [Local Genomes]   |
| [Local TEE]       |        | [Local TEE]       |        | [Local TEE]       |
| [Local DP Budget] |        | [Local DP Budget] |        | [Local DP Budget] |
|                   |        |                   |        |                   |
| Run local GWAS    |        | Run local GWAS    |        | Run local GWAS    |
| statistics        |        | statistics        |        | statistics        |
|                   |        |                   |        |                   |
| Output: encrypted |        | Output: encrypted |        | Output: encrypted |
| summary stats     |        | summary stats     |        | summary stats     |
+--------+----------+        +--------+----------+        +--------+----------+
         |                            |                            |
         +------- SPDZ / TFHE -------+------- SPDZ / TFHE -------+
                                      |
                            [Meta-Analysis Coordinator]
                            - Operates on encrypted aggregates only
                            - Never sees individual genotypes
                            - Produces DP-protected final results
                            - Results publishable in any jurisdiction
```

### 11.3 Governance Model: Smart Contract Data Use Agreements

Traditional data use agreements (DUAs) are paper documents that are difficult to enforce programmatically. RuVector encodes DUA terms as **machine-readable smart contracts** stored on an append-only audit chain (extending the hash-chained witness log of Section 5.3):

```rust
/// Machine-readable data use agreement encoded as a smart contract.
/// Stored on append-only audit chain for tamper-evident enforcement.
pub struct DataUseAgreement {
    /// Unique agreement identifier
    pub agreement_id: String,
    /// Parties to the agreement
    pub parties: Vec<GovernanceParty>,
    /// Data scope: which genomic data is covered
    pub data_scope: Vec<GenomicAccessScope>,
    /// Permitted purposes (enumerated, not open-ended)
    pub permitted_purposes: Vec<ConsentPurpose>,
    /// Prohibited uses (explicit deny list)
    pub prohibited_uses: Vec<ProhibitedUse>,
    /// Jurisdiction constraints
    pub jurisdiction_constraints: Vec<JurisdictionConstraint>,
    /// Duration and termination conditions
    pub duration: AgreementDuration,
    /// Cryptographic signatures from all parties
    pub signatures: Vec<GovernanceSignature>,
    /// Hash of this agreement (for audit chain inclusion)
    pub agreement_hash: [u8; 32],
}

pub struct JurisdictionConstraint {
    /// Jurisdiction where data may reside (ISO 3166-1 alpha-2 country codes)
    pub allowed_jurisdictions: Vec<String>,
    /// Legal basis for any cross-border transfer
    pub transfer_mechanism: Option<TransferMechanism>,
    /// Whether data must be deleted after computation (ephemeral processing)
    pub ephemeral_only: bool,
}

pub enum TransferMechanism {
    /// EU adequacy decision (Article 45)
    AdequacyDecision { decision_reference: String },
    /// Standard Contractual Clauses (Article 46(2)(c))
    StandardContractualClauses { version: String },
    /// Binding Corporate Rules (Article 47)
    BindingCorporateRules { bcr_reference: String },
    /// Explicit consent for specific transfer (Article 49(1)(a))
    ExplicitConsent { consent_id: String },
    /// No transfer permitted (data stays in jurisdiction)
    NoTransfer,
}

pub enum ProhibitedUse {
    /// Insurance underwriting or risk assessment
    InsuranceUnderwriting,
    /// Employment decisions
    EmploymentDecisions,
    /// Law enforcement without court order
    LawEnforcementWithoutCourtOrder,
    /// Re-identification of de-identified data
    ReIdentification,
    /// Sharing with unlisted third parties
    ThirdPartySharing,
    /// Use beyond specified research purpose
    PurposeCreep,
}
```

### 11.4 Sovereignty Router

The `SovereigntyRouter` is a network-layer component that enforces data residency constraints at the infrastructure level, preventing genomic data from being routed to nodes in unauthorized jurisdictions.

```rust
/// Enforces data residency constraints by routing computation to data,
/// never data to computation. Integrates with RuVector's mesh network.
pub struct SovereigntyRouter {
    /// Map of node IDs to their physical jurisdiction
    pub node_jurisdictions: HashMap<String, Jurisdiction>,
    /// Active data use agreements (defines what can flow where)
    pub active_agreements: Vec<DataUseAgreement>,
    /// Routing rules derived from agreements and regulations
    pub routing_rules: Vec<RoutingRule>,
    /// Audit log for all routing decisions
    pub routing_audit: Vec<RoutingDecision>,
}

pub struct Jurisdiction {
    /// ISO 3166-1 alpha-2 country code
    pub country_code: String,
    /// Specific region/state if relevant (e.g., "CA" for California CCPA)
    pub region: Option<String>,
    /// Applicable regulations in this jurisdiction
    pub applicable_regulations: Vec<RegulationId>,
    /// Whether this jurisdiction has EU GDPR adequacy decision
    pub gdpr_adequate: bool,
}

pub struct RoutingRule {
    /// Source jurisdiction
    pub from: String,
    /// Destination jurisdiction
    pub to: String,
    /// What may cross this boundary
    pub allowed_data_types: Vec<AllowedCrossBorderData>,
    /// Legal basis
    pub legal_basis: TransferMechanism,
}

pub enum AllowedCrossBorderData {
    /// Only differentially private aggregate statistics
    DpAggregates { max_epsilon: f64 },
    /// Only encrypted data (TFHE/CKKS ciphertexts)
    EncryptedOnly,
    /// Only ZK proofs (no underlying data)
    ZkProofsOnly,
    /// Only SPDZ secret shares (no reconstructable data)
    SecretSharesOnly,
    /// No data may cross this boundary
    Nothing,
}

impl SovereigntyRouter {
    /// Determine whether a data transfer is permitted and by what mechanism.
    pub fn can_transfer(
        &self,
        data_type: &AllowedCrossBorderData,
        from_node: &str,
        to_node: &str,
    ) -> RoutingDecision {
        let from_jurisdiction = &self.node_jurisdictions[from_node];
        let to_jurisdiction = &self.node_jurisdictions[to_node];

        // Check applicable routing rules
        // Log decision to routing_audit
        // Return permit/deny with legal basis
        unimplemented!("Jurisdiction-aware routing decision")
    }

    /// Route a computation request to the appropriate node(s) based on
    /// where the data resides, rather than moving data to the computation.
    pub fn route_computation(
        &self,
        computation: &ComputationRequest,
        required_data_nodes: &[String],
    ) -> RoutingPlan {
        // Generate a plan that sends computation code to data nodes
        // and aggregates results using SPDZ or TFHE
        unimplemented!("Computation-to-data routing plan")
    }
}
```

### 11.5 Cross-Border Transfer Decision Matrix

| Source | Destination | Genomic Data | DP Aggregates | Encrypted Data | ZK Proofs | SPDZ Shares |
|--------|-------------|-------------|---------------|----------------|-----------|-------------|
| EU | EU | Permitted | Permitted | Permitted | Permitted | Permitted |
| EU | US (no adequacy) | **Blocked** | Permitted (SCCs) | Permitted (SCCs) | Permitted | Permitted (SCCs) |
| EU | UK (adequacy) | Permitted | Permitted | Permitted | Permitted | Permitted |
| EU | Japan (adequacy) | Permitted | Permitted | Permitted | Permitted | Permitted |
| US | US | Permitted | Permitted | Permitted | Permitted | Permitted |
| US | EU | Permitted (DPF) | Permitted | Permitted | Permitted | Permitted |
| Any | Any | **Blocked** | Permitted | Permitted | Permitted | Permitted |

The default rule (last row) ensures that raw genomic data never crosses jurisdictional boundaries unless there is a specific legal basis. Processed outputs (DP aggregates, encrypted data, ZK proofs, SPDZ shares) may flow more freely because they do not constitute personal data transfer under most frameworks (they cannot be used to identify individuals without additional information that is not transferred).

---

## 12. Implementation Priorities

### Phase 1 (Weeks 1-4): Foundation

| Task | Priority | Dependency |
|------|----------|------------|
| Implement GenomicAccessScope and claims extensions | P0 | ADR-010 claims system |
| Deploy GenomicAccessWitness audit log | P0 | ADR-CE-017 witness log |
| Implement privacy budget ledger | P0 | Existing DP module |
| Define encryption tier classification policy | P1 | ClinVar/OMIM integration |
| Implement cryptographic deletion protocol | P1 | Key management infrastructure |

### Phase 2 (Weeks 5-8): Cryptographic Infrastructure

| Task | Priority | Dependency |
|------|----------|------------|
| CKKS integration for Tier 1 vector operations | P0 | Scheme parameter selection |
| Groth16 circuit for genotype predicates | P1 | arkworks-rs integration |
| TEE attestation chain implementation | P1 | Hardware availability |
| HNSW constant-time traversal mode | P2 | Core HNSW refactor |

### Phase 3 (Weeks 9-12): Protocol Integration

| Task | Priority | Dependency |
|------|----------|------------|
| Secure multi-party aggregation protocol | P1 | Institutional partnerships |
| End-to-end encrypted analysis pipeline | P1 | Phase 2 completion |
| GDPR consent management system | P0 | Legal review |
| Compliance certification preparation | P1 | All phases |

### Phase 4 (Weeks 13-18): SOTA Security Enhancements

| Task | Priority | Dependency |
|------|----------|------------|
| Deploy hybrid PQ key exchange (X25519 + Kyber768) for all TLS | P0 | NIST FIPS 203 library integration |
| Replace Ed25519 operational signatures with Dilithium3 | P0 | NIST FIPS 204 library integration |
| Issue SPHINCS+ root CA certificate | P1 | HSM firmware update for PQ support |
| Re-encapsulate all DEKs under Kyber768 hybrid | P1 | Phase 4 PQ deployment |
| Implement RenyiPrivacyAccountant with PRV accounting | P1 | Phase 1 privacy budget ledger |
| Deploy Plonk universal SRS for new predicate circuits | P1 | Phase 2 Groth16 infrastructure |
| Implement TFHE genomic compute for Tier 1 cross-institutional analysis | P2 | Concrete/TFHE-rs integration |
| Deploy SPDZ compute cluster for federated GWAS | P2 | Institutional network agreements |
| Implement Halo2 recursive proof aggregation | P2 | Phase 2 ZK infrastructure |
| Deploy SovereigntyRouter for jurisdiction-aware data routing | P1 | Node jurisdiction registration |
| Implement TeeExecutionContext with full attestation flow | P1 | AMD SEV-SNP / Intel TDX hardware provisioning |
| Encode data use agreements as machine-readable smart contracts | P2 | Legal review of DUA templates |

---

## Consequences

### Benefits
- Genomic data protected by defense-in-depth: encryption, access control, differential privacy, and zero-knowledge proofs
- Compliance with all major regulatory frameworks (HIPAA, GINA, GDPR, FDA 21 CFR Part 11)
- Cryptographic deletion provides a technically sound implementation of the right to erasure
- ZK proofs enable new use cases (pharmacogenomic safety checks, carrier screening) without requiring full genotype disclosure
- Post-quantum cryptography ensures genomic data remains secure against quantum adversaries for the 100+ year shelf life of the data
- Renyi DP accounting enables approximately 3x more research queries within the same privacy budget
- SPDZ protocol provides active security for multi-institutional studies, protecting against malicious adversaries
- Confidential computing with TEE attestation enables secure processing with only ~5% overhead
- Data sovereignty architecture ensures compliance with cross-border data transfer regulations worldwide
- Plonk/Halo2 ZK backends eliminate per-circuit trusted setup and enable recursive proof composition

### Risks
- CKKS overhead may exceed 10x target for complex multi-locus queries requiring deep multiplicative circuits
- Trusted setup ceremony for Groth16 introduces a trust assumption (mitigable via MPC-based setup or migration to Plonk/Halo2)
- TEE availability varies across cloud providers; must maintain software-only fallback
- Privacy budget exhaustion may frustrate researchers; requires clear communication of budget policies
- Post-quantum key sizes increase wire overhead (~2.3 KB per handshake for hybrid KEM); negligible for genomic payloads but measurable for high-frequency API calls
- TFHE ciphertext expansion (8,000x for single genotype) limits applicability to targeted loci, not whole-genome computation
- SPDZ requires pre-generated Beaver triples; offline phase must be planned in advance of computation
- SPHINCS+ signatures (35 KB) are impractical for high-frequency operations; reserved for archival/root CA use only
- Sovereignty routing adds latency for cross-jurisdictional federated studies (computation must travel to each data node)

### Breaking Changes
- All genomic data access now requires claims with GenomicAccessScope; existing API consumers must migrate
- Audit log schema change from generic WitnessLog to GenomicAccessWitness
- Embedding model outputs for Tier 1 regions are encrypted; downstream consumers must support CKKS ciphertext handling
- TLS endpoints will require hybrid PQ key exchange support; clients without PQ-capable TLS libraries must upgrade
- All new operational signatures use Dilithium3 (3,293 bytes) instead of Ed25519 (64 bytes); signature verification code must be updated

---

## References

1. Homer, N., et al. (2008). "Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures Using High-Density SNP Genotyping Microarrays." *PLoS Genetics*.
2. Gymrek, M., et al. (2013). "Identifying Personal Genomes by Surname Inference." *Science*.
3. Erlich, Y., & Narayanan, A. (2014). "Routes for Breaching and Protecting Genetic Privacy." *Nature Reviews Genetics*.
4. Dwork, C., Roth, A., & Vadhan, S. (2010). "Boosting and Differential Privacy." *FOCS*.
5. Cheon, J.H., Kim, A., Kim, M., & Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers." *ASIACRYPT*.
6. Groth, J. (2016). "On the Size of Pairing-Based Non-interactive Arguments." *EUROCRYPT*.
7. NIST SP 800-188: De-Identifying Government Datasets.
8. OWASP Genomic Data Security Guidelines (2025).
9. GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data.
10. Mironov, I. (2017). "Renyi Differential Privacy of the Sampled Gaussian Mechanism." *CSF*.
11. Dong, J., Roth, A., & Su, W.J. (2022). "Gaussian Differential Privacy." *JRSS-B*.
12. Balle, B., Cherubin, G., & Warber, J. (2020). "Hypothesis Testing Interpretations and Renyi Differential Privacy." *AISTATS*.
13. Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2020). "TFHE: Fast Fully Homomorphic Encryption over the Torus." *Journal of Cryptology*.
14. Damgard, I., Pastro, V., Smart, N., & Warinschi, S. (2012). "Multiparty Computation from Somewhat Homomorphic Encryption." *CRYPTO*.
15. Keller, M., Orsini, E., & Scholl, P. (2018). "MASCOT: Faster Malicious Arithmetic Secure Computation with Oblivious Transfer." *ACM CCS*.
16. Gabizon, A., Williamson, Z., & Ciobotaru, O. (2019). "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge." *IACR ePrint*.
17. Bowe, S., Grigg, J., & Hopwood, D. (2019). "Halo: Recursive Proof Composition without a Trusted Setup." *IACR ePrint*.
18. NIST FIPS 203 (2024). "Module-Lattice-Based Key-Encapsulation Mechanism Standard."
19. NIST FIPS 204 (2024). "Module-Lattice-Based Digital Signature Standard."
20. NIST FIPS 205 (2024). "Stateless Hash-Based Digital Signature Standard."
21. Mosca, M. (2018). "Cybersecurity in an Era with Quantum Computers: Will We Be Ready?" *IEEE Security & Privacy*.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Security Architecture Team | Initial proposal |
| 0.2 | 2026-02-11 | Security Architecture Team | SOTA enhancements: post-quantum cryptography (FIPS 203/204/205, hybrid KEM), TFHE bootstrapping for exact encrypted computation, SPDZ protocol for active-secure MPC, Renyi/GDP/f-DP for tight privacy composition, Plonk/Halo2 universal and recursive ZK proofs, confidential computing with TEE attestation architecture, genomic data sovereignty and federated governance with sovereignty routing |
