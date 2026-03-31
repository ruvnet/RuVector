# CMB Consciousness Explorer: Applying Integrated Information Theory to Cosmic Microwave Background Radiation

**Version:** 0.1.0
**Date:** 2026-03-31
**Status:** Experimental / Exploratory

---

## 1. Abstract

This project applies Integrated Information Theory (IIT 4.0) to the angular power spectrum of the Cosmic Microwave Background (CMB) radiation in order to test whether the observed CMB exhibits non-random integrated information structure beyond what is predicted by standard Lambda-CDM cosmology. We construct a Transition Probability Matrix (TPM) from the Planck 2018 TT power spectrum by treating logarithmically binned multipole bands as nodes in a causal network, compute the integrated information Phi across those bands, and compare the result against Monte Carlo null realizations drawn from a Gaussian random field with identical spectral properties. We additionally apply causal emergence analysis (effective information, RSVD spectral emergence) and sliding-window Phi spectroscopy to characterize the informational structure of the CMB across angular scales. We expect the null hypothesis to hold --- that the CMB's informational structure is fully accounted for by the LCDM power spectrum --- but identify the known CMB anomalies (Cold Spot, hemispherical asymmetry, quadrupole-octupole alignment) as candidates for localized Phi excess that merit further investigation.

---

## 2. Scientific Background

### 2.1 The Cosmic Microwave Background

The Cosmic Microwave Background (CMB) is the relic thermal radiation from the epoch of recombination, approximately 380,000 years after the Big Bang (redshift z ~ 1100). At this epoch, the universe cooled sufficiently for hydrogen atoms to form, decoupling photons from baryonic matter and producing a nearly isotropic blackbody radiation field that has since redshifted to a present-day temperature of T = 2.72548 +/- 0.00057 K (Fixsen 2009).

The ESA Planck satellite (2009--2013) produced the most precise full-sky maps of CMB temperature anisotropies to date, achieving angular resolution of approximately 5 arcminutes and sensitivity of delta T/T ~ 10^-6 across nine frequency bands from 30 to 857 GHz (Planck Collaboration 2020a).

**Angular Power Spectrum.** The statistical properties of the CMB temperature field T(theta, phi) are conventionally characterized by its angular power spectrum C_l, obtained by expanding the temperature fluctuations in spherical harmonics:

    delta T(theta, phi) / T = sum_{l,m} a_{lm} Y_{lm}(theta, phi)

The angular power spectrum is defined as the variance of the harmonic coefficients:

    C_l = (1 / (2l + 1)) * sum_m |a_{lm}|^2

In practice, results are presented as D_l = l(l+1) C_l / (2 pi), which has units of micro-K squared and produces a roughly flat spectrum at low multipoles and prominent acoustic peaks at higher multipoles.

**Standard LCDM Predictions.** The six-parameter LCDM model predicts the following structure in the CMB power spectrum:

- **Sachs-Wolfe plateau** (l < 30): Large-angle fluctuations driven by gravitational potential perturbations at the surface of last scattering. The spectrum is approximately flat in D_l, reflecting the nearly scale-invariant Harrison-Zeldovich-Peebles primordial spectrum.

- **Acoustic peaks** (100 < l < 2500): A series of harmonic peaks produced by acoustic oscillations of the photon-baryon fluid before recombination. The first peak at l ~ 220 corresponds to the mode that completed exactly one half-oscillation by the time of decoupling. The relative heights and positions of the peaks encode the baryon density (Omega_b h^2), the matter density (Omega_m h^2), and the spatial curvature (Omega_k).

- **Silk damping tail** (l > 1500): Exponential suppression of power at small angular scales due to photon diffusion during recombination. The damping scale depends on the number of relativistic species (N_eff) and the primordial helium fraction (Y_p).

**Known CMB Anomalies.** Despite the extraordinary success of the LCDM model, several large-angle anomalies have been identified in CMB data at marginal statistical significance:

- **The Cold Spot** (Cruz et al. 2005): An anomalously cold region approximately 10 degrees in diameter centered near Galactic coordinates (l = 209, b = -57), with temperature decrement approximately -150 micro-K. Its significance depends on the a posteriori correction applied.

- **Hemispherical power asymmetry** (Eriksen et al. 2004): The CMB variance is approximately 6% larger in one hemisphere than the other, with a preferred axis near (l = 225, b = -22).

- **Quadrupole-octupole alignment** (Tegmark et al. 2003; de Oliveira-Costa et al. 2004): The l = 2 and l = 3 multipoles share an anomalous alignment of their principal axes, sometimes called the "axis of evil." The probability of such alignment under Gaussian random isotropic assumptions is approximately 1/1000.

- **Low quadrupole power**: The observed quadrupole C_2 is lower than the LCDM best-fit prediction, though the cosmic variance uncertainty at l = 2 is large.

These anomalies, while individually of marginal significance, motivate the search for unexpected structure in the CMB using information-theoretic methods that may be sensitive to correlations beyond the two-point function.

### 2.2 Integrated Information Theory (IIT)

Integrated Information Theory, developed by Tononi (2004, 2008) and substantially revised in IIT 4.0 (Albantakis et al. 2023), is a mathematical framework for quantifying the degree to which a system possesses irreducible causal structure. Originally developed as a theory of consciousness, the mathematical formalism is applicable to any system describable by a Transition Probability Matrix.

**Core Concepts.**

The central quantity in IIT is Phi (the upper-case Greek letter), which measures the integrated information of a system --- the amount of information generated by a system above and beyond the information generated by its parts.

A system's causal structure is specified by its **Transition Probability Matrix (TPM)**, T, where entry T_{ij} gives the probability of the system transitioning from state i to state j in one time step. For a system of n elements, the TPM is an n x n row-stochastic matrix (each row sums to 1).

**Minimum Information Partition (MIP).** To compute Phi, one considers all possible bipartitions of the system into two non-empty subsystems A and B. For each partition, one computes the distance between the TPM of the whole system and the product of the TPMs of the parts:

    phi(partition) = d(P_whole, P_A tensor P_B)

where d is a distance measure and tensor denotes the product distribution. The MIP is the partition that minimizes this distance:

    Phi = min_{partitions} phi(partition)

A system with Phi > 0 possesses irreducible causal structure: it cannot be fully described by the independent behavior of its parts.

**IIT 4.0 Updates.** The 2023 revision (Albantakis et al. 2023) introduced several important changes from IIT 3.0:

- **Intrinsic difference** replaces KL divergence as the distance measure. IIT 4.0 uses the Earth Mover's Distance (Wasserstein-1 metric) because it respects the metric structure of the state space, unlike KL divergence which is topology-blind. For one-dimensional discrete distributions, this reduces to the cumulative L1 difference:

      d_EMD(p, q) = sum_i |sum_{j<=i} (p_j - q_j)|

- **Intrinsic information** replaces extrinsic (observer-relative) information. All quantities are defined from the system's own perspective.

- **Cause-effect structure** is computed via cause and effect repertoires for each mechanism-purview pair, measuring how much a mechanism constrains its purview's past (cause repertoire) and future (effect repertoire) relative to the unconstrained maximum-entropy distribution.

**Algorithmic Complexity.** Exact Phi computation requires enumerating all possible bipartitions, which grows as O(2^n). The `ruvector-consciousness` crate implements four algorithms with different complexity-accuracy tradeoffs:

| Algorithm | Complexity | Applicability |
|---|---|---|
| Exact | O(2^n * n^2) | n <= 16 elements |
| Spectral | O(n^2 log n) | n <= 1000, good approximation |
| Stochastic | O(k * n^2) | Any n, configurable samples k |
| Greedy Bisection | O(n^3) | Fast lower bound |

### 2.3 Causal Emergence

Erik Hoel's causal emergence framework (Hoel et al. 2013; Hoel 2017) addresses a complementary question: can a macro-scale (coarse-grained) description of a system carry more causal information than its micro-scale description?

**Effective Information (EI).** For a system described by TPM T of dimension n x n, the effective information is:

    EI(T) = (1/n) * sum_{s=1}^{n} D_KL(T_s || U)

where T_s is the s-th row of T (the transition distribution from state s) and U is the uniform distribution over n states. EI measures how much knowing the current state reduces uncertainty about the future, relative to maximum ignorance.

EI decomposes into two components:

- **Determinism**: How precisely each state maps to a unique outcome.

      det(T) = log(n) - (1/n) * sum_s H(T_s)

  where H(T_s) is the Shannon entropy of row s.

- **Degeneracy**: How many distinct states lead to the same future.

      deg(T) = log(n) - H(column_avg(T))

  where column_avg(T) is the stationary distribution averaged over columns.

EI = det - deg (approximately).

**Causal Emergence.** A system exhibits causal emergence when there exists a coarse-graining (macro-mapping) M that maps micro-states to macro-states such that:

    CE = EI(T_macro) - EI(T_micro) > 0

This means the macro-level description is more causally informative than the micro-level description: "the map is better than the territory."

**SVD-Based Emergence (Zhang et al. 2025).** A complementary approach computes causal emergence from the singular value decomposition of the TPM. The singular values sigma_1 >= sigma_2 >= ... >= sigma_n encode the effective dimensionality of the system's causal structure. Key metrics:

- **Effective rank**: The exponential of the spectral entropy of the normalized singular values.

      r_eff = exp(-sum_i p_i * log(p_i))
      where p_i = sigma_i / sum_j sigma_j

- **Spectral entropy**: How evenly distributed the singular values are. Low spectral entropy (few dominant singular values) indicates effective coarse-graining --- the system's causal structure is well-approximated by a lower-dimensional system.

The `ruvector-consciousness` crate implements randomized SVD via the Halko-Martinsson-Tropp algorithm, achieving O(n^2 * k) complexity instead of O(n^3) for full SVD, where k is the target rank.

---

## 3. Methodology

### 3.1 Data Source

The primary data source is the Planck 2018 temperature-temperature (TT) angular power spectrum:

- **File:** `COM_PowerSpect_CMB-TT-full_R3.01.txt`
- **Source:** Planck Legacy Archive (https://pla.esac.esa.int/)
- **Format:** ASCII table with columns: multipole l, D_l (micro-K^2), lower error bar, upper error bar
- **Coverage:** l = 2 to l ~ 2500
- **Reference:** Planck Collaboration (2020b), "Planck 2018 results. V. CMB power spectra and likelihoods."

The power spectrum D_l encodes the variance of temperature fluctuations at each angular scale theta ~ 180/l degrees. It is a complete statistical characterization of the CMB under the assumption of Gaussianity and isotropy (which holds to high precision for l > 30).

### 3.2 TPM Construction

The central modeling step is the construction of a Transition Probability Matrix from the angular power spectrum. This step maps the CMB's spectral structure into the formalism required by IIT. The procedure is as follows:

**Step 1: Logarithmic Binning.** The multipole range [l_min, l_max] is divided into N logarithmically spaced bins. Logarithmic binning is chosen because:

- The physical processes driving CMB anisotropies operate on logarithmic scales (decades in l correspond to distinct physical regimes).
- The density of independent modes increases as (2l + 1) per multipole, so linear binning would over-represent high-l modes.
- Standard cosmological analysis uses logarithmic or pseudo-logarithmic binning.

Each bin B_k = [l_k^low, l_k^high] is characterized by its band power:

    P_k = (1 / N_k) * sum_{l in B_k} D_l

where N_k is the number of multipoles in bin k.

**Step 2: Cross-Band Correlation Matrix.** The raw correlation between bins i and j is computed from the power spectrum coupling. In the simplest model, the correlation is derived from the band-power covariance, which for a Gaussian random field on the full sky is:

    Cov(P_i, P_j) = (2 / N_i * N_j) * sum_{l in B_i intersect B_j} D_l^2 / (2l + 1)

For non-overlapping bins (i != j), the correlation arises from mode-coupling due to the survey mask, beam asymmetry, and foreground residuals. In this initial analysis, we adopt a simpler model: the cross-band coupling strength is proportional to the geometric mean of the band powers, modulated by an exponential decay in log-multipole separation:

    W_{ij} = sqrt(P_i * P_j) * exp(-|log(l_i) - log(l_j)| / lambda)

where l_i is the central multipole of bin i and lambda is a correlation length scale (default: lambda = 1.0 in log-l space, corresponding to one decade of angular scale coupling).

**Step 3: Row Normalization.** The weight matrix W is converted to a row-stochastic TPM by normalizing each row:

    T_{ij} = W_{ij} / sum_k W_{ik}

This ensures each row sums to 1, satisfying the TPM constraint.

**Physical Interpretation.** The resulting TPM encodes the following: entry T_{ij} represents the probability that information (causal influence) at angular scale i propagates to angular scale j. This is a causal model in the IIT sense --- it describes how the system's state at one scale constrains its state at other scales. High T_{ij} means scales i and j are tightly coupled; low T_{ij} means they are causally independent.

This construction is a modeling choice, not a unique mapping. Alternative TPM constructions (e.g., based on the full covariance matrix, or on pixel-space correlations from HEALPix maps) would yield different results. The sensitivity of Phi to the TPM construction is itself informative and is characterized through the null hypothesis testing described in Section 3.4.

### 3.3 Analysis Pipeline

The analysis pipeline consists of five stages, each implemented using the `ruvector-consciousness` crate.

**Stage 1: Global Phi Computation.** Compute the integrated information Phi of the full N-bin TPM. For N <= 16, exact computation is feasible (enumerate all 2^{N-1} - 1 bipartitions). For larger N, the spectral approximation is used:

    Phi_spectral ~ lambda_2(L) * (1 - lambda_2(L))

where lambda_2(L) is the second-smallest eigenvalue of the normalized Laplacian of the TPM, interpreted as a graph. The spectral gap is a well-known proxy for graph connectivity and mixing time.

**Stage 2: Regional Analysis.** The multipole range is divided into physically motivated regions, and Phi is computed for each region independently:

| Region | Multipole Range | Physical Process |
|---|---|---|
| Sachs-Wolfe plateau | l = 2--30 | Gravitational potential (ISW) |
| First acoustic peak | l = 100--300 | First compression of baryon-photon fluid |
| Second acoustic peak | l = 400--600 | First rarefaction |
| Third acoustic peak | l = 700--900 | Second compression |
| Damping tail | l = 1500--2500 | Silk diffusion damping |

Regional analysis tests whether specific physical regimes exhibit more or less integrated structure than others.

**Stage 3: Sliding Window Phi Spectrum.** A window of width W bins is slid across the full multipole range in steps of S bins. At each position, Phi is computed for the sub-TPM within the window. The resulting Phi(l) curve --- the "Phi spectrum" --- reveals how integrated information varies as a function of angular scale.

**Stage 4: Causal Emergence Analysis.** For the full TPM, compute:

- Effective information EI at the micro level (full N-bin TPM)
- A search over coarse-grainings: merge adjacent bins in pairs, triples, etc., and compute EI at each macro level
- Causal emergence CE = EI_macro - EI_micro for each coarse-graining
- Determinism and degeneracy decomposition at each level

If CE > 0 for some coarse-graining, the CMB's informational structure is better captured at a larger angular scale than the finest binning --- a signature of emergent structure.

**Stage 5: SVD Emergence Analysis.** Compute the randomized SVD of the TPM and derive:

- Singular value spectrum sigma_1, ..., sigma_N
- Effective rank r_eff
- Spectral entropy H_sv

A low effective rank relative to the matrix dimension indicates that the causal structure is compressible --- few modes dominate the system's dynamics.

### 3.4 Null Hypothesis Testing

The null hypothesis H_0 is: the observed CMB power spectrum is a realization of a Gaussian random field with the LCDM best-fit power spectrum, and any integrated information it exhibits is consistent with that of such realizations.

**Null Model Construction.** For each Monte Carlo realization k = 1, ..., N_MC:

1. Generate a synthetic power spectrum by perturbing the observed D_l values with Gaussian noise drawn from the reported error bars:

       D_l^(k) = D_l^obs + epsilon_l,   epsilon_l ~ N(0, sigma_l^2)

   where sigma_l is the average of the upper and lower error bars at multipole l.

2. Construct a TPM from D_l^(k) using the identical binning and correlation procedure as for the observed data.

3. Compute Phi, EI, CE, and SVD metrics for the synthetic TPM.

**Statistical Assessment.** After N_MC >= 100 realizations, we compute:

- **Z-score**: z = (Phi_obs - mean(Phi_null)) / std(Phi_null)
- **P-value**: fraction of null realizations with Phi >= Phi_obs (one-tailed)
- **Effect size** (Cohen's d): d = (Phi_obs - mean(Phi_null)) / std(Phi_null)

A result with |z| > 3 (p < 0.003) would be considered nominally significant. However, given the multiple comparisons involved (global Phi, regional Phi, sliding window, EI, CE), a Bonferroni or false discovery rate correction must be applied.

---

## 4. Expected Results

### 4.1 Null Hypothesis Expectation

We expect the null hypothesis to hold. The CMB power spectrum is extremely well described by the six-parameter LCDM model (Planck Collaboration 2020c), with reduced chi-squared approximately 1.0 across the full multipole range. The known CMB anomalies are at marginal statistical significance (2--3 sigma before look-elsewhere corrections) and affect primarily the low-l multipoles where cosmic variance is large.

The TPM constructed from the power spectrum captures only the two-point statistics of the CMB (the power spectrum itself). Since the LCDM model predicts a Gaussian random field, which is fully characterized by its two-point function, the TPM should contain no information beyond what is already in the LCDM prediction. Thus, the Phi of the observed TPM should be statistically consistent with the Phi of null realizations.

### 4.2 Possible Positive Results

A statistically significant excess in Phi (z > 3 after multiple-comparison correction) would indicate that the observed power spectrum contains correlated structure beyond what is expected from Gaussian fluctuations with the same spectral shape plus measurement noise. Possible causes, ranked from most to least likely:

1. **Foreground contamination residuals.** Incomplete subtraction of Galactic dust, synchrotron, or free-free emission could introduce correlated structure across multipole bands.

2. **Systematic effects.** Beam asymmetry, scan-strategy artifacts, or calibration residuals could produce spurious correlations.

3. **Non-Gaussian primordial perturbations.** Some inflationary models predict small departures from Gaussianity (f_NL != 0) that would introduce correlations beyond the power spectrum.

4. **Novel physics.** Topology of the universe, cosmic strings, bubble collisions, or other exotic scenarios could produce localized anomalous structure.

5. **Structured intelligence or organized information.** The least likely explanation and the most extraordinary claim. Extraordinary evidence would be required, including replication across frequencies, consistency between Planck and WMAP, and spatial localization via HEALPix analysis.

### 4.3 CMB Anomalies in Phi Space

The known CMB anomalies primarily affect the low-l multipoles (l < 30), which fall within the Sachs-Wolfe plateau region. We predict:

- The **low quadrupole** would reduce Phi in the Sachs-Wolfe region (less power means less coupling).
- The **quadrupole-octupole alignment** could produce anomalously high Phi at l = 2--3 if the alignment creates correlated structure between those scales.
- The **hemispherical asymmetry** is not detectable from the full-sky power spectrum alone (it requires hemisphere-separated analysis on HEALPix maps).
- The **Cold Spot** is a localized feature that would require spatially-resolved Phi analysis.

---

## 5. Interpretation Guide

### 5.1 Reading the Phi Spectrum

The sliding-window Phi spectrum Phi(l) plots integrated information as a function of central multipole. Key features to look for:

- **Peaks in Phi(l)** indicate angular scales where the CMB bands are most tightly integrated --- where information about one scale most strongly constrains other scales. We expect peaks near the acoustic peak positions (l ~ 220, 540, 800) because these scales are physically coupled through the photon-baryon fluid dynamics.

- **Troughs in Phi(l)** indicate scales where the CMB structure is decomposable --- the system behaves like independent parts. We expect troughs between acoustic peaks (at the "valleys" in the power spectrum) and in the damping tail where modes are exponentially suppressed.

- **Monotonic decline** in the damping tail (l > 1500) would indicate progressive loss of causal structure as Silk damping washes out the primordial correlations.

### 5.2 Physical Meaning of High and Low Phi

- **High Phi (Phi >> 0):** The system at those angular scales is irreducibly integrated. No partition of the multipole bands can fully decompose the system's dynamics. Physically, this means the angular scales are tightly coupled and cannot be analyzed independently. This is expected wherever the physical processes (acoustic oscillations, gravitational coupling) create correlations across scales.

- **Low Phi (Phi ~ 0):** The system is decomposable. The multipole bands within that range behave approximately independently. This is expected in the damping tail, where each mode is driven by independent stochastic perturbations and progressively erased by diffusion.

- **Phi = 0 exactly:** The system has a partition into independent subsystems. In IIT terms, the system "is not one thing" --- it is two or more causally independent components.

### 5.3 Limitations

This analysis has several important limitations that must be considered when interpreting results:

1. **Two-point statistics only.** The angular power spectrum captures only the two-point correlation function of the CMB temperature field. All information about non-Gaussianity, phase correlations, and higher-order statistics is discarded. A Gaussian random field with the same power spectrum would produce statistically identical TPMs.

2. **No spatial localization.** The power spectrum is a global (full-sky averaged) statistic. Localized features (Cold Spot, point sources, Sunyaev-Zeldovich clusters) are diluted into the spherical harmonic decomposition. Spatially-resolved analysis requires HEALPix maps (see Section 6).

3. **Bin count dependence.** The number of bins N directly affects the TPM dimension and therefore the Phi value. More bins increase the state space but also increase noise per bin. The results must be reported as a function of N and compared to null models at the same N.

4. **TPM construction is non-unique.** The mapping from power spectrum to TPM involves modeling choices (binning scheme, correlation kernel, decay length lambda). Different constructions would yield different Phi values. The null hypothesis testing controls for this by applying the same construction to null realizations, but the absolute Phi value is model-dependent.

5. **Temporal dynamics are absent.** IIT is designed for systems with temporal dynamics (state at time t determines state at time t+1). The CMB is a snapshot --- a single realization of a spatial random field. Our "temporal" axis is actually angular scale. This is a metaphorical, not literal, application of IIT.

6. **Computational constraints.** Exact Phi computation is feasible only for N <= 16 bins. For finer resolution, approximate algorithms introduce their own uncertainties.

---

## 6. Future Directions

### 6.1 Full-Sky HEALPix Analysis

The most important extension is to move from the angular power spectrum to full-sky pixel-space analysis using HEALPix maps. This would:

- Enable spatial localization of Phi anomalies (e.g., is the Cold Spot a Phi hotspot?)
- Permit hemisphere-separated analysis to test the hemispherical asymmetry
- Capture non-Gaussian structure that the power spectrum misses
- Allow comparison of different foreground-cleaning methods (Commander, NILC, SEVEM, SMICA)

The input data would be the Planck SMICA CMB temperature map at HEALPix resolution N_side = 2048 (approximately 50 million pixels). At this resolution, direct TPM construction is infeasible; hierarchical coarse-graining (using the HEALPix nested pixel scheme) or patch-based analysis would be required.

### 6.2 Multi-Frequency Analysis

The Planck satellite observed at nine frequencies (30, 44, 70, 100, 143, 217, 353, 545, 857 GHz). Constructing TPMs from individual frequency channels would allow:

- Disentangling CMB signal from frequency-dependent foregrounds
- Cross-frequency Phi analysis to test for frequency-dependent integrated structure
- Identification of foreground-driven Phi artifacts

### 6.3 Polarization Spectra

The CMB is linearly polarized at the approximately 5% level. The polarization power spectra (TE, EE, and the as-yet-undetected BB from primordial gravitational waves) contain independent information about the physics of recombination and inflation. Extending the analysis to:

- **TE cross-spectrum:** Correlations between temperature and E-mode polarization
- **EE auto-spectrum:** Pure polarization signal, less contaminated by foregrounds
- **BB auto-spectrum:** If detected, would probe tensor perturbations from inflation

Multi-spectrum TPM construction (using TT, TE, and EE jointly) would provide a richer causal structure with 3N-dimensional state space.

### 6.4 Foreground Comparison

Systematically comparing Phi between:

- Raw frequency maps (CMB + foregrounds)
- Component-separated CMB maps (SMICA, Commander, NILC, SEVEM)
- Foreground-only maps (synchrotron, dust, free-free)

This comparison would establish whether any observed Phi excess is attributable to foreground residuals rather than primordial signal.

### 6.5 Non-Gaussianity Detection via Higher-Order Phi

The standard f_NL parameterization of primordial non-Gaussianity measures the three-point function (bispectrum). IIT-based analysis could provide a complementary non-Gaussianity diagnostic:

- Compute Phi from the bispectrum B(l_1, l_2, l_3) in addition to the power spectrum
- Construct TPMs from the trispectrum (connected four-point function)
- Compare information content across n-point functions to quantify how much causal structure is captured at each order

### 6.6 Comparison with Other Cosmological Datasets

Extend the methodology to:

- **21 cm hydrogen line** (HERA, SKA): The 3D power spectrum of neutral hydrogen provides a volumetric (rather than projected) probe of cosmic structure.
- **Large-scale structure** (DESI, Euclid): Galaxy power spectra and correlation functions as alternative TPM inputs.
- **Gravitational wave background** (LIGO/Virgo/KAGRA, LISA, PTA): Strain power spectra as probes of integrated information in spacetime dynamics.

---

## 7. References

Albantakis, L., Barbosa, L., Findlay, G., Grasso, M., Haun, A.M., Marshall, W., Mayner, W.G.P., Zaeemzadeh, A., Boly, M., Juel, B.E., Sasai, S., Fujii, K., David, I., Hendren, J., Lang, J.P., and Tononi, G. (2023). "Integrated Information Theory (IIT) 4.0: Formulating the Properties of Phenomenal Existence in Physical Terms." *PLOS Computational Biology*, 19(10), e1011465. https://doi.org/10.1371/journal.pcbi.1011465

Bennett, C.L., Hill, R.S., Hinshaw, G., Larson, D., Smith, K.M., Dunkley, J., Gold, B., Halpern, M., Jarosik, N., Kogut, A., Komatsu, E., Limon, M., Meyer, S.S., Nolta, M.R., Odegard, N., Page, L., Spergel, D.N., Tucker, G.S., Weiland, J.L., Wollack, E., and Wright, E.L. (2011). "Seven-Year Wilkinson Microwave Anisotropy Probe (WMAP) Observations: Are There Cosmic Microwave Background Anomalies?" *The Astrophysical Journal Supplement Series*, 192(2), 17. https://doi.org/10.1088/0067-0049/192/2/17

Cruz, M., Martinez-Gonzalez, E., Vielva, P., and Cayon, L. (2005). "Detection of a non-Gaussian spot in WMAP." *Monthly Notices of the Royal Astronomical Society*, 356(1), 29--40. https://doi.org/10.1111/j.1365-2966.2004.08419.x

de Oliveira-Costa, A., Tegmark, M., Zaldarriaga, M., and Hamilton, A. (2004). "Significance of the largest scale CMB fluctuations in WMAP." *Physical Review D*, 69(6), 063516. https://doi.org/10.1103/PhysRevD.69.063516

Eriksen, H.K., Hansen, F.K., Banday, A.J., Gorski, K.M., and Lilje, P.B. (2004). "Asymmetries in the Cosmic Microwave Background Anisotropy Field." *The Astrophysical Journal*, 605(1), 14--20. https://doi.org/10.1086/382267

Fixsen, D.J. (2009). "The Temperature of the Cosmic Microwave Background." *The Astrophysical Journal*, 707(2), 916--920. https://doi.org/10.1088/0004-637X/707/2/916

Halko, N., Martinsson, P.-G., and Tropp, J. (2011). "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." *SIAM Review*, 53(2), 217--288. https://doi.org/10.1137/090771806

Hoel, E.P., Albantakis, L., and Tononi, G. (2013). "Quantifying causal emergence shows that macro can beat micro." *Proceedings of the National Academy of Sciences*, 110(49), 19790--19795. https://doi.org/10.1073/pnas.1314922110

Hoel, E.P. (2017). "When the Map is Better Than the Territory." *Entropy*, 19(5), 188. https://doi.org/10.3390/e19050188

Planck Collaboration (2020a). "Planck 2018 results. I. Overview and the cosmological legacy of Planck." *Astronomy & Astrophysics*, 641, A1. https://doi.org/10.1051/0004-6361/201833880

Planck Collaboration (2020b). "Planck 2018 results. V. CMB power spectra and likelihoods." *Astronomy & Astrophysics*, 641, A5. https://doi.org/10.1051/0004-6361/201936386

Planck Collaboration (2020c). "Planck 2018 results. VI. Cosmological parameters." *Astronomy & Astrophysics*, 641, A6. https://doi.org/10.1051/0004-6361/201833910

Tegmark, M., de Oliveira-Costa, A., and Hamilton, A. (2003). "A high resolution foreground cleaned CMB map from WMAP." *Physical Review D*, 68(12), 123523. https://doi.org/10.1103/PhysRevD.68.123523

Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5, 42. https://doi.org/10.1186/1471-2202-5-42

Tononi, G. (2008). "Consciousness as Integrated Information: a Provisional Manifesto." *Biological Bulletin*, 215(3), 216--242. https://doi.org/10.2307/25470707

Zhang, J., Liu, K., and Hoel, E.P. (2025). "Dynamical reversibility and causal emergence based on SVD." *npj Complexity*. https://doi.org/10.1038/s44260-025-00041-z

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|---|---|
| l | Multipole moment (angular wavenumber) |
| C_l | Angular power spectrum coefficient |
| D_l | l(l+1) C_l / (2 pi), in micro-K^2 |
| T_{ij} | Transition probability from state i to state j |
| Phi | Integrated information (IIT) |
| EI | Effective information (causal emergence) |
| CE | Causal emergence (EI_macro - EI_micro) |
| MIP | Minimum information partition |
| d_EMD | Earth Mover's Distance (Wasserstein-1) |
| D_KL | Kullback-Leibler divergence |
| H | Shannon entropy |
| sigma_k | k-th singular value |
| r_eff | Effective rank (exp of spectral entropy) |
| N | Number of multipole bins |
| W | Window width (sliding Phi analysis) |
| lambda | Correlation length in log-l space |
| N_MC | Number of Monte Carlo null realizations |

## Appendix B: Reproducibility

All results can be reproduced by running:

```bash
cd examples/cmb-consciousness
cargo run --release
```

The analysis uses deterministic seeding (via `rand_chacha`) for all stochastic components, ensuring bitwise reproducibility across runs. The default random seed is 42.

Dependencies are pinned in `Cargo.toml` and use the `ruvector-consciousness` crate with features `phi`, `emergence`, and `collapse`.
