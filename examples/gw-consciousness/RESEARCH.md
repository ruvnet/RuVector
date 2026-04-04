# Gravitational Wave Background Consciousness Analysis

## What is the Gravitational Wave Background?

The gravitational wave background (GWB) is a persistent, low-frequency signal
permeating spacetime, analogous to the cosmic microwave background but in
gravitational waves rather than light. Pulsar timing arrays (PTAs) detect this
background by measuring correlated deviations in the arrival times of radio
pulses from millisecond pulsars.

The GWB manifests as a characteristic strain spectrum h_c(f) across nanohertz
frequencies (1-100 nHz), with the spectral shape encoding the nature of the
source population.

## NANOGrav 15-Year Results (2023)

In June 2023, the NANOGrav collaboration published compelling evidence for a
gravitational wave background using 15 years of pulsar timing data
(Agazie et al. 2023, ApJL 951 L8). Key findings:

- **Detection significance**: 3.5-4 sigma evidence for a common-spectrum process
  with Hellings-Downs spatial correlations (the smoking gun of a GW origin).
- **Spectrum**: 14 frequency bins at f_k = k/T, where T = 16.03 years.
- **Best-fit amplitude**: A = 2.4 x 10^{-15} at f_ref = 1/yr.
- **Spectral index**: alpha = -2/3 (consistent with SMBH binary mergers), but
  other spectral indices are not excluded.
- **Corroborated** by EPTA, PPTA, and CPTA with similar results.

The leading interpretation is that the GWB arises from the incoherent
superposition of gravitational waves from ~10^4-10^5 supermassive black hole
(SMBH) binary systems across the universe. However, exotic cosmological sources
remain viable alternatives.

## Why IIT is a Novel Analysis Tool for GW

Integrated Information Theory (IIT) provides a principled measure of how much a
system's parts are informationally integrated beyond the sum of their
independent contributions. When applied to the GWB:

- **Frequency bins as system elements**: Each bin in the strain spectrum
  represents a "node" in the information-theoretic system.
- **Transition probability matrix**: The spectral correlations between frequency
  bins define a TPM encoding causal relationships.
- **Phi as a discriminator**: Different GW source models predict different
  correlation structures, yielding different Phi values.

This is fundamentally different from standard Bayesian model comparison because
IIT measures the *intrinsic causal structure* of the signal rather than its
statistical fit to a parametric model.

## Methodology

### Spectrum to TPM Construction

1. **Generate strain spectrum** h_c(f) for each source model at 14 NANOGrav
   frequency bins.
2. **Compute pairwise correlations** C_{ij} using a Gaussian kernel in
   log-frequency space, weighted by strain power.
3. **Apply model-specific correlation widths**:
   - SMBH mergers: narrow kernel (sigma=0.3) -- each binary is independent,
     so frequency bins are weakly coupled.
   - Cosmic strings: broad kernel (sigma=2.0) -- the string network produces
     coherent emission across decades of frequency.
   - Primordial GW: moderate kernel (sigma=1.5) -- inflationary correlations.
   - Phase transition: very broad kernel (sigma=3.0) -- the bubble collision
     spectrum is highly correlated.
4. **Row-normalize** to obtain a valid transition probability matrix.

### Analysis Pipeline

1. **Compute IIT Phi** for each source model's TPM using `auto_compute_phi`.
2. **Compute causal emergence** (effective information, determinism, degeneracy)
   for each model.
3. **Compute SVD emergence** (effective rank, spectral entropy, emergence index)
   for each model.
4. **Null hypothesis testing**: Generate 100 realizations of the SMBH model
   with measurement noise, compute Phi for each, and compare the exotic
   model Phi to this null distribution.

### Key Question

Does the GW background show more integrated information than expected from
independent SMBH mergers?

- If Phi(exotic) >> Phi(SMBH) with p < 0.05, this suggests the GWB has a
  correlated cosmological origin.
- If Phi(exotic) ~ Phi(SMBH), the data are consistent with independent mergers.

## Expected Results

### SMBH Mergers (Phi ~ 0)

Each SMBH binary contributes independently to the GWB. The strain at different
frequencies is determined by different populations of binaries at different
orbital separations. This produces a nearly diagonal TPM with low off-diagonal
correlations, yielding Phi close to zero.

### Cosmic Strings (Phi > 0)

A cosmic string network produces gravitational waves through loop oscillations
and cusps. The emission spectrum of a single loop spans many decades of
frequency, creating strong correlations between bins. The resulting TPM has
significant off-diagonal structure, yielding higher Phi.

### Primordial GW (Phi > 0)

Inflationary gravitational waves have a nearly scale-invariant spectrum. The
coherent production mechanism during inflation correlates all frequency bins,
producing moderate Phi.

### Phase Transition (Phi >> 0)

A first-order cosmological phase transition (e.g., electroweak or QCD) produces
a peaked GW spectrum with strong correlations around the peak frequency. The
bubble nucleation and collision process is highly coherent, potentially
producing the highest Phi among all models.

## What a Positive Result Would Mean

A finding that the GWB exhibits higher integrated information than expected from
SMBH mergers would:

1. **Provide model-independent evidence** for a correlated cosmological source,
   complementing Bayesian spectral analysis.
2. **Distinguish source classes** without assuming a specific spectral model --
   IIT measures the causal structure directly.
3. **Motivate targeted searches** for specific exotic sources based on the
   observed correlation pattern.
4. **Demonstrate a new application** of consciousness metrics to astrophysical
   data analysis, opening a novel avenue for gravitational wave science.

Note that a positive IIT result does not imply the GWB is "conscious" in any
meaningful sense. Rather, it indicates that the spectral correlations contain
more integrated causal structure than expected from independent sources, which
is a purely information-theoretic statement about the signal's origin.

## References

- Agazie et al. (2023). "The NANOGrav 15 yr Data Set: Evidence for a
  Gravitational-wave Background." ApJL 951, L8.
- Tononi, G. (2008). "Consciousness as Integrated Information." Biological
  Bulletin, 215(3), 216-242.
- Hoel, E. P. et al. (2013). "Quantifying causal emergence shows that macro
  can beat micro." PNAS, 110(49), 19790-19795.
