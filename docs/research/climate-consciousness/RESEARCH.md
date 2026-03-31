# Climate Teleconnection Consciousness Analysis

## Overview

This example applies Integrated Information Theory (IIT) Phi to climate mode
interactions to study how large-scale climate oscillations form integrated
information systems. The key question is whether the climate system's
teleconnections create genuine integrated information, and whether El Nino
events increase this integration.

## Background

### Climate Teleconnections

Teleconnections are large-scale patterns of climate variability that link
weather and climate conditions across distant regions. The major climate
indices capture these patterns:

- **ENSO (El Nino Southern Oscillation)**: The dominant mode of interannual
  climate variability, involving coupled ocean-atmosphere dynamics in the
  tropical Pacific. Measured by the Nino3.4 index.

- **NAO (North Atlantic Oscillation)**: Pressure difference between the
  Icelandic Low and Azores High, controlling weather patterns across the
  North Atlantic and Europe.

- **PDO (Pacific Decadal Oscillation)**: Long-lived pattern of Pacific
  climate variability, related to but distinct from ENSO on decadal scales.

- **AMO (Atlantic Multidecadal Oscillation)**: Basin-wide sea surface
  temperature variations in the North Atlantic on 60-80 year timescales.

- **IOD (Indian Ocean Dipole)**: East-west temperature gradient in the
  tropical Indian Ocean, strongly coupled to ENSO.

- **SAM (Southern Annular Mode)**: The dominant mode of extratropical
  variability in the Southern Hemisphere, mostly independent.

- **QBO (Quasi-Biennial Oscillation)**: Alternating easterly/westerly winds
  in the equatorial stratosphere with a ~28-month period.

### Known Teleconnection Strengths

The coupling between these modes varies:
- **Strong**: ENSO-IOD (r ~ 0.5-0.7), driven by Walker circulation coupling
- **Moderate**: ENSO-PDO (r ~ 0.3-0.5), Pacific basin shared dynamics
- **Moderate**: NAO-AMO (r ~ 0.2-0.4), Atlantic ocean-atmosphere coupling
- **Weak**: QBO-ENSO (r ~ 0.1-0.2), stratosphere-troposphere interaction
- **Minimal**: SAM is largely independent of tropical modes

### IIT and Climate Systems

Applying IIT to climate modes asks: does the climate system generate more
integrated information than the sum of its regional parts? Specifically:

- **Do teleconnections create integration?** If climate modes are truly
  coupled (not just correlated), the system should have non-trivial Phi.

- **Does El Nino increase integration?** During El Nino events, ENSO
  correlations strengthen by ~50%, potentially increasing system-wide Phi.

- **What are the irreducible subsystems?** The Pacific basin (ENSO, PDO, IOD)
  is expected to be the most integrated regional subsystem.

## Method

### TPM Construction

1. Start with a 7x7 correlation matrix based on known teleconnection strengths.
2. Apply a sharpness exponent (alpha=2.0) to emphasize strong connections.
3. Row-normalize to get transition probabilities.
4. Generate El Nino variant by boosting ENSO correlations 50%.

### Analysis Pipeline

1. Compute Phi for the full 7-index system (neutral and El Nino).
2. Compute Phi for regional subsets: Pacific, Atlantic, Polar.
3. Compare neutral vs El Nino Phi.
4. Temporal analysis: simulate 12 monthly states with seasonal modulation.
5. Causal emergence: find optimal coarse-graining.
6. Null hypothesis testing with shuffled correlations.

### Seasonal Modulation

Climate modes have distinct seasonal cycles:
- **ENSO**: peaks in boreal winter (DJF), weakens in spring (MAM)
- **NAO**: strongest in winter, weakest in summer
- This creates a seasonal Phi cycle that should peak in winter when
  teleconnections are strongest.

## Expected Results

- The Pacific basin should have the highest regional Phi due to strong
  ENSO-IOD-PDO coupling.
- El Nino should increase full-system Phi by strengthening teleconnections.
- Monthly Phi should peak in DJF (boreal winter) when both ENSO and NAO
  teleconnections are strongest.
- The null model (shuffled correlations) should have different Phi,
  confirming that the specific teleconnection structure matters.
- SAM should contribute least to integration (mostly independent).

## Implications

If climate modes show genuine integrated information, it suggests that:
1. The climate system has emergent properties beyond its components.
2. Teleconnections are not merely correlations but genuine causal coupling.
3. El Nino events fundamentally change the system's information architecture.
4. The Pacific basin acts as the "core" of global climate integration.

## References

1. Tononi G. (2004). An information integration theory of consciousness. BMC Neurosci.
2. Hoel E.P. et al. (2013). Quantifying causal emergence. PNAS.
3. Trenberth K.E. et al. (1998). Progress during TOGA in understanding and modeling
   global teleconnections. J. Geophys. Res.
4. McPhaden M.J. et al. (2006). ENSO as an integrating concept in earth science. Science.
5. Saji N.H. et al. (1999). A dipole mode in the tropical Indian Ocean. Nature.
