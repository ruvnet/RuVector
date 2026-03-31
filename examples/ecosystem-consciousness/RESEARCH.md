# Ecosystem Consciousness: IIT Phi as a Food Web Integration Metric

## Motivation

Integrated Information Theory (IIT) quantifies how much a system is
"more than the sum of its parts" through the measure Phi. Food webs
share a structural analogy: a resilient ecosystem cannot be decomposed
into independent sub-networks without losing emergent function. This
example explores whether IIT Phi correlates with ecological resilience.

## Food Web Ecology Background

### Trophic Structure

Ecosystems organize into trophic levels:

1. **Producers** -- autotrophs (plants, algae, coral) that fix energy
2. **Primary consumers** -- herbivores feeding on producers
3. **Secondary consumers** -- predators feeding on herbivores
4. **Decomposers** -- organisms recycling dead matter back to producers
5. **Apex predators** -- top-level predators with no natural enemies

### Resilience and Redundancy

Ecological resilience depends on:

- **Functional redundancy**: multiple species filling similar roles
- **Response diversity**: different species respond differently to
  perturbation
- **Connectivity**: dense interaction networks buffer against single
  species loss
- **Keystone species**: removal causes disproportionate collapse

## IIT as a Resilience Metric

### TPM Construction

We model each species as a "state" and construct the transition
probability matrix from energy flow weights:

    TPM[i][j] = P(energy flows from species j to species i)

Row-normalization ensures each row sums to 1, giving a proper
stochastic matrix.

### Phi and Ecosystem Integration

- **High Phi**: the food web cannot be split into independent
  sub-networks -- every partition loses significant information
  about the whole
- **Low Phi**: the ecosystem decomposes into weakly connected
  modules -- removing one module barely affects the rest

### Species Contribution

We define the "Phi contribution" of species k as:

    C(k) = Phi(full) - Phi(without k)

Species with high C(k) are "consciousness keystones" -- their
removal most reduces the integrated information of the web.

## Expected Results

### Tropical Rainforest (12 species)

- Dense cross-trophic connections and nutrient cycling
- Many redundant pathways between trophic levels
- **Prediction**: HIGH Phi, relatively uniform contributions

### Agricultural Monoculture (8 species)

- Sparse, linear food chains
- Single crop dominates energy flow
- **Prediction**: LOW Phi, highly concentrated contributions

### Coral Reef (10 species)

- Moderate connectivity centered on coral as structural keystone
- Removing coral should cause largest Phi drop
- **Prediction**: MODERATE Phi, coral has disproportionate contribution

## Causal Emergence in Ecosystems

Beyond Phi, we compute causal emergence to ask: does the ecosystem
have a "macro-level" description (e.g., trophic levels) that is more
informative than the species-level description?

- High causal emergence suggests natural macro-level organization
  (trophic levels are real causal entities, not just labels)
- Low causal emergence suggests species-level dynamics dominate

## Limitations

1. **Synthetic data**: real food webs have stochastic, seasonal dynamics
2. **Static TPM**: IIT assumes a fixed transition structure
3. **Small system sizes**: Phi is computationally expensive (exponential
   in system size), limiting analysis to ~15 species
4. **Directionality**: IIT Phi is defined for mechanisms, not flows --
   the food web analogy is suggestive, not rigorous

## References

- Tononi, G. (2008). Consciousness as Integrated Information: a
  Provisional Manifesto. Biological Bulletin, 215(3).
- May, R.M. (1973). Stability and Complexity in Model Ecosystems.
- Dunne, J.A. et al. (2002). Food-web structure and network theory.
- Hoel, E.P. et al. (2013). Quantifying causal emergence shows that
  macro can beat micro.
