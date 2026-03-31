# Quantum Circuit Consciousness: IIT Phi and Entanglement

## Motivation

Integrated Information Theory (IIT) and quantum entanglement both
formalize the idea that a system is "more than the sum of its parts."
This example explores whether IIT's Phi measure, applied to quantum
circuit measurement statistics, captures the same structure as
standard entanglement measures.

## Quantum States Analyzed

### 1. Bell State (2 qubits)

The maximally entangled two-qubit state:

    |Psi> = (|00> + |11>) / sqrt(2)

Prepared by H(0) followed by CNOT(0,1). Measurement in the
computational basis yields 00 or 11 with equal probability, never
01 or 10. This maximal correlation should produce HIGH Phi.

### 2. GHZ State (3 qubits)

The Greenberger-Horne-Zeilinger state:

    |GHZ> = (|000> + |111>) / sqrt(2)

Genuinely multipartite entangled: tracing out any single qubit
destroys all entanglement. Expected to show HIGH Phi and high
emergence (the 3-party correlations cannot be reduced to 2-party).

### 3. Product State (3 qubits)

    |Psi> = |0> x |0> x |0>

Completely separable, no entanglement. The TPM is the identity
matrix (each input maps to itself). Expected Phi = 0 since the
system decomposes perfectly into independent parts.

### 4. W State (3 qubits)

    |W> = (|001> + |010> + |100>) / sqrt(3)

Bipartite entanglement that survives partial trace: tracing out
any one qubit still leaves the other two entangled. Different
entanglement structure from GHZ. Expected: Phi between product
and GHZ.

### 5. Random Circuit (3 qubits, depth 5)

Random single-qubit rotations interleaved with CNOT gates. The
resulting entanglement depends on the specific random gates chosen.
Serves as a control to show that Phi varies continuously with
circuit structure.

## TPM Construction for Quantum Circuits

The key mapping from quantum mechanics to IIT:

    TPM[i][j] = |<j|U|i>|^2

where U is the circuit unitary, and i, j are computational basis
states. This gives the probability of measuring outcome j when the
input is the basis state i. The resulting TPM is doubly stochastic
for unitary circuits (both rows and columns sum to 1).

## Expected Phi Hierarchy

Standard entanglement measures predict:

    Product (0) < W < Bell <= GHZ

IIT's Phi may not follow this ordering exactly because:

1. Phi measures integrated information across the minimum information
   partition (MIP), not entanglement per se
2. The Bell state is 2-qubit while GHZ/W are 3-qubit, so the
   partition spaces differ
3. W state has different entanglement structure (robust to qubit loss)
   which may be valued differently by Phi

## Entanglement Measures for Comparison

- **Concurrence** (2 qubits): measures entanglement of formation
- **Tangle** (3 qubits): measures genuine 3-party entanglement
- **Entanglement entropy**: von Neumann entropy of reduced density
  matrix

The GHZ state has maximal tangle but zero concurrence for any pair.
The W state has zero tangle but nonzero concurrence for every pair.

## Causal Emergence in Quantum Systems

Causal emergence asks: is there a macro-level description of the
quantum system that is more informative than the qubit-level
description? For entangled states, the answer may be yes -- the
entangled subsystem behaves as a single effective degree of freedom.

## Limitations

1. **Classical TPM**: we use |<j|U|i>|^2, discarding quantum phases.
   IIT on the full quantum state (quantum IIT) is an active research
   area.
2. **Measurement basis dependence**: Phi depends on the choice of
   computational basis. A different measurement basis could yield
   different Phi values.
3. **Small systems**: 3 qubits = 8x8 TPM, well within exact Phi
   computation limits but far from interesting quantum advantage
   regimes.

## References

- Tononi, G. (2008). Consciousness as Integrated Information.
- Zanardi, P. et al. (2018). Quantum Integrated Information Theory.
- Greenberger, D.M. et al. (1989). Going Beyond Bell's Theorem.
- Dur, W. et al. (2000). Three qubits can be entangled in two
  inequivalent ways.
- Hoel, E.P. (2017). When the Map Is Better Than the Territory.
