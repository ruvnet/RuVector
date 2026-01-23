/**
 * TypeScript definitions for prime-radiant-advanced-wasm
 *
 * WASM bindings for Prime-Radiant Advanced Math modules:
 * - CohomologyEngine: Sheaf cohomology computations
 * - CategoryEngine: Functorial retrieval and topos operations
 * - HoTTEngine: Type checking and path operations
 * - SpectralEngine: Eigenvalue computation and Cheeger bounds
 * - CausalEngine: Causal inference and interventions
 * - QuantumEngine: Topological invariants and quantum simulation
 */

// ============================================================================
// Common Types
// ============================================================================

export interface WasmError {
  readonly message: string;
  readonly code: string;
}

// ============================================================================
// Cohomology Types
// ============================================================================

export interface SheafNode {
  id: number;
  label: string;
  section: number[];
  weight: number;
}

export interface SheafEdge {
  source: number;
  target: number;
  restriction_map: number[];
  source_dim: number;
  target_dim: number;
}

export interface SheafGraph {
  nodes: SheafNode[];
  edges: SheafEdge[];
}

export interface CohomologyResult {
  h0_dim: number;
  h1_dim: number;
  euler_characteristic: number;
  consistency_energy: number;
  is_consistent: boolean;
}

export interface Obstruction {
  edge_index: number;
  source_node: number;
  target_node: number;
  obstruction_vector: number[];
  magnitude: number;
  description: string;
}

/**
 * Sheaf cohomology computation engine
 */
export class CohomologyEngine {
  /**
   * Create a new cohomology engine with default tolerance (1e-10)
   */
  constructor();

  /**
   * Create with custom tolerance
   */
  static withTolerance(tolerance: number): CohomologyEngine;

  /**
   * Compute cohomology groups of a sheaf graph
   * Returns H^0, H^1 dimensions, Euler characteristic, and consistency energy
   */
  computeCohomology(graph: SheafGraph): CohomologyResult;

  /**
   * Detect all obstructions to global consistency
   * Returns sorted list of obstructions by magnitude (largest first)
   */
  detectObstructions(graph: SheafGraph): Obstruction[];

  /**
   * Compute global sections (H^0)
   */
  computeGlobalSections(graph: SheafGraph): number[][];

  /**
   * Compute consistency energy (sum of squared restriction errors)
   */
  consistencyEnergy(graph: SheafGraph): number;
}

// ============================================================================
// Spectral Types
// ============================================================================

export interface Graph {
  n: number;
  edges: [number, number, number][];  // [source, target, weight]
}

export interface CheegerBounds {
  lower_bound: number;
  upper_bound: number;
  cheeger_estimate: number;
  fiedler_value: number;
}

export interface SpectralGap {
  lambda_1: number;
  lambda_2: number;
  gap: number;
  ratio: number;
}

export interface MinCutPrediction {
  predicted_cut: number;
  lower_bound: number;
  upper_bound: number;
  confidence: number;
  cut_nodes: number[];
}

/**
 * Spectral analysis engine for graph theory computations
 */
export class SpectralEngine {
  /**
   * Create a new spectral engine with default configuration
   */
  constructor();

  /**
   * Create with custom configuration
   */
  static withConfig(
    num_eigenvalues: number,
    tolerance: number,
    max_iterations: number
  ): SpectralEngine;

  /**
   * Compute Cheeger bounds using spectral methods
   * Returns lower/upper bounds on isoperimetric number
   */
  computeCheegerBounds(graph: Graph): CheegerBounds;

  /**
   * Compute eigenvalues of the graph Laplacian
   */
  computeEigenvalues(graph: Graph): number[];

  /**
   * Compute algebraic connectivity (Fiedler value = second smallest eigenvalue)
   */
  algebraicConnectivity(graph: Graph): number;

  /**
   * Compute spectral gap information
   */
  computeSpectralGap(graph: Graph): SpectralGap;

  /**
   * Predict minimum cut using spectral methods
   */
  predictMinCut(graph: Graph): MinCutPrediction;

  /**
   * Compute Fiedler vector (eigenvector for second smallest eigenvalue)
   */
  computeFiedlerVector(graph: Graph): number[];
}

// ============================================================================
// Causal Types
// ============================================================================

export interface CausalVariable {
  name: string;
  var_type: 'continuous' | 'discrete' | 'binary';
}

export interface CausalEdge {
  from: string;
  to: string;
}

export interface CausalModel {
  variables: CausalVariable[];
  edges: CausalEdge[];
}

export interface InterventionResult {
  variable: string;
  original_value: number;
  intervened_value: number;
  affected_variables: string[];
  causal_effect: number;
}

export interface DSeparationResult {
  x: string;
  y: string;
  conditioning: string[];
  d_separated: boolean;
}

/**
 * Causal inference engine based on structural causal models
 */
export class CausalEngine {
  /**
   * Create a new causal engine
   */
  constructor();

  /**
   * Check d-separation between two variables given a conditioning set
   */
  checkDSeparation(
    model: CausalModel,
    x: string,
    y: string,
    conditioning: string[]
  ): DSeparationResult;

  /**
   * Compute causal effect via do-operator
   */
  computeCausalEffect(
    model: CausalModel,
    treatment: string,
    outcome: string,
    treatment_value: number
  ): InterventionResult;

  /**
   * Get topological order of variables
   */
  topologicalOrder(model: CausalModel): string[];

  /**
   * Find all confounders between treatment and outcome
   */
  findConfounders(
    model: CausalModel,
    treatment: string,
    outcome: string
  ): string[];

  /**
   * Check if model is a valid DAG (no cycles)
   */
  isValidDag(model: CausalModel): boolean;
}

// ============================================================================
// Quantum Types
// ============================================================================

export interface Complex {
  re: number;
  im: number;
}

export interface QuantumState {
  amplitudes: Complex[];
  dimension: number;
}

export interface TopologicalInvariant {
  betti_numbers: number[];
  euler_characteristic: number;
  is_connected: boolean;
}

export interface FidelityResult {
  fidelity: number;
  trace_distance: number;
}

/**
 * Quantum computing and topological analysis engine
 */
export class QuantumEngine {
  /**
   * Create a new quantum engine
   */
  constructor();

  /**
   * Compute topological invariants of a simplicial complex
   * @param simplices Array of simplices, each simplex is an array of vertex indices
   */
  computeTopologicalInvariants(simplices: number[][]): TopologicalInvariant;

  /**
   * Compute quantum state fidelity |<psi|phi>|^2
   */
  computeFidelity(state1: QuantumState, state2: QuantumState): FidelityResult;

  /**
   * Create a GHZ state (|0...0> + |1...1>)/sqrt(2)
   */
  createGHZState(num_qubits: number): QuantumState;

  /**
   * Create a W state (|10...0> + |01...0> + ... + |0...01>)/sqrt(n)
   */
  createWState(num_qubits: number): QuantumState;

  /**
   * Compute entanglement entropy of a subsystem
   */
  computeEntanglementEntropy(state: QuantumState, subsystem_size: number): number;

  /**
   * Apply a single-qubit gate to a quantum state
   * @param gate 2x2 complex matrix
   * @param target_qubit Index of target qubit (0-indexed)
   */
  applyGate(state: QuantumState, gate: Complex[][], target_qubit: number): QuantumState;
}

// ============================================================================
// Category Types
// ============================================================================

export interface CatObject {
  id: string;
  dimension: number;
  data: number[];
}

export interface Morphism {
  source: string;
  target: string;
  matrix: number[];
  source_dim: number;
  target_dim: number;
}

export interface Category {
  name: string;
  objects: CatObject[];
  morphisms: Morphism[];
}

export interface Functor {
  name: string;
  source_category: string;
  target_category: string;
  object_map: Record<string, string>;
}

export interface RetrievalResult {
  object_id: string;
  similarity: number;
}

/**
 * Category theory engine for functorial operations
 */
export class CategoryEngine {
  /**
   * Create a new category engine
   */
  constructor();

  /**
   * Compose two morphisms: g . f
   */
  composeMorphisms(f: Morphism, g: Morphism): Morphism;

  /**
   * Verify categorical laws (identity and associativity)
   */
  verifyCategoryLaws(category: Category): boolean;

  /**
   * Functorial retrieval: find k most similar objects to query
   */
  functorialRetrieve(category: Category, query: number[], k: number): RetrievalResult[];

  /**
   * Apply morphism to data
   */
  applyMorphism(morphism: Morphism, data: number[]): number[];

  /**
   * Verify that a functor preserves composition
   */
  verifyFunctoriality(functor: Functor, source_category: Category): boolean;
}

// ============================================================================
// HoTT Types
// ============================================================================

export interface HoTTType {
  name: string;
  level: number;
  kind: 'unit' | 'bool' | 'nat' | 'product' | 'sum' | 'function' | 'identity' | 'var';
  params: string[];
}

export interface HoTTTerm {
  kind: 'var' | 'star' | 'true' | 'false' | 'zero' | 'succ' | 'lambda' | 'app' | 'pair' | 'refl' | 'compose' | 'inverse';
  value?: string;
  children: HoTTTerm[];
}

export interface HoTTPath {
  base_type: HoTTType;
  start: HoTTTerm;
  end: HoTTTerm;
  proof: HoTTTerm;
}

export interface TypeCheckResult {
  is_valid: boolean;
  inferred_type?: HoTTType;
  error?: string;
}

export interface PathOperationResult {
  is_valid: boolean;
  result_path?: HoTTPath;
  error?: string;
}

/**
 * Homotopy Type Theory engine for type checking and path operations
 */
export class HoTTEngine {
  /**
   * Create a new HoTT engine
   */
  constructor();

  /**
   * Create with strict mode enabled
   */
  static withStrictMode(strict: boolean): HoTTEngine;

  /**
   * Type check a term against an expected type
   */
  typeCheck(term: HoTTTerm, expected_type: HoTTType): TypeCheckResult;

  /**
   * Infer the type of a term
   */
  inferType(term: HoTTTerm): TypeCheckResult;

  /**
   * Compose two paths: p . q
   */
  composePaths(path1: HoTTPath, path2: HoTTPath): PathOperationResult;

  /**
   * Invert a path: p^-1
   */
  invertPath(path: HoTTPath): PathOperationResult;

  /**
   * Create a reflexivity path: refl_a : a = a
   */
  createReflPath(type: HoTTType, point: HoTTTerm): HoTTPath;

  /**
   * Check if two types are equivalent (related to univalence)
   */
  checkTypeEquivalence(type1: HoTTType, type2: HoTTType): boolean;
}

// ============================================================================
// Module Functions
// ============================================================================

/**
 * Get library version
 */
export function getVersion(): string;

/**
 * Initialize the WASM module (call once before using engines)
 */
export function initModule(): void;

/**
 * Default export: initialize function
 */
export default function init(): Promise<void>;
