/**
 * RuVector WASM Unified - quantum backend
 *
 * Quantum gate simulation is delegated to the optional `@ruvector/ruqu-wasm`
 * package (^2.1.0), a state-vector simulator. It is loaded lazily by
 * `initQuantumBackend()` (also called best-effort by `UnifiedEngine.init()`).
 * Without it, every quantum operation throws a `QuantumBackendError` instead
 * of returning placeholder results.
 *
 * A `QuantumState` is a deferred circuit: each gate appends to `gates` and the
 * circuit is re-simulated. ruqu exposes outcome probabilities, not complex
 * amplitudes, so `amplitudes[i]` holds the magnitude `sqrt(P(i))` and `phases`
 * is zero-filled. Basis index `i` is little-endian: qubit 0 is the least
 * significant bit.
 */

import type { QuantumState } from './types';

/** Gate types accepted by the quantum backend. */
export type QuantumGateType = 'H' | 'X' | 'Y' | 'Z' | 'CNOT' | 'RX' | 'RY' | 'RZ' | 'CZ' | 'SWAP';

/** A single gate in a deferred circuit. */
export interface QuantumGateOp {
  type: QuantumGateType;
  targets: number[];
  parameter?: number;
}

/** Subset of ruqu-wasm's `WasmQuantumCircuit` used here. */
interface RuquCircuit {
  h(q: number): void;
  x(q: number): void;
  y(q: number): void;
  z(q: number): void;
  rx(q: number, angle: number): void;
  ry(q: number, angle: number): void;
  rz(q: number, angle: number): void;
  cnot(control: number, target: number): void;
  cz(a: number, b: number): void;
  swap(a: number, b: number): void;
  measure(q: number): void;
  free(): void;
}

interface RuquSimulation {
  probabilities: number[];
  measurements: Array<{ qubit: number; result: boolean; probability: number }>;
}

/** Subset of the `@ruvector/ruqu-wasm` module used here. */
export interface RuquModule {
  WasmQuantumCircuit: new (numQubits: number) => RuquCircuit;
  simulate(circuit: RuquCircuit): RuquSimulation;
  max_qubits(): number;
  initSync?: (module: { module: BufferSource | WebAssembly.Module }) => unknown;
  default?: (init?: { module_or_path: unknown }) => Promise<unknown>;
}

/** Where the quantum backend stands. */
export type QuantumBackendStatus = 'uninitialized' | 'ready' | 'missing' | 'failed';

/** Thrown by quantum operations when the backend cannot serve them. */
export class QuantumBackendError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'QuantumBackendError';
  }
}

const RUQU_PACKAGE = '@ruvector/ruqu-wasm';
const INSTALL_HINT =
  `Quantum operations in @ruvector/wasm-unified need the optional ${RUQU_PACKAGE} ` +
  `package (^2.1.0). Install it with \`npm install ${RUQU_PACKAGE}\`, then ` +
  '`await initQuantumBackend()` (or `await engine.init()`).';

let backend: RuquModule | null = null;
let status: QuantumBackendStatus = 'uninitialized';
let failure = '';
let pending: Promise<void> | null = null;

/** Options for {@link initQuantumBackend}. */
export interface QuantumBackendOptions {
  /** An already initialised ruqu-wasm module (skips import and wasm load). */
  module?: RuquModule;
  /** Wasm bytes, module or URL to initialise ruqu-wasm with (browsers). */
  wasm?: BufferSource | WebAssembly.Module | URL | string;
}

function isModuleNotFound(err: unknown): boolean {
  const code = (err as { code?: string } | null)?.code;
  if (code === 'ERR_MODULE_NOT_FOUND' || code === 'MODULE_NOT_FOUND') return true;
  return err instanceof Error && /Cannot find (module|package)/.test(err.message);
}

type RuquImporter = () => Promise<unknown>;

const importRuqu: RuquImporter = () => {
  const specifier: string = RUQU_PACKAGE;
  return import(specifier);
};
let ruquImporter: RuquImporter = importRuqu;

/** Test hook: replace the module importer (undefined restores the default). */
export function setRuquImporterForTesting(importer: RuquImporter | undefined): void {
  ruquImporter = importer ?? importRuqu;
}

async function loadRuqu(options: QuantumBackendOptions): Promise<RuquModule> {
  if (options.module) return options.module;
  const mod = (await ruquImporter()) as RuquModule;
  const isNode = typeof process !== 'undefined' && !!process.versions?.node;
  if (options.wasm !== undefined) {
    if (!mod.default) throw new Error(`${RUQU_PACKAGE} has no async initialiser`);
    await mod.default({ module_or_path: options.wasm });
  } else if (isNode) {
    // wasm-bindgen's default loader fetches a file: URL, which Node's fetch
    // rejects, so hand it the bytes directly.
    const { readFileSync } = await import('node:fs');
    const { createRequire } = await import('node:module');
    const wasmPath = createRequire(import.meta.url).resolve(`${RUQU_PACKAGE}/ruqu_wasm_bg.wasm`);
    if (!mod.initSync) throw new Error(`${RUQU_PACKAGE} has no initSync`);
    mod.initSync({ module: readFileSync(wasmPath) });
  } else if (mod.default) {
    await mod.default();
  }
  return mod;
}

/**
 * Load and initialise `@ruvector/ruqu-wasm`. Safe to call more than once.
 * @throws QuantumBackendError when the package is missing or fails to load
 */
export async function initQuantumBackend(options: QuantumBackendOptions = {}): Promise<void> {
  if (backend && !options.module) return;
  if (!pending || options.module) {
    pending = loadRuqu(options).then(
      (mod) => {
        backend = mod;
        status = 'ready';
        failure = '';
      },
      (err: unknown) => {
        pending = null;
        if (isModuleNotFound(err)) {
          status = 'missing';
        } else {
          status = 'failed';
          failure = err instanceof Error ? err.message : String(err);
          if ((err as { code?: string }).code === 'ERR_REQUIRE_ESM') {
            failure += ' (@ruvector/ruqu-wasm is ESM-only: use import, or Node >= 20.19 for require)';
          }
        }
        throw new QuantumBackendError(quantumBackendMessage());
      }
    );
  }
  return pending;
}

/** Current backend status. */
export function quantumBackendStatus(): QuantumBackendStatus {
  return status;
}

/** Test hook: drop the loaded backend and return to 'uninitialized'. */
export function resetQuantumBackend(): void {
  backend = null;
  status = 'uninitialized';
  failure = '';
  pending = null;
}

function quantumBackendMessage(): string {
  switch (status) {
    case 'missing':
      return `${RUQU_PACKAGE} is not installed. ${INSTALL_HINT}`;
    case 'failed':
      return `${RUQU_PACKAGE} failed to load: ${failure}. ${INSTALL_HINT}`;
    default:
      return 'Quantum backend not initialised: call `await initQuantumBackend()` ' +
        `(or \`await engine.init()\`) before quantum operations. ${INSTALL_HINT}`;
  }
}

function requireBackend(): RuquModule {
  if (!backend) throw new QuantumBackendError(quantumBackendMessage());
  return backend;
}

function checkQubit(q: number, numQubits: number): void {
  if (!Number.isInteger(q) || q < 0 || q >= numQubits) {
    throw new RangeError(`qubit index ${q} out of range for a ${numQubits}-qubit state`);
  }
}

function applyGate(c: RuquCircuit, g: QuantumGateOp): void {
  const [a = 0, b = 0] = g.targets;
  const p = g.parameter ?? 0;
  switch (g.type) {
    case 'H': c.h(a); break;
    case 'X': c.x(a); break;
    case 'Y': c.y(a); break;
    case 'Z': c.z(a); break;
    case 'RX': c.rx(a, p); break;
    case 'RY': c.ry(a, p); break;
    case 'RZ': c.rz(a, p); break;
    case 'CNOT': c.cnot(a, b); break;
    case 'CZ': c.cz(a, b); break;
    case 'SWAP': c.swap(a, b); break;
  }
}

function simulate(numQubits: number, gates: readonly QuantumGateOp[], measure: readonly number[] = []): RuquSimulation {
  const ruqu = requireBackend();
  const circuit = new ruqu.WasmQuantumCircuit(numQubits);
  try {
    for (const g of gates) applyGate(circuit, g);
    for (const q of measure) circuit.measure(q);
    return ruqu.simulate(circuit);
  } finally {
    circuit.free();
  }
}

function toState(
  numQubits: number,
  gates: QuantumGateOp[],
  probabilities: readonly number[],
  entanglementMap: Map<number, number[]>,
  measured = false
): QuantumState {
  const size = 2 ** numQubits;
  const amplitudes = new Float32Array(size);
  for (let i = 0; i < size; i++) amplitudes[i] = Math.sqrt(Math.max(0, probabilities[i] ?? 0));
  return { amplitudes, phases: new Float32Array(size), entanglementMap, numQubits, gates, measured };
}

function stateQubits(state: QuantumState): number {
  return state.numQubits ?? Math.round(Math.log2(state.amplitudes.length));
}

function stateGates(state: QuantumState): QuantumGateOp[] {
  if (state.gates) return state.gates as QuantumGateOp[];
  // A state built outside this module: only |0...0> can be replayed.
  const isGround = state.amplitudes[0] === 1 && state.amplitudes.every((a, i) => i === 0 || a === 0);
  if (!isGround) {
    throw new QuantumBackendError(
      'QuantumState has no gate history; create states with quantumInit() and apply gates to them'
    );
  }
  return [];
}

/** |0...0> on `numQubits` qubits. */
export function quantumInitState(numQubits: number): QuantumState {
  const ruqu = requireBackend();
  const max = ruqu.max_qubits();
  if (!Number.isInteger(numQubits) || numQubits < 1 || numQubits > max) {
    throw new RangeError(`numQubits must be an integer in [1, ${max}], got ${numQubits}`);
  }
  const probs = new Array<number>(2 ** numQubits).fill(0);
  probs[0] = 1;
  return toState(numQubits, [], probs, new Map());
}

/** Append a gate and re-simulate. */
export function quantumApply(state: QuantumState, gate: QuantumGateOp): QuantumState {
  requireBackend();
  if (state.measured) {
    throw new QuantumBackendError('cannot apply gates to a measured state: measurement is terminal');
  }
  const n = stateQubits(state);
  for (const q of gate.targets) checkQubit(q, n);
  if (gate.targets.length > 1 && new Set(gate.targets).size !== gate.targets.length) {
    throw new RangeError(`${gate.type} needs distinct qubits, got ${gate.targets.join(', ')}`);
  }
  const gates = [...stateGates(state), gate];
  let map = state.entanglementMap;
  if (gate.type === 'CNOT' || gate.type === 'CZ') {
    const [a = 0, b = 0] = gate.targets;
    map = new Map(map);
    map.set(a, [...(map.get(a) ?? []), b]);
  }
  return toState(n, gates, simulate(n, gates).probabilities, map);
}

/**
 * Measure `qubits` (all when empty) in the computational basis. Sampled by
 * ruqu: repeated calls on the same state give independent outcomes.
 */
export function quantumMeasureState(
  state: QuantumState,
  qubits?: number[]
): { bitstring: number[]; probability: number; collapsedState: QuantumState } {
  requireBackend();
  if (state.measured) {
    throw new QuantumBackendError('state was already measured: measurement is terminal');
  }
  const n = stateQubits(state);
  const targets = qubits && qubits.length > 0 ? qubits : Array.from({ length: n }, (_, i) => i);
  for (const q of targets) checkQubit(q, n);
  const gates = stateGates(state);
  const sim = simulate(n, gates, targets);
  let probability = 1;
  const bitstring = targets.map((q) => {
    const m = sim.measurements.find((x) => x.qubit === q);
    if (!m) throw new QuantumBackendError(`ruqu returned no measurement for qubit ${q}`);
    probability *= m.probability;
    return m.result ? 1 : 0;
  });
  return {
    bitstring,
    probability,
    collapsedState: toState(n, [...gates], sim.probabilities, new Map(state.entanglementMap), true),
  };
}

/** Thrown for operations ruqu does not back. */
export function quantumUnsupported(op: string): never {
  throw new QuantumBackendError(
    `${op} is not supported by the ${RUQU_PACKAGE} backend; build the circuit from gates instead`
  );
}
