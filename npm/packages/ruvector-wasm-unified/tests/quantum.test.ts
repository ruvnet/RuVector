/**
 * Quantum gates via the optional @ruvector/ruqu-wasm backend (installed as an
 * optionalDependency in development).
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  createExoticEngine,
  createUnifiedEngine,
  initQuantumBackend,
  quantumBackendStatus,
  QuantumBackendError,
} from '../src/index';
import { resetQuantumBackend, setRuquImporterForTesting } from '../src/quantum';

const sq = (x: number | undefined) => (x ?? 0) ** 2;

describe('without an initialised backend', () => {
  beforeAll(() => resetQuantumBackend());
  afterAll(() => resetQuantumBackend());

  it('throws a clear error instead of returning fake results', () => {
    const exotic = createExoticEngine();
    expect(() => exotic.quantumInit(2)).toThrow(QuantumBackendError);
    expect(() => exotic.quantumInit(2)).toThrow(/initQuantumBackend\(\)/);
    expect(() => exotic.quantumInit(2)).toThrow(/@ruvector\/ruqu-wasm/);
  });

  it('reports a missing package with an install hint', async () => {
    setRuquImporterForTesting(() =>
      Promise.reject(Object.assign(new Error("Cannot find package '@ruvector/ruqu-wasm'"), { code: 'ERR_MODULE_NOT_FOUND' }))
    );
    try {
      await expect(initQuantumBackend()).rejects.toThrow(/is not installed.*npm install @ruvector\/ruqu-wasm/);
      expect(quantumBackendStatus()).toBe('missing');
      expect(() => createExoticEngine().quantumInit(1)).toThrow(/is not installed/);
      // engine.init() stays best-effort: other engines keep working
      const engine = await createUnifiedEngine();
      await expect(engine.init()).resolves.toBeUndefined();
    } finally {
      setRuquImporterForTesting(undefined);
      resetQuantumBackend();
    }
  });
});

describe('with @ruvector/ruqu-wasm', () => {
  beforeAll(async () => {
    resetQuantumBackend();
    await initQuantumBackend();
  });

  it('loads the real backend', () => {
    expect(quantumBackendStatus()).toBe('ready');
  });

  it('Hadamard puts a qubit in equal superposition', () => {
    const ex = createExoticEngine();
    const s = ex.quantumHadamard(ex.quantumInit(1), 0);
    expect(sq(s.amplitudes[0])).toBeCloseTo(0.5, 5);
    expect(sq(s.amplitudes[1])).toBeCloseTo(0.5, 5);
  });

  it('H + CNOT gives a Bell state', () => {
    const ex = createExoticEngine();
    const bell = ex.quantumCnot(ex.quantumHadamard(ex.quantumInit(2), 0), 0, 1);
    expect(sq(bell.amplitudes[0])).toBeCloseTo(0.5, 5);
    expect(sq(bell.amplitudes[1])).toBeCloseTo(0, 5);
    expect(sq(bell.amplitudes[2])).toBeCloseTo(0, 5);
    expect(sq(bell.amplitudes[3])).toBeCloseTo(0.5, 5);
    expect(bell.entanglementMap.get(0)).toEqual([1]);
  });

  it('measuring a Bell state gives correlated ~50/50 outcomes', () => {
    const ex = createExoticEngine();
    const bell = ex.quantumCnot(ex.quantumHadamard(ex.quantumInit(2), 0), 0, 1);
    const counts: Record<string, number> = {};
    const trials = 400;
    for (let i = 0; i < trials; i++) {
      const m = ex.quantumMeasure(bell);
      const key = m.bitstring.join('');
      counts[key] = (counts[key] ?? 0) + 1;
      expect(m.bitstring).toHaveLength(2);
      expect(m.probability).toBeCloseTo(0.5, 5);
    }
    expect(Object.keys(counts).sort()).toEqual(['00', '11']);
    expect(counts['00']! / trials).toBeGreaterThan(0.35);
    expect(counts['00']! / trials).toBeLessThan(0.65);
  });

  it('collapsed state is terminal', () => {
    const ex = createExoticEngine();
    const m = ex.quantumMeasure(ex.quantumHadamard(ex.quantumInit(1), 0));
    expect(m.collapsedState.measured).toBe(true);
    expect(sq(m.collapsedState.amplitudes[m.bitstring[0]!])).toBeCloseTo(1, 5);
    expect(() => ex.quantumHadamard(m.collapsedState, 0)).toThrow(/terminal/);
  });

  it('X on qubit 0 sets the least significant bit', () => {
    const ex = createExoticEngine();
    const h = ex.quantumHadamard(ex.quantumHadamard(ex.quantumInit(2), 0), 0); // H·H = I
    expect(sq(h.amplitudes[0])).toBeCloseTo(1, 5);
    const m = ex.quantumMeasure(ex.quantumCnot(ex.quantumHadamard(ex.quantumInit(2), 1), 1, 0), [0]);
    expect(m.bitstring).toHaveLength(1);
  });

  it('phase gate leaves probabilities unchanged', () => {
    const ex = createExoticEngine();
    const s = ex.quantumPhase(ex.quantumHadamard(ex.quantumInit(1), 0), 0, Math.PI / 4);
    expect(sq(s.amplitudes[0])).toBeCloseTo(0.5, 5);
  });

  it('rejects out-of-range qubits', () => {
    const ex = createExoticEngine();
    expect(() => ex.quantumHadamard(ex.quantumInit(2), 2)).toThrow(RangeError);
    expect(() => ex.quantumCnot(ex.quantumInit(2), 1, 1)).toThrow(RangeError);
  });

  it('amplify and VQE throw rather than return placeholders', () => {
    const ex = createExoticEngine();
    const s = ex.quantumInit(2);
    expect(() => ex.quantumAmplify(s, (a) => a)).toThrow(QuantumBackendError);
    expect(() => ex.quantumVqe(new Float32Array(4))).toThrow(QuantumBackendError);
  });

  it('UnifiedEngine.init() wires the backend', async () => {
    resetQuantumBackend();
    const engine = await createUnifiedEngine();
    await engine.init();
    const s = engine.exotic.quantumHadamard(engine.exotic.quantumInit(1), 0);
    expect(sq(s.amplitudes[1])).toBeCloseTo(0.5, 5);
    expect(engine.version()).toBe('1.1.0');
  });
});
