"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.RvfSolver = void 0;
const wire_1 = require("./wire");
let wasmExports = null;
async function getWasm() {
    if (wasmExports)
        return wasmExports;
    // Dynamic import to support both CJS and ESM
    const initModule = await Promise.resolve().then(() => __importStar(require('../pkg/rvf_solver')));
    const init = initModule.default || initModule;
    wasmExports = await init();
    return wasmExports;
}
/** Puzzle difficulty range accepted by the solver. */
const MIN_DIFFICULTY = 1;
const MAX_DIFFICULTY = 10;
function readJson(wasm, handle, lenFn, readFn) {
    const len = lenFn(handle);
    if (len <= 0)
        return null;
    const ptr = wasm.rvf_solver_alloc(len);
    if (ptr === 0)
        return null;
    try {
        readFn(handle, ptr);
        const buf = new Uint8Array(wasm.memory.buffer, ptr, len);
        const text = new TextDecoder().decode(buf);
        return JSON.parse(text);
    }
    finally {
        wasm.rvf_solver_free(ptr, len);
    }
}
/**
 * RVF Self-Learning Solver.
 *
 * Wraps the rvf-solver-wasm WASM module providing:
 * - PolicyKernel with Thompson Sampling (two-signal model)
 * - Context-bucketed bandit (18 buckets)
 * - KnowledgeCompiler with signature-based pattern cache
 * - Speculative dual-path execution
 * - Three-loop adaptive solver (fast/medium/slow)
 * - SHAKE-256 tamper-evident witness chain
 */
class RvfSolver {
    constructor(handle, wasm) {
        this.handle = handle;
        this.wasm = wasm;
    }
    /**
     * Create a new solver instance.
     * Initializes the WASM module on first call.
     */
    static async create() {
        const wasm = await getWasm();
        const handle = wasm.rvf_solver_create();
        if (handle < 0) {
            throw new Error('Failed to create solver instance (max 8 concurrent instances)');
        }
        return new RvfSolver(handle, wasm);
    }
    /**
     * Train the solver on generated puzzles.
     *
     * Uses the three-loop architecture:
     * - Fast loop: constraint propagation solver
     * - Medium loop: PolicyKernel skip-mode selection
     * - Slow loop: KnowledgeCompiler pattern distillation
     *
     * @throws RangeError if `count` is not a positive integer, a difficulty is
     *   outside 1-10, `minDifficulty > maxDifficulty`, or `seed` is not an integer.
     */
    train(options) {
        const count = (0, wire_1.checkInt)('count', options.count, 1);
        const minDiff = (0, wire_1.checkInt)('minDifficulty', options.minDifficulty ?? MIN_DIFFICULTY, MIN_DIFFICULTY, MAX_DIFFICULTY);
        const maxDiff = (0, wire_1.checkInt)('maxDifficulty', options.maxDifficulty ?? MAX_DIFFICULTY, MIN_DIFFICULTY, MAX_DIFFICULTY);
        if (minDiff > maxDiff) {
            throw new RangeError(`minDifficulty (${minDiff}) must not exceed maxDifficulty (${maxDiff})`);
        }
        const [seedLo, seedHi] = (0, wire_1.splitSeed)(options.seed);
        const correct = this.wasm.rvf_solver_train(this.handle, count, minDiff, maxDiff, seedLo, seedHi);
        if (correct === -2) {
            throw new RangeError('Training failed: invalid arguments');
        }
        if (correct < 0) {
            throw new Error('Training failed: invalid handle');
        }
        const result = readJson(this.wasm, this.handle, (h) => this.wasm.rvf_solver_result_len(h), (h, p) => this.wasm.rvf_solver_result_read(h, p));
        return result ? (0, wire_1.toTrainResult)(result) : {
            trained: count,
            correct,
            accuracy: correct / count,
            patternsLearned: 0,
        };
    }
    /**
     * Run the full acceptance test with training/holdout cycles.
     *
     * Runs all three ablation modes:
     * - Mode A: Fixed heuristic policy
     * - Mode B: Compiler-suggested policy
     * - Mode C: Learned Thompson Sampling policy
     *
     * Returns the full manifest with per-cycle metrics and witness chain.
     *
     * @throws RangeError if `holdoutSize`, `cycles` or `stepBudget` is not a
     *   positive integer, `trainingPerCycle` is not a non-negative integer, or
     *   `seed` is not an integer.
     */
    acceptance(options) {
        const opts = options ?? {};
        const holdout = (0, wire_1.checkInt)('holdoutSize', opts.holdoutSize ?? 50, 1);
        const training = (0, wire_1.checkInt)('trainingPerCycle', opts.trainingPerCycle ?? 200, 0);
        const cycles = (0, wire_1.checkInt)('cycles', opts.cycles ?? 5, 1);
        const budget = (0, wire_1.checkInt)('stepBudget', opts.stepBudget ?? 500, 1);
        const [seedLo, seedHi] = (0, wire_1.splitSeed)(opts.seed);
        const status = this.wasm.rvf_solver_acceptance(this.handle, holdout, training, cycles, budget, seedLo, seedHi);
        if (status === -2) {
            throw new RangeError('Acceptance test failed: invalid arguments');
        }
        if (status < 0) {
            throw new Error('Acceptance test failed: invalid handle');
        }
        const manifest = readJson(this.wasm, this.handle, (h) => this.wasm.rvf_solver_result_len(h), (h, p) => this.wasm.rvf_solver_result_read(h, p));
        if (!manifest) {
            throw new Error('Failed to read acceptance manifest');
        }
        return (0, wire_1.toManifest)(manifest);
    }
    /**
     * Get the current policy state (Thompson Sampling parameters,
     * context buckets, KnowledgeCompiler cache stats).
     */
    policy() {
        const raw = readJson(this.wasm, this.handle, (h) => this.wasm.rvf_solver_policy_len(h), (h, p) => this.wasm.rvf_solver_policy_read(h, p));
        return raw ? (0, wire_1.toPolicyState)(raw) : null;
    }
    /**
     * Get the raw SHAKE-256 witness chain bytes.
     *
     * The witness chain is 73 bytes per entry and provides
     * tamper-evident proof of all training/acceptance operations.
     * Verifiable using `rvf_witness_verify` from `@ruvector/rvf-wasm`.
     */
    witnessChain() {
        const len = this.wasm.rvf_solver_witness_len(this.handle);
        if (len <= 0)
            return null;
        const ptr = this.wasm.rvf_solver_alloc(len);
        if (ptr === 0)
            return null;
        try {
            this.wasm.rvf_solver_witness_read(this.handle, ptr);
            const buf = new Uint8Array(this.wasm.memory.buffer, ptr, len);
            // Copy to avoid referencing WASM memory after free
            return new Uint8Array(buf);
        }
        finally {
            this.wasm.rvf_solver_free(ptr, len);
        }
    }
    /**
     * Destroy the solver instance and free WASM resources.
     */
    destroy() {
        if (this.handle > 0) {
            this.wasm.rvf_solver_destroy(this.handle);
            this.handle = 0;
        }
    }
}
exports.RvfSolver = RvfSolver;
//# sourceMappingURL=solver.js.map