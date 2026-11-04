/**
 * @ruvector/all — opt-in umbrella over the RuVector toolkit.
 *
 * Requiring this package loads nothing. Each member is loaded on first use:
 *
 * - CommonJS members are lazy getters: reading `all.sona` calls
 *   `require('@ruvector/sona')` the first time and returns the cached module
 *   afterwards.
 * - ESM-only members (`type: "module"` wasm-bindgen packages with no `require`
 *   export) are exposed as async loaders that use `import()`, because
 *   `require()` of ESM is not available on every supported Node release
 *   (>=20.0.0). Example: `const graph = await all.loadGraphWasm()`.
 *   wasm-bindgen packages may still need their own init call in Node; see
 *   each package's README.
 *
 * `load(key)` is a uniform async loader that works for every member.
 *
 * From ESM, use the default import only (`import all from '@ruvector/all'`).
 * The getters are defined at runtime, so Node cannot detect them as named
 * exports: `import { sona } from '@ruvector/all'` fails in Node ESM even
 * though TypeScript accepts it.
 */

/** Lazy getter → `require('ruvector')`. */
export declare const ruvector: typeof import('ruvector');
/** Lazy getter → `require('@ruvector/rvf')`. */
export declare const rvf: typeof import('@ruvector/rvf');
/** Lazy getter → `require('@ruvector/sona')`. */
export declare const sona: typeof import('@ruvector/sona');
/** Lazy getter → `require('@ruvector/diskann')`. */
export declare const diskann: typeof import('@ruvector/diskann');
/** Lazy getter → `require('@ruvector/tiny-dancer')`. */
export declare const tinyDancer: typeof import('@ruvector/tiny-dancer');
/** Lazy getter → `require('rvlite')` (resolves the package's CJS build). */
export declare const rvlite: typeof import('rvlite');
/** Lazy getter → `require('@ruvector/rvf-solver')`. */
export declare const rvfSolver: typeof import('@ruvector/rvf-solver');

/** ESM-only. Async loader → `import('@ruvector/rabitq-wasm')`. */
export declare function loadRabitqWasm(): Promise<typeof import('@ruvector/rabitq-wasm')>;
/** ESM-only. Async loader → `import('@ruvector/graph-wasm')`. */
export declare function loadGraphWasm(): Promise<typeof import('@ruvector/graph-wasm')>;
/** ESM-only. Async loader → `import('@ruvector/ruvllm-wasm')`. */
export declare function loadRuvllmWasm(): Promise<typeof import('@ruvector/ruvllm-wasm')>;
/** ESM-only. Async loader → `import('@ruvector/emergent-time')`. */
export declare function loadEmergentTime(): Promise<typeof import('@ruvector/emergent-time')>;
/** ESM-only. Async loader → `import('@ruvector/rvf-wasm')`. */
export declare function loadRvfWasm(): Promise<typeof import('@ruvector/rvf-wasm')>;

export interface MemberModules {
  ruvector: typeof import('ruvector');
  rvf: typeof import('@ruvector/rvf');
  sona: typeof import('@ruvector/sona');
  diskann: typeof import('@ruvector/diskann');
  tinyDancer: typeof import('@ruvector/tiny-dancer');
  rvlite: typeof import('rvlite');
  rvfSolver: typeof import('@ruvector/rvf-solver');
  rabitqWasm: typeof import('@ruvector/rabitq-wasm');
  graphWasm: typeof import('@ruvector/graph-wasm');
  ruvllmWasm: typeof import('@ruvector/ruvllm-wasm');
  emergentTime: typeof import('@ruvector/emergent-time');
  rvfWasm: typeof import('@ruvector/rvf-wasm');
}

export type MemberKey = keyof MemberModules;

export interface MemberInfo {
  readonly key: MemberKey;
  /** npm package name. */
  readonly pkg: string;
  /** `cjs` → lazy getter; `esm` → async loader named by `loader`. */
  readonly kind: 'cjs' | 'esm';
  readonly loader?: string;
}

/** Manifest of every bundled package. */
export declare const members: ReadonlyArray<MemberInfo>;

/** Uniform async loader for any member (CJS or ESM). */
export declare function load<K extends MemberKey>(key: K): Promise<MemberModules[K]>;

/** True once a member has been loaded (or its ESM import has started). */
export declare function isLoaded(key: MemberKey): boolean;
