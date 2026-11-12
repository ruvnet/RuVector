'use strict';

/**
 * @ruvector/all — opt-in umbrella over the RuVector toolkit.
 *
 * Nothing is loaded when this file is required. CommonJS members are exposed
 * as lazy getters (the first property access calls `require`). ESM-only
 * members (the wasm-bindgen packages) cannot be `require`d on every Node 20
 * release, so they are exposed as cached async loaders that use `import()`.
 */

/**
 * Member manifest. `kind: 'cjs'` → lazy getter `all[key]`.
 * `kind: 'esm'` → async loader `all[loader]()`.
 */
const members = Object.freeze([
  { key: 'ruvector', pkg: 'ruvector', kind: 'cjs' },
  { key: 'rvf', pkg: '@ruvector/rvf', kind: 'cjs' },
  { key: 'sona', pkg: '@ruvector/sona', kind: 'cjs' },
  { key: 'diskann', pkg: '@ruvector/diskann', kind: 'cjs' },
  { key: 'tinyDancer', pkg: '@ruvector/tiny-dancer', kind: 'cjs' },
  { key: 'rvlite', pkg: 'rvlite', kind: 'cjs' },
  { key: 'rvfSolver', pkg: '@ruvector/rvf-solver', kind: 'cjs' },
  { key: 'rabitqWasm', pkg: '@ruvector/rabitq-wasm', kind: 'esm', loader: 'loadRabitqWasm' },
  { key: 'graphWasm', pkg: '@ruvector/graph-wasm', kind: 'esm', loader: 'loadGraphWasm' },
  { key: 'ruvllmWasm', pkg: '@ruvector/ruvllm-wasm', kind: 'esm', loader: 'loadRuvllmWasm' },
  { key: 'emergentTime', pkg: '@ruvector/emergent-time', kind: 'esm', loader: 'loadEmergentTime' },
  { key: 'rvfWasm', pkg: '@ruvector/rvf-wasm', kind: 'esm', loader: 'loadRvfWasm' },
  { key: 'eMeans', pkg: '@ruvector/e-means', kind: 'esm', loader: 'loadEMeans' },
].map(Object.freeze));

const byKey = new Map(members.map((m) => [m.key, m]));
const cache = new Map(); // key -> module (cjs) | Promise<module> (esm)

function requireMember(m) {
  if (!cache.has(m.key)) cache.set(m.key, require(m.pkg));
  return cache.get(m.key);
}

function importMember(m) {
  if (!cache.has(m.key)) {
    const p = import(m.pkg);
    // Drop a failed import from the cache so a later call can retry.
    p.catch(() => cache.delete(m.key));
    cache.set(m.key, p);
  }
  return cache.get(m.key);
}

function memberOrThrow(key) {
  const m = byKey.get(key);
  if (!m) throw new TypeError(`@ruvector/all: unknown member "${key}"`);
  return m;
}

/** Load any member by key. Always async, works for CJS and ESM members. */
function load(key) {
  const m = byKey.get(key);
  if (!m) return Promise.reject(new TypeError(`@ruvector/all: unknown member "${key}"`));
  if (m.kind === 'esm') return importMember(m);
  try {
    return Promise.resolve(requireMember(m));
  } catch (err) {
    return Promise.reject(err);
  }
}

/** True once a member has been loaded (or an ESM import has started). */
function isLoaded(key) {
  memberOrThrow(key);
  return cache.has(key);
}

for (const m of members) {
  if (m.kind === 'cjs') {
    Object.defineProperty(module.exports, m.key, {
      enumerable: true,
      configurable: false,
      get: () => requireMember(m),
    });
  } else {
    Object.defineProperty(module.exports, m.loader, {
      enumerable: true,
      configurable: false,
      writable: false,
      value: () => importMember(m),
    });
  }
}

Object.defineProperty(module.exports, 'members', { enumerable: true, value: members });
Object.defineProperty(module.exports, 'load', { enumerable: true, value: load });
Object.defineProperty(module.exports, 'isLoaded', { enumerable: true, value: isLoaded });
