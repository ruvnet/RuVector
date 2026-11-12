'use strict';

// Platform loader for @ruvector/kge — mirrors npm/packages/typesafe/index.js.
// Order: native N-API binary for this platform, then the wasm-pack (nodejs)
// fallback. `KGE_BACKEND=wasm` forces the fallback. No `child_process`, no
// `fetch`, no network — the module only ever `require()`s a local artifact.

const fs = require('fs');
const path = require('path');

// process.platform + process.arch -> the .node filename build-native.sh writes.
const platformMap = {
  linux: {
    x64: 'kge.linux-x64-gnu.node',
    arm64: 'kge.linux-arm64-gnu.node',
  },
  darwin: {
    x64: 'kge.darwin-x64.node',
    arm64: 'kge.darwin-arm64.node',
  },
  win32: {
    x64: 'kge.win32-x64-msvc.node',
  },
};

// A missing artifact is not an error — we fall through to the next candidate.
// A real dlopen/ABI failure is, and must surface rather than silently
// downgrading a native install to wasm.
function tryRequire(id) {
  try {
    return require(id);
  } catch (err) {
    if (err && err.code === 'MODULE_NOT_FOUND') return null;
    throw err;
  }
}

function loadNative() {
  const file = platformMap[process.platform] && platformMap[process.platform][process.arch];
  if (!file) return null;
  // 1. A binary bundled in or built into this package (local dev, and the
  //    self-contained 0.1.x releases).
  // If the file is THERE, any failure to load it is a real fault (a missing
  // system library, an ABI mismatch) and must surface rather than silently
  // downgrading to wasm. `KGE_BACKEND=wasm` is the documented escape.
  const local = path.join(__dirname, 'native', file);
  if (fs.existsSync(local)) return require(local);
  // 2. The per-platform package, once the optionalDependencies bump lands
  //    (ADR-001 §5: those packages must exist on npm before the meta package
  //    declares them). Resolving it by name here keeps the loader ready.
  return tryRequire(`@ruvector/kge-${file.slice('kge.'.length, -'.node'.length)}`);
}

function loadWasm() {
  // wasm-pack --target nodejs output; crate name dashes -> underscores.
  return require('./wasm/ruvector_kge_wasm.js');
}

let impl;
let backend;

if (process.env.KGE_BACKEND === 'wasm') {
  impl = loadWasm();
  backend = 'wasm';
} else {
  impl = loadNative();
  if (impl) {
    backend = 'native';
  } else {
    impl = loadWasm();
    backend = 'wasm';
  }
}

module.exports = {
  Model: impl.Model,
  version: impl.version,
  backend,
};
