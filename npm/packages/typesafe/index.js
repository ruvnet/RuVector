'use strict';

// Platform loader for @ruvector/typesafe — mirrors npm/packages/router/index.js.
// Order: native N-API binary for this platform, then the wasm-pack (nodejs)
// fallback. `TYPESAFE_BACKEND=wasm` forces the fallback. No `child_process`,
// no `fetch`, no network — the module only ever `require()`s a local artifact.

const path = require('path');

// process.platform + process.arch -> the .node filename build-native.sh writes.
const platformMap = {
  linux: {
    x64: 'typesafe.linux-x64-gnu.node',
    arm64: 'typesafe.linux-arm64-gnu.node',
  },
  darwin: {
    x64: 'typesafe.darwin-x64.node',
    arm64: 'typesafe.darwin-arm64.node',
  },
  win32: {
    x64: 'typesafe.win32-x64-msvc.node',
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
    if (err && /not found|cannot open|no such file/i.test(String(err.message))) return null;
    throw err;
  }
}

function loadNative() {
  const file = platformMap[process.platform] && platformMap[process.platform][process.arch];
  if (!file) return null;
  // 1. A binary bundled in or built into this package (local dev, and the
  //    self-contained 0.1.x releases).
  const local = tryRequire(path.join(__dirname, 'native', file));
  if (local) return local;
  // 2. The per-platform package, once the optionalDependencies bump lands
  //    (ADR-002 §5: those packages must exist on npm before the meta package
  //    declares them). Resolving it by name here keeps the loader ready.
  return tryRequire(`@ruvector/typesafe-${file.slice('typesafe.'.length, -'.node'.length)}`);
}

function loadWasm() {
  // wasm-pack --target nodejs output; crate name dashes -> underscores.
  return require('./wasm/ruvector_typesafe_wasm.js');
}

let impl;
let backend;

if (process.env.TYPESAFE_BACKEND === 'wasm') {
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
  Engine: impl.Engine,
  version: impl.version,
  backend,
};
