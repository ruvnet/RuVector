'use strict';

// Platform loader for @ruvector/kge — mirrors npm/packages/typesafe/index.js.
// Order: native N-API binary for this platform, then the wasm-pack (nodejs)
// fallback. `KGE_BACKEND=wasm` forces the fallback. No `child_process`, no
// `fetch`, no network — the module only ever `require()`s a local artifact.

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

function loadNative() {
  const file = platformMap[process.platform] && platformMap[process.platform][process.arch];
  if (!file) return null;
  try {
    return require(path.join(__dirname, 'native', file));
  } catch (err) {
    if (err && err.code === 'MODULE_NOT_FOUND') return null;
    // A real dlopen/ABI error should surface, not be silently swallowed.
    if (err && /not found|cannot open|no such file/i.test(String(err.message))) return null;
    throw err;
  }
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
