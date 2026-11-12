const { accessSync } = require('node:fs');
const { join } = require('node:path');
for (const target of ['node', 'web']) {
  for (const file of ['ruvector_mincut_wasm.js', 'ruvector_mincut_wasm.d.ts', 'ruvector_mincut_wasm_bg.wasm']) {
    accessSync(join(__dirname, '..', target, file));
  }
}
accessSync(join(__dirname, '../node/package.json'));
