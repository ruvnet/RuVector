#!/usr/bin/env node
// Compiles test/types/learning-import-compat.ts against dist/*.d.ts: LearningEngine.import()
// must accept a pre-0.3.22 export() shape (no `neural`). Requires `npm run build` first.
'use strict';
const { execFileSync } = require('child_process');
const path = require('path');

const root = path.join(__dirname, '..');
const tsc = require.resolve('typescript/bin/tsc', { paths: [root] });
try {
  execFileSync(process.execPath, [tsc, '--noEmit', '--strict', '--skipLibCheck', '--target', 'ES2020',
    '--module', 'commonjs', '--moduleResolution', 'node', '--esModuleInterop', '--types', 'node',
    path.join(__dirname, 'types', 'learning-import-compat.ts')], { cwd: root, stdio: 'pipe', encoding: 'utf8' });
} catch (e) {
  console.error(e.stdout || e.message);
  console.error('FAIL: LearningEngine.import() rejects a pre-0.3.22 export() shape');
  process.exit(1);
}
console.log('ok: LearningEngine.import() accepts pre-0.3.22 and current export() shapes');
