#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Generated in the metaharness style — the `earth-pulse-harness` CLI entry point.
//
// Plain ESM JavaScript on purpose: it runs as-is via `npx earth-pulse-harness`
// with NO build step. `npm run build` (tsc) is only needed if you extend the
// TypeScript in src/ or want the `pipeline` command to run the compiled pod.
// The published package ships this file directly (see "bin" + "files" in
// package.json), so the CLI works the moment `npm install` has resolved
// @metaharness/kernel + @metaharness/host-claude-code.

import { loadKernel } from '@metaharness/kernel';
import adapter from '@metaharness/host-claude-code';

const HARNESS_NAME = 'earth-pulse-harness';

/** `earth-pulse-harness init` — boot the kernel + host adapter and report status. */
async function init() {
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  console.log(`${HARNESS_NAME} — kernel ${info.version} (${kernel.backend})`);
  console.log(`Host adapter: ${adapter.name}`);
  console.log('Earth Pulse Observatory: freeze the physics, evolve the harness.');
  console.log(`Run \`${HARNESS_NAME} doctor\` to verify the install.`);
  return 0;
}

/** `earth-pulse-harness doctor` — verify the install end-to-end (kernel + host resolve). */
async function doctor() {
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  const checks = [
    ['kernel loads', !!kernel],
    ['kernel reports a version', typeof info.version === 'string' && info.version.length > 0],
    ['kernel backend is native|wasm|js', ['native', 'wasm', 'js'].includes(kernel.backend)],
    ['host adapter has a name', typeof adapter?.name === 'string' && adapter.name.length > 0],
  ];
  let ok = true;
  for (const [label, pass] of checks) {
    console.log(`${pass ? 'PASS' : 'FAIL'} ${label}`);
    if (!pass) ok = false;
  }
  console.log(
    ok
      ? `\n${HARNESS_NAME}: all checks passed (kernel ${info.version}, ${kernel.backend} backend, host ${adapter.name})`
      : `\n${HARNESS_NAME}: doctor found problems`,
  );
  return ok ? 0 : 1;
}

/**
 * `earth-pulse-harness pipeline` — run the detect→extract→embed→score pod over the
 * bundled fixtures. Requires `npm run build` first (loads the compiled dist/).
 * Degrades gracefully with a hint if the pod has not been built yet.
 */
async function pipeline() {
  let mod;
  try {
    mod = await import('../dist/pipeline.js');
  } catch (e) {
    if (e?.code === 'ERR_MODULE_NOT_FOUND') {
      console.error(`${HARNESS_NAME}: pipeline not built yet — run \`npm run build\` first.`);
      return 1;
    }
    throw e;
  }
  const { readFileSync } = await import('node:fs');
  const { fileURLToPath } = await import('node:url');
  const { dirname, resolve } = await import('node:path');
  const here = dirname(fileURLToPath(import.meta.url));
  const win = JSON.parse(readFileSync(resolve(here, '../tests/fixtures/sample-window.json'), 'utf8'));
  const env = JSON.parse(readFileSync(resolve(here, '../tests/fixtures/sample-env.json'), 'utf8'));
  const out = mod.runPipeline(win, env);
  console.log(JSON.stringify(out, null, 2));
  return out.event ? 0 : 1;
}

/**
 * Dispatch one CLI invocation. Exported (not just run on import) so a test can
 * drive it without spawning a subprocess. Returns the intended exit code.
 */
export async function run(argv) {
  const cmd = argv[0] ?? 'init';
  switch (cmd) {
    case 'init':
      return init();
    case 'doctor':
      return doctor();
    case 'pipeline':
      return pipeline();
    case '--version':
    case '-v': {
      const kernel = await loadKernel();
      console.log(kernel.version());
      return 0;
    }
    case '--help':
    case '-h':
      console.log(
        `Usage: ${HARNESS_NAME} <command>\n\n  init      boot the kernel + host adapter (default)\n  doctor    verify the install end-to-end\n  pipeline  run detect->extract->embed->score over the bundled fixtures\n  --version print the kernel version`,
      );
      return 0;
    default:
      console.error(`Unknown command: ${cmd}. Try \`${HARNESS_NAME} --help\`.`);
      return 2;
  }
}

// CLI guard: execute only when invoked directly (not when imported by a test).
// npm's bin shims pass a NON-normalized argv[1] and may differ in case, so
// realpath BOTH sides before comparing — a naive string === misses the npx/shim
// path and the CLI silently no-ops.
import { fileURLToPath } from 'node:url';
import { realpathSync } from 'node:fs';
import { argv } from 'node:process';
const invokedDirectly = (() => {
  if (!argv[1]) return false;
  try {
    const a = realpathSync(argv[1]);
    const b = realpathSync(fileURLToPath(import.meta.url));
    return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;
  } catch {
    return false;
  }
})();
if (invokedDirectly) {
  run(argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
