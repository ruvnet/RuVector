'use strict';
/**
 * Run a CLI shipped by @ruvector/ruvllm-gpu-loop (the GPU job loop, adapter
 * archives and bounded weight experiments that were @ruvector/ruvllm 4.0.x).
 * Used by `ruvllm gpu ...` and the `ruvllm-gpu-loop` / `ruvllm-adapters` bins.
 */
const { spawnSync } = require('node:child_process');
const { readFileSync } = require('node:fs');
const { dirname, resolve } = require('node:path');

const GPU_LOOP_PACKAGE = '@ruvector/ruvllm-gpu-loop';

function resolveGpuBin(binName) {
  let manifestPath;
  try {
    manifestPath = require.resolve(`${GPU_LOOP_PACKAGE}/package.json`);
  } catch {
    const error = new Error(
      `"${binName}" is provided by the optional package ${GPU_LOOP_PACKAGE}, which is not installed ` +
      `(optional dependencies are skipped by --omit=optional, and it requires Node 22+).\n` +
      `Install it with: npm install ${GPU_LOOP_PACKAGE}`);
    error.code = 'RUVLLM_GPU_LOOP_MISSING';
    throw error;
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const rel = manifest.bin && manifest.bin[binName];
  if (typeof rel !== 'string') {
    const error = new Error(`${GPU_LOOP_PACKAGE}@${manifest.version} has no "${binName}" command`);
    error.code = 'RUVLLM_GPU_LOOP_BIN_MISSING';
    throw error;
  }
  return resolve(dirname(manifestPath), rel);
}

/** Run `binName` from the GPU-loop package with `args`; exits with its status. */
function delegate(binName, args) {
  let binPath;
  try {
    binPath = resolveGpuBin(binName);
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
  const result = spawnSync(process.execPath, [binPath, ...args], { stdio: 'inherit' });
  if (result.error) {
    console.error(`Error: could not run ${binName}: ${result.error.message}`);
    process.exit(1);
  }
  if (result.signal) process.kill(process.pid, result.signal);
  process.exit(result.status === null ? 1 : result.status);
}

/**
 * `ruvllm gpu <args>`:
 *   gpu loop <args>      -> ruvllm-gpu-loop (status, tick, ...)
 *   gpu adapters <args>  -> ruvllm-adapters (export, training-plan, select)
 *   gpu <args>           -> ruvllm-gpu      (run, fixture, --help; the 4.0.x `ruvllm` CLI)
 */
function runGpuCommand(args) {
  const [first, ...rest] = args;
  if (first === 'loop') return delegate('ruvllm-gpu-loop', rest);
  if (first === 'adapters') return delegate('ruvllm-adapters', rest);
  if (first === undefined) return delegate('ruvllm-gpu', ['--help']);
  return delegate('ruvllm-gpu', args);
}

module.exports = { GPU_LOOP_PACKAGE, resolveGpuBin, delegate, runGpuCommand };
