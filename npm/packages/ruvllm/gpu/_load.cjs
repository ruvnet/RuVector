'use strict';
// CommonJS side of gpu/_load.mjs. @ruvector/ruvllm-gpu-loop is ESM-only;
// require() of it works on Node >= 20.19 / 22.12 (require(esm)). On older
// Node use `await import('@ruvector/ruvllm/<subpath>')` instead.
const GPU_LOOP_PACKAGE = '@ruvector/ruvllm-gpu-loop';

function gpuLoopMissingError(specifier, cause) {
  const error = new Error(
    `${specifier} needs the optional package ${GPU_LOOP_PACKAGE}, which is not installed ` +
    `(optional dependencies are skipped by --omit=optional / --no-optional, and it requires Node 22+). ` +
    `Install it with: npm install ${GPU_LOOP_PACKAGE}`);
  error.code = 'RUVLLM_GPU_LOOP_MISSING';
  error.cause = cause;
  return error;
}

function loadGpuLoop(subpath) {
  const ours = subpath ? `@ruvector/ruvllm/${subpath}` : '@ruvector/ruvllm/engine';
  try {
    require.resolve(`${GPU_LOOP_PACKAGE}/package.json`);
  } catch (cause) {
    throw gpuLoopMissingError(ours, cause);
  }
  const specifier = subpath ? `${GPU_LOOP_PACKAGE}/${subpath}` : GPU_LOOP_PACKAGE;
  try {
    return require(specifier);
  } catch (error) {
    if (error && (error.code === 'ERR_REQUIRE_ESM' || error.code === 'ERR_REQUIRE_ASYNC_MODULE')) {
      const wrapped = new Error(`${ours} is ESM-only on this Node version (${process.version}); use: await import('${ours}')`);
      wrapped.code = 'RUVLLM_GPU_LOOP_ESM_ONLY';
      wrapped.cause = error;
      throw wrapped;
    }
    throw error;
  }
}

module.exports = { GPU_LOOP_PACKAGE, gpuLoopMissingError, loadGpuLoop };
