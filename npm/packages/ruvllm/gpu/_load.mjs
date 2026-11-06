// Lazy bridge from @ruvector/ruvllm subpaths to @ruvector/ruvllm-gpu-loop.
// The GPU job loop, Vast.ai host adapter, adapter archives and bounded
// weight-experiment engine shipped as @ruvector/ruvllm 4.0.x; they now live in
// their own package, installed here as an optional dependency. Nothing is
// loaded until one of these subpaths is imported.
import { createRequire } from 'node:module';

export const GPU_LOOP_PACKAGE = '@ruvector/ruvllm-gpu-loop';
const require = createRequire(import.meta.url);

export function gpuLoopMissingError(specifier, cause) {
  const error = new Error(
    `${specifier} needs the optional package ${GPU_LOOP_PACKAGE}, which is not installed ` +
    `(optional dependencies are skipped by --omit=optional / --no-optional, and it requires Node 22+). ` +
    `Install it with: npm install ${GPU_LOOP_PACKAGE}`,
    { cause });
  error.code = 'RUVLLM_GPU_LOOP_MISSING';
  return error;
}

function assertInstalled(specifier) {
  try {
    require.resolve(`${GPU_LOOP_PACKAGE}/package.json`);
  } catch (cause) {
    throw gpuLoopMissingError(specifier, cause);
  }
}

export async function loadGpuLoop(subpath) {
  const specifier = subpath ? `${GPU_LOOP_PACKAGE}/${subpath}` : GPU_LOOP_PACKAGE;
  assertInstalled(subpath ? `@ruvector/ruvllm/${subpath}` : '@ruvector/ruvllm/engine');
  return import(specifier);
}
