// @ruvector/ruvllm/engine -> @ruvector/ruvllm-gpu-loop (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('');
export const { sha256, parseJson, checkedPath, fileDigest, runExperiment } = mod;
