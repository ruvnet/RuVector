// @ruvector/ruvllm/gpu-spool-adapter -> @ruvector/ruvllm-gpu-loop/gpu-spool-adapter (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('gpu-spool-adapter');
export const { submit, inspect, cleanup } = mod;
