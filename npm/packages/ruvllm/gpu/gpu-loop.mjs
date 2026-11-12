// @ruvector/ruvllm/gpu-loop -> @ruvector/ruvllm-gpu-loop/gpu-loop (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('gpu-loop');
export const { readLoopConfig, readLoopStatus, tickLoop } = mod;
