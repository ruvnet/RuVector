// @ruvector/ruvllm/gpu-host -> @ruvector/ruvllm-gpu-loop/gpu-host (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('gpu-host');
export const { loadHostAdapter } = mod;
