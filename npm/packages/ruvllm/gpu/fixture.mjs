// @ruvector/ruvllm/fixture -> @ruvector/ruvllm-gpu-loop/fixture (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('fixture');
export const { createFixture } = mod;
