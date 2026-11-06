// @ruvector/ruvllm/adapters -> @ruvector/ruvllm-gpu-loop/adapters (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('adapters');
export const { exportArchive, trainingPlan, selectDevelopmentCandidate } = mod;
