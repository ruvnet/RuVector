// @ruvector/ruvllm/vast-host-adapter -> @ruvector/ruvllm-gpu-loop/vast-host-adapter (lazy; see ./_load.mjs)
import { loadGpuLoop } from './_load.mjs';

const mod = await loadGpuLoop('vast-host-adapter');
export const { HOST_CONFIG_ENV, readHostConfig, onstartScript, createVastAdapter, submit, inspect, cleanup } = mod;
