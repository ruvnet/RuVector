'use strict';
// @ruvector/ruvllm/gpu-host -> @ruvector/ruvllm-gpu-loop/gpu-host (lazy; see ./_load.cjs)
module.exports = require('./_load.cjs').loadGpuLoop('gpu-host');
