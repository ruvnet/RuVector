'use strict';
// @ruvector/ruvllm/gpu-spool-adapter -> @ruvector/ruvllm-gpu-loop/gpu-spool-adapter (lazy; see ./_load.cjs)
module.exports = require('./_load.cjs').loadGpuLoop('gpu-spool-adapter');
