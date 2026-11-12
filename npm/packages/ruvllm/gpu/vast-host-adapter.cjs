'use strict';
// @ruvector/ruvllm/vast-host-adapter -> @ruvector/ruvllm-gpu-loop/vast-host-adapter (lazy; see ./_load.cjs)
module.exports = require('./_load.cjs').loadGpuLoop('vast-host-adapter');
