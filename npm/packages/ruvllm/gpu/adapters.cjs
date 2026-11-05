'use strict';
// @ruvector/ruvllm/adapters -> @ruvector/ruvllm-gpu-loop/adapters (lazy; see ./_load.cjs)
module.exports = require('./_load.cjs').loadGpuLoop('adapters');
