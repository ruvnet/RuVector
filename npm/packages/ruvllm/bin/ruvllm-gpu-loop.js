#!/usr/bin/env node
'use strict';
// Compatibility bin: `ruvllm-gpu-loop` shipped with @ruvector/ruvllm 4.0.x and now lives in
// @ruvector/ruvllm-gpu-loop (installed as an optional dependency).
require('./gpu-delegate.js').delegate('ruvllm-gpu-loop', process.argv.slice(2));
