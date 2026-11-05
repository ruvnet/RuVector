// Preloaded by gpu-delegate.test.js: fail loudly if dist/cjs is required.
const Module = require('node:module');
const original = Module.prototype.require;
Module.prototype.require = function patched(id) {
  if (/dist[\\/]cjs[\\/]index\.js$/.test(id)) throw new Error('dist/cjs was loaded');
  return original.apply(this, arguments);
};
