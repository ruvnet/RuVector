// Dedicated worker: no eval, arbitrary module URL, filesystem path or method dispatch.
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
let send, listen, bindings;
if (isNode) {
  const { parentPort } = await import('node:worker_threads');
  if (!parentPort) throw new Error('Must run in a worker');
  send = value => parentPort.postMessage(value);
  listen = fn => parentPort.on('message', fn);
  bindings = (await import('../node/ruvector_mincut_wasm.js')).default;
} else {
  send = value => globalThis.postMessage(value);
  listen = fn => globalThis.addEventListener('message', event => fn(event.data));
  bindings = await import('../web/ruvector_mincut_wasm.js');
  await bindings.default();
}
let router;
listen(({ id, operation, args }) => {
  try {
    let result;
    if (operation === 'create') {
      if (router) throw new Error('Router already initialized');
      router = new bindings.WasmRoadRouter(...args);
    } else {
      if (!router) throw new Error('Router not initialized');
      switch (operation) {
        case 'prepare': result = router.prepare(...args); break;
        case 'route': result = router.route(...args); break;
        case 'update': result = router.update(...args); break;
        case 'setCoordinates': result = router.setCoordinates(...args); break;
        case 'nearest': result = router.nearest(...args); break;
        default: throw new Error('Unknown operation');
      }
    }
    send({ id, result });
  } catch (error) {
    send({ id, error: String(error.message ?? error) });
  }
});
