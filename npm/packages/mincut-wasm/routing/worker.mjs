// Dedicated worker: no eval, arbitrary module URL, filesystem path or method dispatch.
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
let send, bindings, handler;
const waiting = [];
const receive = message => { if (handler) handler(message); else waiting.push(message); };
if (isNode) {
  const { parentPort } = await import('node:worker_threads');
  if (!parentPort) throw new Error('Must run in a worker');
  send = value => parentPort.postMessage(value);
  parentPort.on('message', receive);
  bindings = (await import('../node/ruvector_mincut_wasm.js')).default;
} else {
  send = value => globalThis.postMessage(value);
  globalThis.addEventListener('message', event => receive(event.data));
  bindings = await import('../web/ruvector_mincut_wasm.js');
  await bindings.default();
}
let router;
handler = ({ id, operation, args }) => {
  try {
    let result;
    if (operation === 'create' || operation === 'createRuField') {
      if (router) throw new Error('Router already initialized');
      router = operation === 'create' ? new bindings.WasmRoadRouter(...args) : new bindings.WasmRuFieldRouter(...args);
    } else {
      if (!router) throw new Error('Router not initialized');
      switch (operation) {
        case 'prepare': result = router.prepare(...args); break;
        case 'route': result = router.route(...args); break;
        case 'update': result = router.update(...args); break;
        case 'setCoordinates': result = router.setCoordinates(...args); break;
        case 'nearest': result = router.nearest(...args); break;
        case 'bindZone': result = router.bindZone(...args); break;
        case 'bindCell': result = router.bindCell(...args); break;
        case 'ingestRuField': result = router.ingestRuField(...args); break;
        case 'expire': result = router.expire(...args); break;
        default: throw new Error('Unknown operation');
      }
    }
    send({ id, result });
  } catch (error) {
    send({ id, error: String(error.message ?? error) });
  }
};
for (const message of waiting) handler(message);
waiting.length = 0;
