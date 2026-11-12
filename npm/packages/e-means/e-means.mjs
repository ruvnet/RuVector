import { init as initRuntime } from './runtime.mjs';
export { fit, predict } from './runtime.mjs';

let initialization;

/** Load the bundled Rust runtime once; pass bytes or a compiled module to override it. */
export async function init(input) {
  if (!initialization) {
    initialization = (async () => {
      let source = input;
      if (source === undefined) {
        const url = new URL('./wasm/ruvector_e_means_bg.wasm', import.meta.url);
        if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
          const { readFile } = await import('node:fs/promises');
          source = await readFile(url);
        } else {
          source = url;
        }
      }
      await initRuntime(source);
    })().catch((error) => {
      initialization = undefined;
      const message = typeof error?.message === 'string' ? error.message : String(error);
      if (message.startsWith('Could not initialize @ruvector/e-means:')) throw error;
      throw new Error(`Could not initialize @ruvector/e-means: ${message}`, { cause: error });
    });
  }
  await initialization;
}
