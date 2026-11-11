import { mkdir, copyFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** Build a static-WASM Worker locally; this function never deploys anything. */
export async function buildWorker(outDirectory, entryPoint = fileURLToPath(new URL('./worker.mjs', import.meta.url))) {
  if (typeof outDirectory !== 'string' || outDirectory.length === 0) throw new TypeError('outDirectory is required');
  let build;
  try { ({ build } = await import('esbuild')); }
  catch (cause) { throw new Error('buildWorker requires esbuild; install it as a development dependency', { cause }); }
  const output = resolve(outDirectory), wasm = join(output, 'e-means.wasm');
  await mkdir(output, { recursive: true });
  await copyFile(fileURLToPath(import.meta.resolve('@ruvector/e-means/wasm')), wasm);
  const script = join(output, 'worker.mjs');
  await build({ entryPoints: [entryPoint], outfile: script, bundle: true, format: 'esm', platform: 'browser',
    target: 'es2022', legalComments: 'none', plugins: [{ name: 'e-means-static-wasm', setup(builder) {
      builder.onResolve({ filter: /^@ruvector\/e-means\/wasm$/ }, () => ({ path: './e-means.wasm', external: true }));
    } }] });
  return { script, wasm };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const artifact = await buildWorker(process.argv[2] ?? fileURLToPath(new URL('./dist/', import.meta.url)));
  console.log(JSON.stringify(artifact));
}
