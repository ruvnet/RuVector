import { mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const vendor = join(root, 'src', 'vendor');
await mkdir(vendor, { recursive: true });
for (const [source, output] of [
  ['@ruvector/rvf-wasm/wasm', 'rvf.wasm'],
  ['@ruvector/mincut-wasm/wasm', 'mincut.wasm'],
]) {
  await copyFile(fileURLToPath(import.meta.resolve(source)), join(vendor, output));
}
