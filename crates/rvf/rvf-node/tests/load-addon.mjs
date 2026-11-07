// Locate the locally built addon: $RVF_NODE_ADDON, else the `.node` file
// `napi build --platform` leaves in the crate directory.
import { createRequire } from 'node:module';
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const crateDir = join(dirname(fileURLToPath(import.meta.url)), '..');

export function loadAddon() {
  if (process.env.RVF_NODE_ADDON) return require(process.env.RVF_NODE_ADDON);
  const built = readdirSync(crateDir).find((f) => f.endsWith('.node'));
  if (!built) {
    throw new Error('no addon: run `npx @napi-rs/cli build --platform --release` in crates/rvf/rvf-node');
  }
  return require(join(crateDir, built));
}

/** Deterministic vectors (LCG), flat Float32Array of n * dim. */
export function vectors(n, dim, seed = 1) {
  const out = new Float32Array(n * dim);
  let x = BigInt(seed) | 1n;
  for (let i = 0; i < out.length; i++) {
    x = (x * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    out[i] = Number(x >> 40n) / 2 ** 24 - 0.5;
  }
  return out;
}
