import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { mkdir } from 'node:fs/promises';

const root = fileURLToPath(new URL('../../../../', import.meta.url));
const manifest = fileURLToPath(new URL('../../../../crates/ruvector-e-means/Cargo.toml', import.meta.url));
const target = fileURLToPath(new URL('../../../../target/e-means-wasm/', import.meta.url));
const output = fileURLToPath(new URL('../wasm/', import.meta.url));

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited ${result.status}`);
}

const version = spawnSync('wasm-bindgen', ['--version'], { encoding: 'utf8' });
if (version.error || version.status !== 0 || version.stdout.trim() !== 'wasm-bindgen 0.2.125') {
  throw new Error('Install wasm-bindgen-cli@0.2.125 and the wasm32-unknown-unknown Rust target before building');
}
await mkdir(output, { recursive: true });
run('cargo', ['build', '--locked', '--manifest-path', manifest, '--release', '--target', 'wasm32-unknown-unknown', '--features', 'wasm', '--target-dir', target]);
run('wasm-bindgen', [`${target}/wasm32-unknown-unknown/release/ruvector_e_means.wasm`, '--target', 'web', '--out-dir', output, '--out-name', 'ruvector_e_means']);
run(process.execPath, [fileURLToPath(new URL('./check-artifact.mjs', import.meta.url))]);
