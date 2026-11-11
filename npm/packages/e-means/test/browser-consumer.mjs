import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const page = `<!doctype html><html><body data-result="pending"><script type="module">
import { init, fit, predict } from '/e-means.mjs';
try {
  await init();
  const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const result = fit(data, 2, { clusters: 1, epochs: 2, batchSize: 2, seed: 42 });
  const labels = predict(data, 2, result.centroids);
  if (!(result.centroids instanceof Float32Array) || !(labels instanceof Uint32Array) ||
      String(result.centroids) !== '5,6' || String(labels) !== '0,0,0,0,0' ||
      result.mse !== 16 || result.updates !== 6) throw new Error('Rust/WASM golden result differs');
  document.body.dataset.result = 'pass';
  document.body.textContent = 'Browser loaded bundled WASM and passed the shared numerical oracle';
} catch (error) {
  document.body.dataset.result = 'fail';
  document.body.textContent = error.stack || error.message;
}
</script></body></html>`;

const routes = new Map([
  ['/e-means.mjs', ['../e-means.mjs', 'application/javascript']],
  ['/runtime.mjs', ['../runtime.mjs', 'application/javascript']],
  ['/wasm/ruvector_e_means.js', ['../wasm/ruvector_e_means.js', 'application/javascript']],
  ['/wasm/ruvector_e_means_bg.wasm', ['../wasm/ruvector_e_means_bg.wasm', 'application/wasm']],
]);
let fetchedWasm = false;
const server = createServer(async (request, response) => {
  try {
    if (request.url === '/') {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.end(page);
      return;
    }
    const route = routes.get(request.url);
    if (!route) {
      response.writeHead(404);
      response.end();
      return;
    }
    if (request.url.endsWith('.wasm')) fetchedWasm = true;
    const bytes = await readFile(new URL(route[0], import.meta.url));
    response.writeHead(200, { 'content-type': route[1] });
    response.end(bytes);
  } catch (error) {
    response.writeHead(500);
    response.end(error.message);
  }
});

const profile = await mkdtemp(join(tmpdir(), 'e-means-chromium-'));
let browser;
function stopBrowser() {
  if (!browser?.pid) return;
  try { process.kill(-browser.pid, 'SIGKILL'); } catch (error) {
    if (error.code !== 'ESRCH') throw error;
  }
}
try {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const chromium = process.env.E_MEANS_CHROMIUM || 'chromium';
  const url = `http://127.0.0.1:${server.address().port}/`;
  const result = await new Promise((resolve, reject) => {
    const child = browser = spawn(chromium, ['--headless', '--no-sandbox', '--disable-dev-shm-usage',
      '--disable-gpu', '--disable-background-networking', '--no-first-run',
      `--user-data-dir=${profile}`, '--dump-dom', '--virtual-time-budget=10000', url],
    { stdio: ['ignore', 'pipe', 'pipe'], detached: true });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    const timer = setTimeout(() => {
      stopBrowser();
      reject(new Error(`Chromium browser consumer exceeded 25 seconds\n${stdout}\n${stderr.slice(-2000)}`));
    }, 25_000);
    child.once('error', (error) => { clearTimeout(timer); reject(error); });
    child.once('close', (status) => { clearTimeout(timer); resolve({ status, stdout, stderr }); });
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /data-result="pass"/, result.stdout + '\n' + result.stderr.slice(-1500));
  assert.ok(fetchedWasm, 'browser must fetch the bundled .wasm through default init()');
  console.log('Chromium browser consumer passed: default WASM fetch, fit, predict and numerical oracle');
} finally {
  stopBrowser();
  await new Promise((resolve) => server.close(resolve));
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
