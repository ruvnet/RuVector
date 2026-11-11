import assert from 'node:assert/strict';
import { constants } from 'node:fs';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { access, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const page = `<!doctype html><html><head><link rel="icon" href="data:,"></head><body data-result="pending"><script type="module">
import { init, fit, predict } from '/e-means.mjs';
try {
  await init();
  const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const result = fit(data, 2, { clusters: 1, epochs: 2, batchSize: 2, seed: 42 });
  const labels = predict(data, 2, result.centroids);
  if (!(result.centroids instanceof Float32Array) || !(labels instanceof Uint32Array) ||
      String(result.centroids) !== '5,6' || String(labels) !== '0,0,0,0,0' ||
      result.mse !== 16 || result.updates !== 6) throw new Error('Rust/WASM golden result differs');
  globalThis.eMeansBrowserOracle = { centroids: Array.from(result.centroids), labels: Array.from(labels),
    mse: result.mse, updates: result.updates, typed: result.centroids instanceof Float32Array && labels instanceof Uint32Array };
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
    const bytes = await readFile(new URL(route[0], import.meta.url));
    response.writeHead(200, { 'content-type': route[1] });
    response.end(bytes);
  } catch (error) {
    response.writeHead(500);
    response.end(error.message);
  }
});

let browser;
const diagnostics = [];
try {
  const tooling = process.env.E_MEANS_PLAYWRIGHT_ROOT;
  assert.ok(tooling, 'Set E_MEANS_PLAYWRIGHT_ROOT to the external pinned Playwright tooling directory');
  const require = createRequire(resolve(tooling, 'package.json'));
  assert.equal(require('playwright/package.json').version, '1.62.1', 'Browser qualification requires Playwright 1.62.1');
  const { chromium } = require('playwright');
  const executablePath = process.env.E_MEANS_CHROMIUM || chromium.executablePath();
  await access(executablePath, constants.X_OK);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const deadline = Date.now() + 30_000;
  const remaining = () => Math.max(1, deadline - Date.now());
  browser = await chromium.launch({ executablePath, headless: true, timeout: Math.min(25_000, remaining()),
    args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const browserPage = await browser.newPage();
  const responses = new Map();
  let rejectPage;
  const pageFailure = new Promise((_, reject) => { rejectPage = reject; });
  // An error may arrive between browser operations; keep it handled until the
  // next race observes it. It still rejects qualification, never marks success.
  pageFailure.catch(() => {});
  const fail = (message) => { diagnostics.push(message); rejectPage(new Error(message)); };
  browserPage.on('pageerror', (error) => fail(`Page error: ${error.message}`));
  browserPage.on('console', (message) => { if (message.type() === 'error') fail(`Browser console: ${message.text()}`); });
  browserPage.on('requestfailed', (request) => fail(`Browser request failed: ${request.url()} ${request.failure()?.errorText}`));
  browserPage.on('response', (response) => {
    const url = new URL(response.url());
    if (url.origin === origin) responses.set(url.pathname, response.status());
  });
  const response = await Promise.race([browserPage.goto(origin + '/', { waitUntil: 'load', timeout: remaining() }), pageFailure]);
  assert.equal(response.status(), 200, 'Browser page must load successfully');
  await Promise.race([browserPage.waitForFunction(() => ['pass', 'fail'].includes(document.body.dataset.result),
    undefined, { timeout: remaining() }), pageFailure]);
  const result = await browserPage.evaluate(() => ({ status: document.body.dataset.result,
    text: document.body.textContent, oracle: globalThis.eMeansBrowserOracle }));
  assert.equal(result.status, 'pass', result.text);
  assert.deepEqual(result.oracle, { centroids: [5, 6], labels: [0, 0, 0, 0, 0], mse: 16, updates: 6, typed: true });
  for (const path of routes.keys()) assert.equal(responses.get(path), 200, `Browser must fetch ${path} successfully`);
  assert.equal(diagnostics.length, 0, diagnostics.join('\n'));
} catch (error) {
  throw new Error(`Chromium browser consumer failed: ${error.message}\n${diagnostics.join('\n')}`, { cause: error });
} finally {
  await browser?.close();
  if (server.listening) {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}
console.log('Chromium browser consumer passed: default WASM fetch, fit, predict and numerical oracle');
