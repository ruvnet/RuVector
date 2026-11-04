'use strict';
// Run after `npm run build`: node --test test/*.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { SciPixClient } = require('../dist/index.js');

function captureFetch() {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ text: 'ok', latex: 'x', confidence: 1 }), { status: 200 });
  };
  return calls;
}

// Every base64-encoded JPEG starts with "/9j/" (0xFF 0xD8 0xFF).
const JPEG_B64 = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01]).toString('base64');

test('raw base64 JPEG string is sent as image data, not read as a file path', async () => {
  assert.ok(JPEG_B64.startsWith('/9j/'));
  const calls = captureFetch();
  const client = new SciPixClient({ maxRetries: 0 });
  await client.ocr(JPEG_B64);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].body.image, JPEG_B64);
  assert.equal(calls[0].body.imageType, 'jpeg');
});

test('absolute file paths are still read from disk', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'scipix-'));
  const file = path.join(dir, 'eq.png');
  const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  fs.writeFileSync(file, png);
  const calls = captureFetch();
  try {
    await new SciPixClient({ maxRetries: 0 }).ocr(file);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  assert.equal(calls[0].body.image, png.toString('base64'));
  assert.equal(calls[0].body.imageType, 'png');
});

test('plain base64 PNG string and data URL still work', async () => {
  const calls = captureFetch();
  const client = new SciPixClient({ maxRetries: 0 });
  const pngB64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).toString('base64');
  await client.ocr(pngB64);
  await client.ocr(`data:image/jpeg;base64,${JPEG_B64}`);
  assert.equal(calls[0].body.image, pngB64);
  assert.equal(calls[1].body.image, JPEG_B64);
  assert.equal(calls[1].body.imageType, 'jpeg');
});
