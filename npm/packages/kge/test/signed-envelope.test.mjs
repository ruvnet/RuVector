// Signed model envelopes: save({ key }) / loadKge(json, { key }).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pkgDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const built = existsSync(join(pkgDir, 'dist', 'index.js')) && existsSync(join(pkgDir, 'native', 'kge.linux-x64-gnu.node'));
const KEY = 'a-test-key-of-at-least-16-bytes';

test('signed save verifies with the key and rejects edits, wrong keys and unsigned models', { skip: !built && 'native addon or dist not built' }, () => {
  const { createKge, loadKge, KgeError } = require(join(pkgDir, 'dist', 'index.js'));
  const kge = createKge({ dims: 8, seed: 1 });
  kge.addTriples([{ s: 'Ada', r: 'bornIn', o: 'London' }, { s: 'London', r: 'locatedIn', o: 'England' }]);

  const signed = kge.save({ key: KEY });
  assert.match(signed, /"hmac_sha256":"[0-9a-f]{64}"/);
  assert.equal(loadKge(signed, { key: KEY }).stats().entities, 3);
  assert.equal(loadKge(signed).stats().entities, 3, 'still loads without a key');

  const expectInvalid = (fn, why) => assert.throws(fn, (e) => e instanceof KgeError && e.kind === 'invalid', why);
  expectInvalid(() => loadKge(signed, { key: 'another-key-of-16-bytes' }), 'wrong key');
  expectInvalid(() => loadKge(kge.save(), { key: KEY }), 'unsigned model');
  expectInvalid(() => loadKge(signed.replace('"Ada"', '"Eve"'), { key: KEY }), 'edited payload');
  expectInvalid(() => kge.save({ key: 'short' }), 'key too short');
  // A key option that is present but unset (e.g. a missing env var) fails closed.
  expectInvalid(() => loadKge(kge.save(), { key: undefined }), 'undefined key on load');
  expectInvalid(() => kge.save({ key: undefined }), 'undefined key on save');
});
