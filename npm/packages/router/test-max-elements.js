// #1099: `maxElements` is a capacity hint unless `enforceMaxElements` is set.
const assert = require('node:assert/strict');
const { VectorDb } = require('./index.js');

const vec = (x) => new Float32Array([x, 1 - x, 0.5, 0.25]);

// 1. Default: maxElements alone is only a hint (Martin's repro: 5 → 12 inserts).
{
  const db = new VectorDb({ dimensions: 4, maxElements: 5 });
  for (let i = 0; i < 12; i++) db.insert(`v${i}`, vec(i / 12));
  assert.equal(db.count(), 12, 'maxElements without enforceMaxElements must not cap inserts');
}

// 2. Enforced: the insert that would exceed the bound throws a typed error.
{
  const db = new VectorDb({ dimensions: 4, maxElements: 5, enforceMaxElements: true });
  for (let i = 0; i < 5; i++) db.insert(`v${i}`, vec(i / 5));
  assert.throws(() => db.insert('v5', vec(0.9)), (e) => {
    assert.match(e.message, /^ERR_CAPACITY_EXCEEDED: /);
    assert.match(e.message, /maxElements is 5/);
    return true;
  });
  assert.equal(db.count(), 5, 'a rejected insert must not be stored');

  // Replacing an existing id at capacity is allowed.
  db.insert('v2', vec(0.33));
  assert.equal(db.count(), 5);

  // Deleting frees a slot.
  assert.equal(db.delete('v0'), true);
  db.insert('v5', vec(0.9));
  assert.equal(db.count(), 5);
}

// 3. enforceMaxElements without maxElements is a configuration error, not a silent default.
assert.throws(
  () => new VectorDb({ dimensions: 4, enforceMaxElements: true }),
  /enforceMaxElements requires maxElements/,
);

console.log('✓ maxElements / enforceMaxElements semantics (#1099)');
