const { test } = require('node:test');
const assert = require('node:assert/strict');
const { RoadRouter } = require(process.env.MINCUT_ADDON);
const U = a => new Uint32Array(a);
test('native directed routing rejects illegal turns and malformed inputs', () => {
  assert.throws(() => new RoadRouter(-1,U([]),U([]),U([])));
  const r = new RoadRouter(4,U([0,1,1,2,1,3,3,2]),U([1,1,2,2]),U([0,1]));
  try {
    r.prepare(U([0,2]),1000);
    assert.equal(r.route(0,2,true,1000).cost,5);
    assert.deepEqual(r.route(0,2,true,1000).arcs,[0,2,3]);
    assert.equal(r.route(2,0,true,1000),null);
    assert.throws(() => r.route(0.1,2,true,1000));
    assert.throws(() => r.route(0,2,true,0));
    r.update(U([2]),U([0xffffffff]));
    assert.equal(r.route(0,2,true,1000),null);
    r.update(U([2]),U([0]));
    assert.equal(r.route(0,2,true,1000).cost,3);
    assert.throws(() => r.update(U([2,999]),U([100,100])));
    assert.equal(r.route(0,2,true,1000).cost,3);
    r.setCoordinates(new Float64Array([0,179.9,0,-179.8,90,0,-90,0]));
    assert.equal(r.nearest(0,-180,50000).node,0);
    assert.throws(() => r.nearest(91,0,10));
  } finally { r.clear(); }
  assert.throws(() => r.route(0,2,true,1000), /cleared/);
});
