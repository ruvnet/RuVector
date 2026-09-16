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

test('native RuField awareness updates exact routes and expires', () => {
  const { RuFieldRoadRouter } = require(process.env.MINCUT_ADDON);
  const event=(id,timestamp)=>JSON.stringify({event_id:id,timestamp_ns:timestamp,observation:{zone_id:'room-a',space_cell:null,confidence:1,features:{presence:1,motion_energy:1},privacy_class:'P2'},provenance:{synthetic:false}});
  const r=new RuFieldRoadRouter(4,U([0,1,1,3,0,2,2,3]),U([10,10,30,30]),U([]),{maxPenalty:100,closeAtMillionths:900000,ttlNs:100,maxLatenessNs:10});
  try {
    r.bindZone('room-a',1);assert.deepEqual(r.route(0,3,false,1000).nodes,[0,1,3]);
    assert.throws(()=>r.ingestRufield(event('bad',100),false,100),/unverified/);
    const update=r.ingestRufield(event('e1',100),true,100);assert.equal(update.changedArcs,2);
    assert.deepEqual(r.route(0,3,false,1000).nodes,[0,2,3]);assert.equal(r.expire(200),2);
    assert.equal(r.route(0,3,false,1000).cost,20);
  } finally {r.clear();}
});
