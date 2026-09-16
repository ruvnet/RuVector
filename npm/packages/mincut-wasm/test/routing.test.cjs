const { test } = require('node:test');
const assert = require('node:assert/strict');
const api = require('@ruvector/mincut-wasm');
const U = values => new Uint32Array(values);

test('WASM directed routes, restrictions, updates and geographic lookup', () => {
  const router = new api.WasmRoadRouter(4, U([0,1,1,2,1,3,3,2]), U([1,1,2,2]), U([0,1]));
  try {
    router.prepare(U([0,2]), 1000);
    assert.deepEqual(router.route(0,2,true,1000).arcs, [0,2,3]);
    assert.equal(router.route(0,2,true,1000).cost, 5);
    assert.equal(router.route(2,0,true,1000), undefined);
    router.update(U([2]), U([0xffffffff]));
    assert.equal(router.route(0,2,true,1000), undefined);
    router.update(U([2]), U([0]));
    assert.equal(router.route(0,2,true,1000).cost, 3);
    assert.throws(() => router.update(U([0,9]), U([1,1])));
    assert.throws(() => router.route(-1,2,true,1000));
    assert.throws(() => router.route(0.5,2,true,1000));
    assert.throws(() => router.route(0,2,true,0));
    assert.throws(() => router.prepare(U([0]), Infinity));
    router.setCoordinates(new Float64Array([0,179.9,0,-179.8,90,0,-90,0]));
    assert.equal(router.nearest(0,-180,50000).node,0);
    assert.throws(() => router.nearest(NaN,0,10));
    assert.throws(() => router.setCoordinates(new Float64Array([0,0])));
  } finally { router.free(); }
});

test('worker routes, rejects bad input, and terminates on abort', async () => {
  const { RoutingWorker } = await import('@ruvector/mincut-wasm/routing');
  await assert.rejects(RoutingWorker.create(2,U([0,1]),U([1_000_000_001])), /cost/);
  const router = await RoutingWorker.create(3,U([0,1,1,2]),U([2,3]));
  try {
    await assert.rejects(router.route({ huge: [] }, 2), /arguments/);
    await assert.rejects(router.nearest(Infinity, 0, 1), /query/);
    await assert.rejects(router.update(new Uint32Array(new SharedArrayBuffer(4)), U([1])), /updates/);
    await assert.rejects(router.route(0, 2, { signal: {} }), /signal/);
    await router.prepare(U([0]), { budget:1000 });
    assert.equal((await router.route(0,2)).cost,5);
    await router.setCoordinates(new Float64Array([0,0,1,1,2,2]));
    assert.equal((await router.nearest(1,1,10)).node,1);
    await router.update(U([1]),U([0xffffffff]));
    assert.equal(await router.route(0,2),undefined);
    await assert.rejects(router.route(0,2,{timeoutMs:Infinity}), /timeout/);
    const controller = new AbortController();
    const pending = router.route(0,2,{signal:controller.signal});
    controller.abort();
    await assert.rejects(pending, /aborted/);
    await assert.rejects(router.route(0,2), /closed/);
  } finally { router.close(); }
});

test('worker queue bounds, timeout and postMessage errors release pending calls', async () => {
  const { RoutingWorker } = await import('@ruvector/mincut-wasm/routing');
  const fake = { on() {}, postMessage() {}, terminate() {} };
  let router = new RoutingWorker(fake);
  const queued = Array.from({length:32}, () => router.route(0,1).catch(e => e));
  await assert.rejects(router.route(0,1), /queue full/);
  router.close();
  assert.ok((await Promise.all(queued)).every(x => x instanceof Error));
  router = new RoutingWorker(fake);
  await assert.rejects(router.route(0,1,{timeoutMs:1}), /timed out/);
  router = new RoutingWorker({...fake, postMessage(){throw new Error('clone failure');}});
  await assert.rejects(router.route(0,1), /clone failure/);
  router.close();
});

test('RuField awareness replans, rejects unverified live events and expires risk', async () => {
  const event = (id, timestamp, privacy='P2', synthetic=false) => JSON.stringify({
    event_id:id,timestamp_ns:timestamp,
    observation:{zone_id:'room-a',space_cell:null,confidence:1,features:{presence:1,motion_energy:1,transient:0},privacy_class:privacy},
    provenance:{synthetic}
  });
  const router = new api.WasmRuFieldRouter(4,U([0,1,1,3,0,2,2,3]),U([10,10,30,30]),U([]),100,900000,100,10);
  try {
    router.bindZone('room-a',1);
    assert.deepEqual(router.route(0,3,false,1000).nodes,[0,1,3]);
    assert.throws(()=>router.ingestRuField(event('bad',100),false,100),/unverified/);
    const update=router.ingestRuField(event('e1',100),true,100);
    assert.equal(update.riskMillionths,1_000_000);assert.equal(update.changedArcs,2);
    assert.deepEqual(router.route(0,3,false,1000).nodes,[0,2,3]);
    assert.equal(router.ingestRuField(event('e1',100),true,100).duplicate,true);
    assert.equal(router.expire(200),2);
    assert.equal(router.route(0,3,false,1000).cost,20);
  } finally {router.free();}

  const { RoutingWorker } = await import('@ruvector/mincut-wasm/routing');
  const worker=await RoutingWorker.createRuField(4,U([0,1,1,3,0,2,2,3]),U([10,10,30,30]),U([]),{maxPenalty:100,closeAtMillionths:900000,ttlNs:100,maxLatenessNs:10});
  try {
    await worker.bindZone('room-a',1);
    await assert.rejects(worker.ingestRuField(JSON.parse(event('bad',100)),false,100),/unverified/);
    assert.equal((await worker.ingestRuField(JSON.parse(event('e2',100)),true,100)).changedArcs,2);
    assert.deepEqual((await worker.route(0,3,{landmarks:false,budget:1000})).nodes,[0,2,3]);
    assert.equal(await worker.expire(200),2);
  } finally {worker.close();}
});
