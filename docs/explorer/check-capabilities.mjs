import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// The extension is browser-facing JavaScript; use a data URL for Node package independence.
const source=await readFile(new URL('./capability-math.js',import.meta.url),'utf8');
const m=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const p=m.makeDataset();assert.deepEqual(p,m.makeDataset());assert.equal(p.length,360);
assert.equal(m.distance([0,0,0],[3,4,0]),5);assert.equal(m.nearest(p,p[7].v,1)[0].id,7);
assert(Math.abs(m.softmax([1000,1001,1002]).reduce((a,b)=>a+b,0)-1)<1e-12);
assert.equal(m.poincareDistance([.1,.2],[.1,.2]),0);assert.throws(()=>m.poincareDistance([1,0],[0,0]));
assert.equal(m.decay(24,24),.5);assert(m.quantizationMSE(p,2)>m.quantizationMSE(p,8));
assert(m.permitted(p,1).every(x=>x.c===0));assert.equal(m.permitted(p,15).length,360);
assert.equal(m.compact(p,64).length,64);assert(m.compact(p,64,'lfu').every(x=>x.hits>=m.compact(p,64,'lfu').at(-1).hits));
assert.deepEqual(m.hybridRank(p,[0,0,0],1).slice(0,10).map(x=>x.id),m.nearest(p,[0,0,0]).map(x=>x.id));
assert.equal(m.exactMinCut().weight,3);assert(m.typedDecision([0,0,0]).abstain);assert(!m.typedDecision([.8,0,0]).abstain);
console.log('PASS: deterministic data, exact retrieval, normalized attention, hyperbolic bounds, temporal decay, quantization, masks, compaction, hybrid ranking, exact cut, abstention');
