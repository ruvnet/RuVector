import assert from 'node:assert/strict';
import {evaluateLab,labScene,labPoints} from './experience-labs.js';
const q=[.2,.1,.3];
const low=evaluateLab('compress',{query:q,bits:2}),high=evaluateLab('compress',{query:q,bits:8});
assert.equal(low.bytes,270);assert.equal(high.bytes,1080);assert.equal(high.fullBytes,4320);
assert(high.mse<low.mse);assert(high.recall>=0&&high.recall<=1);
assert.deepEqual(high,evaluateLab('compress',{query:q,bits:8}));
for(const policy of ['lru','lfu']){
  const all=evaluateLab('compact',{strength:1,policy}),small=evaluateLab('compact',{strength:.01,policy});
  assert.equal(all.kept.length,360);assert.equal(all.recall,1);assert(small.kept.length<all.kept.length);
  assert(small.kept.every(p=>all.keptIds.has(p.id)));assert.equal(new Set(small.kept.map(p=>p.id)).size,small.capacity);
}
const permissive=evaluateLab('route',{strength:.01}),strict=evaluateLab('route',{strength:1});
assert.deepEqual(permissive.decision.scores,strict.decision.scores);
assert(permissive.classified.filter(p=>!p.decision.abstain).length>=strict.classified.filter(p=>!p.decision.abstain).length);
assert(Math.abs(strict.decision.scores.reduce((a,b)=>a+b,0)-1)<1e-10);
assert(evaluateLab('route',{query:[0,0,0],strength:.7}).decision.abstain);
for(const mode of ['compact','compress','route']){
  const r=evaluateLab(mode),svg=labScene(r,{selected:labPoints[0].id});assert(!/NaN|undefined/.test(svg));assert(svg.includes('data-rvcl-point'));
}
console.log('PASS: reproducible quantization, exact payload sizes, retention identity and capacity, threshold monotonicity, normalized scores and valid SVG');
