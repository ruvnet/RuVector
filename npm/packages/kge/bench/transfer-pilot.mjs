// Exploratory source pretraining with replay, not cross-domain transfer.
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
import { load } from './datasets/fb15k237.mjs';
const require = createRequire(import.meta.url);
const bindingPath = process.env.KGE_BENCH_BINDING;
assert(bindingPath, 'explicit verified binding required');
const binding = require(bindingPath);
assert.equal(binding.backend, 'native');
const out = resolve(process.env.KGE_PILOT_OUT || 'pilot-output');
mkdirSync(out, { recursive: true });
const hash = x => createHash('sha256').update(typeof x === 'string' || Buffer.isBuffer(x) ? x : JSON.stringify(x)).digest('hex');
const key = t => JSON.stringify([t.s, t.r, t.o]);
const ordered = xs => [...xs].sort((a,b) => hash(key(a)).localeCompare(hash(key(b))));
const call = (m, method, arg) => {
  const r = JSON.parse(m[method](JSON.stringify(arg)));
  assert(!r.error, JSON.stringify(r.error));
  return r;
};
const finite = x => {
  if (typeof x === 'number') assert(Number.isFinite(x), 'nonfinite result');
  else if (x && typeof x === 'object') Object.values(x).forEach(finite);
};
const parameterHash = m => {
  const x = JSON.parse(m.toJson()).model;
  return hash({ tables:x.tables, entities:x.entities, relations:x.relations });
};
const ds = await load({ limit:500 });
const freq = new Map();
ds.splits.train.forEach(t => freq.set(t.r, (freq.get(t.r)||0)+1));
const rels = [...freq].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).map(x=>x[0]);
const sourceRelations = new Set(rels.filter((_,i)=>i%2===0));
const source = ordered(ds.splits.train.filter(t=>sourceRelations.has(t.r))).slice(0,1024);
assert.equal(source.length,1024,'insufficient source sample');
const sourceEntities = new Set(source.flatMap(t=>[t.s,t.o]));
const target = ordered(ds.splits.train.filter(t=>!sourceRelations.has(t.r)&&sourceEntities.has(t.s)&&sourceEntities.has(t.o))).slice(0,1024);
assert.equal(target.length,1024,'insufficient mapped target sample');
const targetRelations = new Set(target.map(t=>t.r));
const entities = new Set([...source,...target].flatMap(t=>[t.s,t.o]));
const relations = new Set([...source,...target].map(t=>t.r));
const eligible = t => entities.has(t.s)&&entities.has(t.o)&&relations.has(t.r);
const test = ds.splits.test.filter(t=>eligible(t)&&targetRelations.has(t.r));
assert(test.length>=20,'insufficient target test queries');
const trainKeys = new Set([...source,...target].map(key));
assert(test.every(t=>!trainKeys.has(key(t))),'train/test overlap');
const canonical = [...ds.splits.train,...ds.splits.valid,...ds.splits.transfer,...ds.splits.test];
const testKeys = new Set(test.map(key));
const filterTruth = [...new Map(canonical.filter(t=>eligible(t)&&!testKeys.has(key(t))).map(t=>[key(t),t])).values()];
const seeds=[11,23,47];
const config={scorer:'hole',dims:32,batch_size:128,lr:0.05,n3_lambda:0.001,source_epochs:8,adapt_epochs:4,seeds};
const manifest={
  schema:'kge-replay-pilot/1',hypothesis:'source pretraining improves target relation MRR against matched adaptation and matched positive-example exposure controls',
  scope:'exploratory relation-subset pretraining with mixed replay inside one FB15k237 subgraph; not cross-domain transfer',
  acceptance:'all seeds improve target MRR against equal-exposure control; otherwise no retained gain',
  sourceHead:'18889bbddabc1ed28c95766fa32038dfdae76eb3',runner_sha256:hash(readFileSync(new URL(import.meta.url))),
  binding:{path:bindingPath,backend:binding.backend,version:binding.version()},
  dataset:ds.fileHashes,config,
  counts:{canonicalSubgraph:ds.counts,source:source.length,target:target.length,test:test.length,excludedTest:ds.splits.test.length-test.length,entities:entities.size,relations:relations.size,targetRelations:targetRelations.size,filterTruth:filterTruth.length},
  sampleHashes:{source:hash(source),target:hash(target),test:hash(test),filterTruth:hash(filterTruth)},
  budgets:{warmExamples:16384,coldExamples:8192,equalExposureExamples:16384,perProcessTimeoutSeconds:60},
  controls:'all arms clone identical untrained checkpoint; same vocabulary; target facts registered as valid before source phase; source trainer consumes only train tags; all evaluation facts registered after every checkpoint freezes; equal-exposure control uses two optimizer-reset phases',
  createdAt:new Date().toISOString()
};
const resume=process.env.KGE_PILOT_RESUME==='1';
if(resume){
  const frozen=JSON.parse(readFileSync(out+'/manifest.json','utf8'));
  assert.deepEqual(manifest.sampleHashes,frozen.sampleHashes);
  assert.deepEqual(manifest.config,frozen.config);
  Object.assign(manifest,frozen);
}
if(!resume){
assert(!existsSync(out+'/manifest.json'),'refuse to overwrite a frozen pilot');
writeFileSync(out+'/manifest.json',JSON.stringify(manifest,null,2));
}
const records=resume?JSON.parse(readFileSync(out+'/frozen-checkpoints.json','utf8')):[];
const tag=(xs,split)=>xs.map(t=>({...t,split}));
if(!resume)for(const seed of seeds){
  const initial=new binding.Model(JSON.stringify({scorer:'hole',dims:32,seed}));
  call(initial,'addTriplesJson',tag(source,'train'));
  call(initial,'addTriplesJson',tag(target,'valid'));
  call(initial,'predictJson',{s:source[0].s,r:source[0].r,k:1});
  const initialJson=initial.toJson();
  const initialHash=parameterHash(initial);
  for(const arm of ['warm','cold','equal_exposure']){
    let m=binding.Model.fromJson(initialJson);
    assert.equal(parameterHash(m),initialHash);
    const phases=[];
    const fit=epochs=>{
      const before=performance.now();
      const result=call(m,'trainJson',{epochs,batch_size:128,lr:0.05,n3_lambda:0.001,seed});
      finite(result);assert.equal(result.epochs,epochs);assert(result.batches>0);
      phases.push({epochs,milliseconds:performance.now()-before,result});
    };
    if(arm==='warm'){
      fit(8);
      const checkpoint=m.toJson();
      writeFileSync(out+'/source-'+seed+'.json',checkpoint);
      m=binding.Model.fromJson(checkpoint);
    }
    call(m,'addTriplesJson',tag(target,'train'));
    if(arm==='equal_exposure')fit(4);
    fit(4);
    const checkpoint=m.toJson();
    writeFileSync(out+'/'+arm+'-'+seed+'.json',checkpoint);
    records.push({seed,arm,initialHash,checkpointHash:hash(checkpoint),phases});
  }
}
// All nine trained checkpoints exist before any final-test evaluation.
if(!resume)writeFileSync(out+'/frozen-checkpoints.json',JSON.stringify(records,null,2));
for(const row of records){
  const checkpoint=readFileSync(out+'/'+row.arm+'-'+row.seed+'.json','utf8');
  assert.equal(hash(checkpoint),row.checkpointHash);
  const m=binding.Model.fromJson(checkpoint);
  const before=parameterHash(m);
  call(m,'addTriplesJson',tag(filterTruth,'valid'));
  call(m,'addTriplesJson',tag(test,'test'));
  assert.equal(parameterHash(m),before,'evaluation registration changed vocabulary or weights');
  row.metrics=call(m,'evalJson',{split:'test',filtered:true,tieBreak:'random',seed:row.seed});
  finite(row.metrics);
  assert.equal(row.metrics.filtered,true);
  row.mrr=row.metrics.report.combined.mrr;
  assert(Number.isFinite(row.mrr)&&row.mrr>=0&&row.mrr<=1);
}
const paired=seeds.map(seed=>{
  const r=arm=>records.find(x=>x.seed===seed&&x.arm===arm).mrr;
  return{seed,warm:r('warm'),cold:r('cold'),equalExposure:r('equal_exposure'),vsCold:r('warm')-r('cold'),vsEqual:r('warm')-r('equal_exposure')};
});
const result={manifestHash:hash(manifest),records,paired,decision:paired.every(x=>x.vsEqual>0)?'EXPLORATORY_GAIN_REQUIRES_CONFIRMATION':'NO_CONSISTENT_GAIN',limits:manifest.scope};
writeFileSync(out+'/results.json',JSON.stringify(result,null,2));
console.log(JSON.stringify({counts:manifest.counts,paired,decision:result.decision}));
