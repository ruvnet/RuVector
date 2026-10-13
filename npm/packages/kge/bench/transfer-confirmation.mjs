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
const priorSourceRelations = new Set(rels.filter((_,i)=>i%2===0));
const priorSource = ordered(ds.splits.train.filter(t=>priorSourceRelations.has(t.r))).slice(0,1024);
assert.equal(priorSource.length,1024,'insufficient priorSource sample');
const priorSourceEntities = new Set(priorSource.flatMap(t=>[t.s,t.o]));
const priorTarget = ordered(ds.splits.train.filter(t=>!priorSourceRelations.has(t.r)&&priorSourceEntities.has(t.s)&&priorSourceEntities.has(t.o))).slice(0,1024);
assert.equal(priorTarget.length,1024,'insufficient mapped priorTarget sample');
const priorTargetRelations = new Set(priorTarget.map(t=>t.r));
const priorEntities = new Set([...priorSource,...priorTarget].flatMap(t=>[t.s,t.o]));
const priorRelations = new Set([...priorSource,...priorTarget].map(t=>t.r));
const priorEligible = t => priorEntities.has(t.s)&&priorEntities.has(t.o)&&priorRelations.has(t.r);
const priorTest = ds.splits.test.filter(t=>priorEligible(t)&&priorTargetRelations.has(t.r));

assert.equal(priorTest.length,159);
assert.equal(hash(priorTest),'e5585af2a06f9ca1d8f6dc0fd89c0f786c42474616594dad6e29b44a74ecfce3');
const priorKeys=new Set(priorTest.map(key));
const tailKey=t=>JSON.stringify([t.s,t.r]);
const headKey=t=>JSON.stringify([t.r,t.o]);
const priorTail=new Set(priorTest.map(tailKey));
const priorHead=new Set(priorTest.map(headKey));
const freshRels=[...rels].sort((a,b)=>hash('confirm-20261013-v1:'+a).localeCompare(hash('confirm-20261013-v1:'+b)));
const sourceRelations = new Set(freshRels.filter((_,i)=>i%2===0));
const source = ordered(ds.splits.train.filter(t=>sourceRelations.has(t.r))).slice(0,1024);
assert.equal(source.length,1024,'insufficient source sample');
const sourceEntities = new Set(source.flatMap(t=>[t.s,t.o]));
const target = ordered(ds.splits.train.filter(t=>!sourceRelations.has(t.r)&&sourceEntities.has(t.s)&&sourceEntities.has(t.o))).slice(0,1024);
assert.equal(target.length,1024,'insufficient mapped target sample');
const targetRelations = new Set(target.map(t=>t.r));
const entities = new Set([...source,...target].flatMap(t=>[t.s,t.o]));
const relations = new Set([...source,...target].map(t=>t.r));
const eligible = t => entities.has(t.s)&&entities.has(t.o)&&relations.has(t.r);
const test = ds.splits.test.filter(t=>eligible(t)&&targetRelations.has(t.r)&&!priorKeys.has(key(t))&&!priorTail.has(tailKey(t))&&!priorHead.has(headKey(t)));
assert(test.length>=20,'insufficient target test queries');
const trainKeys = new Set(ds.splits.train.map(key));
assert(test.every(t=>!trainKeys.has(key(t))),'train/test overlap');
const canonical = [...ds.splits.train,...ds.splits.valid,...ds.splits.transfer,...ds.splits.test];
const testKeys = new Set(test.map(key));
const filterTruth = [...new Map(canonical.filter(t=>eligible(t)&&!testKeys.has(key(t))).map(t=>[key(t),t])).values()];
const seeds=[101,102,103,104,105,106,107,108,109,110];
const config={scorer:'hole',dims:32,batch_size:128,lr:0.05,n3_lambda:0.001,source_epochs:8,adapt_epochs:4,seeds};
const manifest={
  schema:'kge-replay-confirmation/1',partitionSalt:'confirm-20261013-v1',priorExcludedTestHash:hash(priorTest),hypothesis:'source pretraining improves target relation MRR against matched adaptation and matched positive-example exposure controls',
  scope:'exploratory relation-subset pretraining with mixed replay inside one FB15k237 subgraph; not cross-domain transfer',
  acceptance:'mean paired MRR gain >0 and lower 95% seed-bootstrap bound >0; 20000 resamples, RNG seed20261013; otherwise NOT_CONFIRMED',
  sourceHead:'e4e8a5ca9da1aa6b1ff3331902db9fd87abb090b',runner_sha256:hash(readFileSync(new URL(import.meta.url))),
  binding:{path:bindingPath,backend:binding.backend,version:binding.version()},
  dataset:ds.fileHashes,config,
  counts:{canonicalSubgraph:ds.counts,source:source.length,target:target.length,test:test.length,excludedTest:ds.splits.test.length-test.length,entities:entities.size,relations:relations.size,targetRelations:targetRelations.size,filterTruth:filterTruth.length},
  sampleHashes:{source:hash(source),target:hash(target),test:hash(test),filterTruth:hash(filterTruth)},
  budgets:{warmSource:12288,warmTarget:4096,equalSource:8192,equalTarget:8192,coldSource:4096,coldTarget:4096,warmUpdates:128,equalUpdates:128,coldUpdates:64,warmExamples:16384,coldExamples:8192,equalExposureExamples:16384,perProcessTimeoutSeconds:60},
  controls:'all arms clone identical untrained checkpoint; same vocabulary; target facts registered as valid before source phase; source trainer consumes only train tags; all evaluation facts registered after every checkpoint freezes; equal-exposure control uses two optimizer-reset phases',
  createdAt:new Date().toISOString()
};
const phase=process.env.KGE_CONFIRM_PHASE;
assert(['freeze','run'].includes(phase),'explicit freeze or run required');
if(phase==='freeze'){
  assert(!existsSync(out+'/manifest.json'),'refuse to replace preregistration');
  writeFileSync(out+'/manifest.json',JSON.stringify(manifest,null,2));
  console.log(JSON.stringify({phase,manifestHash:hash(manifest),counts:manifest.counts}));
  process.exit(0);
}
const frozen=JSON.parse(readFileSync(out+'/manifest.json','utf8'));
assert.deepEqual(manifest.sampleHashes,frozen.sampleHashes);
assert.deepEqual(manifest.config,frozen.config);
assert.equal(manifest.runner_sha256,frozen.runner_sha256);
assert.equal(manifest.acceptance,frozen.acceptance);
assert(!existsSync(out+'/frozen-checkpoints.json'),'refuse to retrain frozen confirmation');
assert(!existsSync(out+'/results.json'),'refuse to overwrite result');
Object.assign(manifest,frozen);
const records=[];
const tag=(xs,split)=>xs.map(t=>({...t,split}));
for(const seed of seeds){
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
      assert.equal(result.batches,epochs===8?8:16,'unexpected optimization batches');
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
// All thirty trained checkpoints exist before any final-test evaluation.
writeFileSync(out+'/frozen-checkpoints.json',JSON.stringify(records,null,2));
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

assert.equal(records.length,30);
assert.equal(paired.length,10);
const differences=paired.map(x=>x.vsEqual);
const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
let rng=20261013;
const random=()=>{rng=(Math.imul(rng,1664525)+1013904223)>>>0;return rng/4294967296;};
const boot=Array.from({length:20000},()=>mean(differences.map(()=>differences[Math.floor(random()*differences.length)]))).sort((a,b)=>a-b);
const summary={warmMean:mean(paired.map(x=>x.warm)),coldMean:mean(paired.map(x=>x.cold)),equalMean:mean(paired.map(x=>x.equalExposure)),meanDelta:mean(differences),ci95:[boot[499],boot[19499]],positiveSeeds:differences.filter(x=>x>0).length,unit:'training seed, conditional on this selected query set'};
const result={manifestHash:hash(manifest),records,paired,summary,decision:summary.meanDelta>0&&summary.ci95[0]>0?'CONFIRMED_ON_THIS_PARTITION':'NOT_CONFIRMED',limits:manifest.scope};
writeFileSync(out+'/results.json',JSON.stringify(result,null,2));
console.log(JSON.stringify({counts:manifest.counts,paired,summary,decision:result.decision}));
