// Native KGE operational controls: fixed 50 epochs and support-timed calibration.
// One training call per model preserves optimizer state. No score-based tuning.
import {readFileSync, writeFileSync, realpathSync} from 'node:fs';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import assert from 'node:assert/strict';

const [input, bindingPath, output, nativePath] = process.argv.slice(2);
assert(input && bindingPath && output && nativePath);
const bytes = readFileSync(input), payload = JSON.parse(bytes);
const hash = x => createHash('sha256').update(x).digest('hex');
const require = createRequire(import.meta.url);
const binding = require(resolve(bindingPath));
assert.equal(binding.backend, 'native');
const loadedNative=Object.keys(require.cache).filter(x=>x.endsWith('.node')).map(x=>realpathSync(x));
assert.deepEqual(loadedNative,[realpathSync(nativePath)],'actual native module differs from frozen binary');
const parse = text => {const r=JSON.parse(text); assert(!r.error, JSON.stringify(r.error)); return r;};
const rows=payload.triples;
const supportCount=rows.filter(x=>x.split==='train').length;
const testCount=rows.filter(x=>x.split==='test').length;
assert(supportCount > 0 && testCount > 0);
const config = {dims:64,batch_size:256,lr:.1,optimizer:{kind:'adagrad',epsilon:1e-8},
  loss:{kind:'one_vs_all'},n3_lambda:.001};
function model(seed) {
  const m=new binding.Model(JSON.stringify({scorer:'hole',dims:64,seed}));
  parse(m.addTriplesJson(JSON.stringify(rows)));
  return m;
}
function fit(m,seed,epochs) {
  const start=performance.now();
  const telemetry=parse(m.trainJson(JSON.stringify({...config,seed,epochs})));
  const seconds=(performance.now()-start)/1000;
  assert.equal(telemetry.epochs,epochs);
  assert.equal(telemetry.batches,Math.ceil(supportCount/256));
  assert(Number.isFinite(telemetry.loss));
  return {seconds,epochs,examples:supportCount*epochs,telemetry};
}
// Calibrate cost on a discarded model; no valid/test score is ever requested.
const calibration=fit(model(919191),919191,1000);
const estimatedEpochs=Math.max(1,Math.min(20000,Math.floor(1.9/calibration.seconds*1000)));
const result={inputHash:hash(bytes),runnerHash:hash(readFileSync(new URL(import.meta.url))),
  engineVersion:binding.version(),backend:binding.backend,config,calibration,estimatedEpochs,arms:[]};
for(const seed of [101,202,303]) {
  for(const arm of ['kge50','kgeTimed']) {
    const m=model(seed);
    const initialHash=hash(m.toJson());
    const training=fit(m,seed,arm==='kge50'?50:estimatedEpochs);
    const checkpointHash=hash(m.toJson());
    assert.notEqual(initialHash,checkpointHash);
    const start=performance.now();
    const raw=parse(m.evalJson(JSON.stringify({split:'test',filtered:true,tieBreak:'bottom'})));
    assert.equal(raw.filtered,true);
    const metrics=raw.report?.combined;
    assert(metrics && Number.isFinite(metrics.mrr) && metrics.mrr > 0 && metrics.mrr <= 1);
    assert.equal(metrics.count,2*testCount);
    assert.equal(checkpointHash,hash(m.toJson()));
    result.arms.push({arm,seed,training:{...training,secondsBudget:2,
      withinTenPercent:Math.abs(training.seconds-2)/2 <= .1},
      initialHash,checkpointHash,evaluation:{metrics,seconds:(performance.now()-start)/1000}});
  }
}
writeFileSync(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({graph:payload.name,support:supportCount,kgeEpochs:estimatedEpochs}));
