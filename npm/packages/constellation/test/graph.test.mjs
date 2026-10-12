import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, discoverPackages, normalizePackage, sweep, planImprovements, compareReports } from '../src/index.mjs';
const pack = (name, extra={}) => ({name,'dist-tags':{latest:'1.0.0'},maintainers:[{name:'ruvnet'}],versions:{'1.0.0':{name,version:'1.0.0',license:'MIT',repository:'https://github.com/ruvnet/example',dist:{integrity:'sha512-example'},...extra}}});
test('runtime SCCs include optional installs but exclude peer-only backedges',()=>{
  const g=buildGraph([{name:'a',dependencies:{b:'1'}},{name:'b',peerDependencies:{a:'1'},optionalDependencies:{c:'1'}},{name:'c',dependencies:{b:'1'}}]);
  assert.deepEqual(g.runtimeCycles,[['b','c']]); assert.equal(g.edges.length,4);
});
test('pagination detects missing names instead of calling truncated discovery complete',async()=>{
  let calls=0;
  const r=await discoverPackages('ruvnet',async()=>++calls===1?{total:251,objects:Array.from({length:250},(_,i)=>({package:{name:`p${i}`}}))}:{total:251,objects:[{package:{name:'p0'}}]});
  assert.equal(calls,2); assert.equal(r.complete,false); assert.equal(r.uniqueCount,250);
});
test('normalization preserves ownership uncertainty, lifecycle risk and floating deps',()=>{
  const input=pack('a',{dependencies:{b:'latest'},scripts:{postinstall:'node install.js'}}); input.maintainers=[];
  const p=normalizePackage(input);
  assert(p.findings.includes('maintainer-unverified'));assert(p.findings.includes('install-script-review'));assert(p.findings.includes('dependencies:unbounded:b'));
});
test('failed metadata fetch remains an incomplete report; audit is not transitive',async()=>{
  const r=await sweep({request:async(url)=>{
    if(url.includes('/search?'))return {total:2,objects:[{package:{name:'a'}},{package:{name:'b'}}]};
    if(url.endsWith('/a'))return pack('a');
    if(url.endsWith('/bulk'))return {};
    throw new Error('unavailable');
  }});
  assert.equal(r.complete,false);assert.equal(r.coverage.errors.length,1);assert.equal(r.coverage.transitiveAudit,false);
});
test('audit outage does not certify package security',async()=>{
  const r=await sweep({request:async(url)=>{
    if(url.includes('/search?')) return {total:1,objects:[{package:{name:'a'}}]};
    if(url.endsWith('/a'))return pack('a');throw new Error('audit unavailable');
  }}); assert.equal(r.advisoryStatus,'unknown');assert.equal(r.complete,false);
});
test('advisory priority and prior-run deltas are explicit without synthetic learning labels',()=>{
  const old={packages:[{name:'a',version:'1',findings:['missing-license']}]};
  const report={packages:[{name:'a',version:'2',findings:[]}],graph:{edges:[]},advisories:{a:[{severity:'critical',id:42}]}};
  const plan=planImprovements(report,old);assert.equal(plan.queue[0].score,100);assert.equal(plan.learnedModel,false);assert.equal(plan.queue[0].publishAuthority,false);
  assert.deepEqual(compareReports(report,old).changed,[{name:'a',before:'1',after:'2'}]);assert.equal(plan.delta.resolvedFindings.length,1);
});
