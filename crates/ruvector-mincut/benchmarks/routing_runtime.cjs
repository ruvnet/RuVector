// Real complete API timings, including worker message/serialization overhead.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { performance } = require('node:perf_hooks');
const { pathToFileURL } = require('node:url');
function data(name) {
  const bytes = fs.readFileSync(path.join(process.env.ROADS_DIR, `${name}.roads`));
  let o=0;
  const u32=()=>{const v=bytes.readUInt32LE(o);o+=4;return v;};
  const n=u32(), m=u32(), q=u32();
  assert(n<=1_000_000 && m<=4_000_000 && q<=10000);
  const endpoints=new Uint32Array(m*2), costs=new Uint32Array(m);
  for(let i=0;i<m;i++){endpoints[i*2]=u32();endpoints[i*2+1]=u32();costs[i]=u32();}
  const coords=new Float64Array(n*2);
  for(let i=0;i<coords.length;i++){coords[i]=bytes.readDoubleLE(o);o+=8;}
  const k=u32();assert(k<=16);const landmarks=new Uint32Array(k);
  for(let i=0;i<k;i++)landmarks[i]=u32();
  const queries=[];
  for(let i=0;i<q;i++){const s=u32(),t=u32();const cost=Number(bytes.readBigInt64LE(o));o+=8;queries.push([s,t,cost]);}
  const traceCount=u32();assert(traceCount<=128);const traces=[];
  for(let i=0;i<traceCount;i++){const s=u32(),t=u32(),id=u32();const closed=Number(bytes.readBigInt64LE(o));o+=8;const original=Number(bytes.readBigInt64LE(o));o+=8;traces.push([s,t,id,closed,original]);}
  assert.equal(o,bytes.length);
  return {n,m,endpoints,costs,coords,landmarks,queries,traces};
}
function summary(values){const a=[...values].sort((a,b)=>a-b);return {medianMs:(a[(a.length-1)>>1]+a[a.length>>1])/2,p95Ms:a[Math.ceil(a.length*.95)-1],samples:a.length};}
async function main(){
  const kind=process.argv[2];assert(['native','wasm','worker'].includes(kind));
  const packageRoot=process.env.MINCUT_WASM_PACKAGE || path.resolve(__dirname,'../../..','npm/packages/mincut-wasm');
  let Constructor,Worker;
  if(kind==='native')Constructor=require(process.env.MINCUT_ADDON).RoadRouter;
  if(kind==='wasm')Constructor=require(path.join(packageRoot,'node/ruvector_mincut_wasm.js')).WasmRoadRouter;
  if(kind==='worker')({RoutingWorker:Worker}=await import(pathToFileURL(path.join(packageRoot,'routing/client.mjs')).href));
  for(const name of ['NY','BAY']){
    const d=data(name);let start=performance.now();
    const r=kind==='worker'?await Worker.create(d.n,d.endpoints,d.costs):new Constructor(d.n,d.endpoints,d.costs,new Uint32Array());
    const buildMs=performance.now()-start;
    const route=(s,t)=>kind==='worker'?r.route(s,t,{budget:20_000_000}):r.route(s,t,true,20_000_000);
    try {
      start=performance.now();await r.setCoordinates(d.coords);const mapMs=performance.now()-start;
      start=performance.now();if(kind==='worker')await r.prepare(d.landmarks,{budget:100_000_000});else r.prepare(d.landmarks,100_000_000);
      const prepareMs=performance.now()-start;
      const samples=[[],[]];
      for(let i=0;i<d.queries.length;i++){
        const [s,t,cost]=d.queries[i];
        for(let repeat=0;repeat<3;repeat++){
          start=performance.now();const result=await route(s,t);const ms=performance.now()-start;
          assert.equal(result?.cost??-1,cost);
          if(result){
            assert.equal(result.nodes[0],s);assert.equal(result.nodes.at(-1),t);
            let total=0;for(let j=0;j<result.arcs.length;j++){
              const id=result.arcs[j];assert.equal(d.endpoints[id*2],result.nodes[j]);assert.equal(d.endpoints[id*2+1],result.nodes[j+1]);total+=d.costs[id];
            }assert.equal(total,cost);
          }
          samples[i%2].push(ms);
        }
      }
      for(const [s,t,id,expected,original] of d.traces){
        if(kind==='worker')await r.prepare(d.landmarks,{budget:100_000_000});else r.prepare(d.landmarks,100_000_000);
        await r.update(new Uint32Array([id]),new Uint32Array([0xffffffff]));
        const closed=await route(s,t);
        assert.equal(closed?.cost??-1,expected);assert(!closed?.arcs.includes(id));
        await r.update(new Uint32Array([id]),new Uint32Array([d.costs[id]]));
        assert.equal((await route(s,t)).cost,original);
      }
      const snap=await r.nearest(d.coords[0],d.coords[1],1);
      assert(snap && snap.distanceM<=1);
      console.log(JSON.stringify({runtime:kind,dataset:name,nodes:d.n,arcs:d.m,buildMs,mapMs,prepareMs,closurePairs:d.traces.length,uniform:summary(samples[0]),local:summary(samples[1]),cumulativeProcessPeakRssKiB:process.resourceUsage().maxRSS}));
    }finally{if(kind==='worker')r.close();else if(kind==='wasm')r.free();else r.clear();}
  }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
