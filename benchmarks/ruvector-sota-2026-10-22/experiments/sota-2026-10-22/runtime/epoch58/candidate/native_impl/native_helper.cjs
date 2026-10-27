'use strict';
// E58: one source-pinned native corpus. No text, labels, model execution or network.
// Actual native order remains authoritative; exact cosine is diagnostic only.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {performance}=require('node:perf_hooks');
const KIND='REAL_NATIVE_TYPED_NLI_COVERAGE_ORDER_DEV';
const BINARY_SHA='ff559e9fac2e3d29a8b32e890fdb047bcd4bd28f389b4731ad72148b221d5a2e';
const KEYS=Object.freeze(['native_index_initializations_attempted','native_index_initializations_completed',
'native_insertions_attempted','native_insertions_completed','native_queries_attempted','native_queries_completed',
'native_len_calls_attempted','native_len_calls_completed','native_hit_source_validations',
'native_binary_loads_attempted','native_binary_loads_completed','native_version_calls_attempted',
'native_version_calls_completed','Node_network_operations_attempted']);
const need=(v,c)=>{if(!v)throw Error(c);};
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const own=(x,k)=>Object.prototype.hasOwnProperty.call(x,k);
const byID=(a,b)=>a<b?-1:a>b?1:0;
function geometry(c){
 need(c&&c.schema_version===1&&c.epoch===58&&c.kind===KIND&&['engineering','dev'].includes(c.phase),'NATIVE_PLAN_KIND');
 const n=c.corpus_document_count,q=c.query_count;
 need(Number.isInteger(n)&&((c.phase==='engineering'&&n===128&&q===16)||(c.phase==='dev'&&n>=1&&n<=24&&q===17)),'NATIVE_PLAN_GEOMETRY');
 need(typeof c.plan_sha256==='string'&&/^[a-f0-9]{64}$/.test(c.plan_sha256),'NATIVE_MASTER_PLAN_SHA');
 need(Number.isFinite(c.native_helper_max_seconds)&&c.native_helper_max_seconds>0&&c.native_helper_max_seconds<=60,'NATIVE_HELPER_CAP');
 return {n,q,k:n};
}
function limits(c){const {n,q}=geometry(c);return Object.fromEntries(KEYS.map((k,i)=>[k,[1,1,n,n,q,q,2,2,n*q,1,1,1,1,0][i]]));}
function ids(a,n,code){
 need(Array.isArray(a)&&a.length===n&&a.every(x=>typeof x==='string'&&x.length>0&&Buffer.byteLength(x)<=256&&!/[\u0000-\u001f\u007f]/.test(x))&&new Set(a).size===n,code);return a;
}
function decode(b,n){
 need(b&&Object.keys(b).sort().join(',')==='bytes,data_base64,dtype,encoding,sha256,shape'&&b.encoding==='base64'&&b.dtype==='float32le'&&Array.isArray(b.shape)&&b.shape.length===2&&b.shape[0]===n&&b.shape[1]===384&&b.bytes===n*1536&&typeof b.data_base64==='string','BUNDLE_F32_SCHEMA');
 const raw=Buffer.from(b.data_base64,'base64');need(raw.length===b.bytes&&raw.toString('base64')===b.data_base64&&hash(raw)===b.sha256,'BUNDLE_F32_IDENTITY');
 const out=[];for(let r=0;r<n;r++){const v=new Float32Array(384);let norm=0;
  for(let j=0;j<384;j++){v[j]=raw.readFloatLE(r*1536+j*4);need(Number.isFinite(v[j]),'F32_FINITE');norm+=v[j]*v[j];}
  need(Math.abs(Math.sqrt(norm)-1)<=1e-5,'F32_NORMALIZED');out.push(v);}return out;
}
function vectorBytes(v){
 need(v&&Object.keys(v).length===384&&Object.keys(v).every((k,i)=>k===String(i)),'NATIVE_VECTOR_SHAPE');
 const b=Buffer.alloc(1536);for(let j=0;j<384;j++){need(typeof v[j]==='number'&&Number.isFinite(v[j])&&Object.is(Math.fround(v[j]),v[j]),'NATIVE_VECTOR_F32');b.writeFloatLE(v[j],j*4);}return b;
}
function compact(h,vectors,index){
 need(h&&typeof h.id==='string'&&index.has(h.id),'NATIVE_KNOWN_SOURCE_ID');
 need(!own(h,'metadata'),'NATIVE_METADATA_ABSENT');
 const i=index.get(h.id),a=vectorBytes(h.vector),b=vectorBytes(vectors[i]);
 need(a.equals(b),'NATIVE_SOURCE_VECTOR_BITS');need(typeof h.score==='number'&&Number.isFinite(h.score),'NATIVE_SCORE_FINITE');
 return {id:h.id,score:h.score,vector_receipt:{source_vector_index:i,dimensions:384,dtype:'float32le',sha256:hash(a),source_equal_before_stripping:true},metadata_receipt:{present:false}};
}
function tiedOrder(hits){
 const out=[];for(let i=0;i<hits.length;){let j=i+1;while(j<hits.length&&hits[j].score===hits[i].score)j++;
 out.push(...hits.slice(i,j).sort((a,b)=>byID(a.id,b.id)));i=j;}return out;
}
function exactCosine(vectors,query,sourceIDs){
 let nq=0;for(const v of query)nq+=v*v;
 const rows=vectors.map((v,i)=>{let dot=0,nv=0;for(let j=0;j<384;j++){dot+=v[j]*query[j];nv+=v[j]*v[j];}
 return {id:sourceIDs[i],cosine:dot/Math.sqrt(nv*nq)};});
 need(rows.every(x=>Number.isFinite(x.cosine)),'EXACT_COSINE_FINITE');
 return rows.sort((a,b)=>b.cosine-a.cosine||byID(a.id,b.id));
}
function checked(r){
 need(r&&Object.keys(r).sort().join(',')==='bytes,path,sha256'&&typeof r.path==='string'&&Number.isInteger(r.bytes)&&r.bytes>=0&&/^[a-f0-9]{64}$/.test(r.sha256),'ASSET_REF');
 const p=path.resolve(r.path);for(let x=p;;x=path.dirname(x)){need(!fs.lstatSync(x).isSymbolicLink(),'ASSET_NO_SYMLINK');if(path.dirname(x)===x)break;}
 const b=fs.readFileSync(p);need(b.length===r.bytes&&hash(b)===r.sha256,'ASSET_SIZE_SHA');return p;
}
function newJSON(p,x){const b=Buffer.from(JSON.stringify(x)+'\n'),fd=fs.openSync(p,'wx',0o600);try{let i=0;while(i<b.length){const n=fs.writeSync(fd,b,i,b.length-i);need(n>0,'OUTPUT_SHORT_WRITE');i+=n;}fs.fsyncSync(fd);}finally{fs.closeSync(fd);}const d=fs.openSync(path.dirname(p),'r');try{fs.fsyncSync(d);}finally{fs.closeSync(d);}return {path:p,bytes:b.length,sha256:hash(b)};}
function makeCounters(c,onEvent){
 const bound=limits(c),values=Object.fromEntries(KEYS.map(k=>[k,0]));
 return {values,bound,
 attempt(base,event){const a=base+'_attempted',d=base+'_completed';need(own(bound,a)&&values[a]<bound[a]&&values[a]===values[d],'NATIVE_ATTEMPT_CAP_OR_INFLIGHT');values[a]++;if(event)onEvent(event);},
 complete(base){const a=base+'_attempted',d=base+'_completed';need(values[d]+1===values[a]&&values[d]<bound[d],'NATIVE_COMPLETE_ACCOUNTING');values[d]++;},
 hit(){need(values.native_hit_source_validations<bound.native_hit_source_validations,'NATIVE_HIT_CAP');values.native_hit_source_validations++;},
 terminal(observedHits){need(KEYS.filter(k=>k!=='native_hit_source_validations').every(k=>values[k]===bound[k])&&values.native_hit_source_validations===observedHits,'NATIVE_TERMINAL_COUNTERS');}
 };
}
function percentile(a,q){need(a.length>0,'LATENCY_NONEMPTY');const x=[...a].sort((a,b)=>a-b),p=(x.length-1)*q,l=Math.floor(p),r=Math.ceil(p);return x[l]+(x[r]-x[l])*(p-l);}
async function runCorpus(core,c,sourceIDs,queryIDs,vectors,queries,counted,progress,folder){
 const {n,q}=geometry(c),index=new Map(sourceIDs.map((id,i)=>[id,i]));
 const t=performance.now();counted.attempt('native_index_initializations','INDEX_BEFORE');
 const db=new core.VectorDb({dimensions:384,distanceMetric:core.JsDistanceMetric.Cosine,storagePath:path.join(folder,'ruvector.db')});
 counted.complete('native_index_initializations');progress('INDEX_COMPLETE');
 counted.attempt('native_len_calls',null);need(await db.len()===0,'NATIVE_INDEX_NOT_EMPTY');counted.complete('native_len_calls');
 for(let i=0;i<n;i++){counted.attempt('native_insertions','INSERT_BEFORE');await db.insert({id:sourceIDs[i],vector:vectors[i]});counted.complete('native_insertions');}
 counted.attempt('native_len_calls',null);need(await db.len()===n,'NATIVE_INDEX_INCOMPLETE');counted.complete('native_len_calls');progress('INSERTIONS_COMPLETE');
 const insertSeconds=(performance.now()-t)/1000,rows=[];let observedHits=0;
 for(let i=0;i<q;i++){
  counted.attempt('native_queries','QUERY_BEFORE');const started=performance.now(),hits=await db.search({vector:queries[i],k:n});counted.complete('native_queries');
  need(Array.isArray(hits)&&hits.length<=n&&new Set(hits.map(x=>x.id)).size===hits.length,'NATIVE_UNIQUE_BOUNDED_HITS');
  const raw=hits.map(h=>{const x=compact(h,vectors,index);counted.hit();observedHits++;return x;}),ordered=tiedOrder(raw);
  const exact=exactCosine(vectors,queries[i],sourceIDs),seen=new Set(ordered.map(x=>x.id)),missing=sourceIDs.filter(id=>!seen.has(id));
  rows.push({query_id:queryIDs[i],ids:ordered.map(x=>x.id),scores:ordered.map(x=>x.score),rank_scores:ordered.map((_,j)=>n-j),
   raw_ids:raw.map(x=>x.id),raw_scores:raw.map(x=>x.score),raw_hits:raw,requested_k:n,
   elapsed_seconds:(performance.now()-started)/1000,retrieval_miss:missing.length>0,coverage_complete:missing.length===0,missing_source_ids:missing,
   fallback_required:missing.length>0,exact_cosine_diagnostic:exact,exact_full_order_agreement:JSON.stringify(exact.map(x=>x.id))===JSON.stringify(ordered.map(x=>x.id)),
   native_order_used_except_returned_equal_score_ties:true,exact_order_never_substituted:true});
  progress('QUERY_COMPLETE');
 }
 return {queries:rows,index_initialization_and_insert_seconds:insertSeconds,observedHits,
  query_latency_seconds:{p50:percentile(rows.map(x=>x.elapsed_seconds),.5),p95:percentile(rows.map(x=>x.elapsed_seconds),.95)}};
}
module.exports={KIND,BINARY_SHA,KEYS,geometry,limits,ids,decode,vectorBytes,compact,tiedOrder,exactCosine,checked,newJSON,makeCounters,runCorpus,percentile};
if(require.main===module){
 const START=performance.now(),result={schema_version:1,epoch:58,queries:[],errors:[]};let output,fd,progressPath,counted,rows=0,bytes=0,maxSeconds=60;
 function progress(event){need(performance.now()-START<maxSeconds*1000,'NATIVE_HELPER_TIME_BOUND');Object.assign(result,counted.values);result.stage=event;
  if(fd!==undefined){const line=JSON.stringify({event,monotonic_milliseconds:performance.now()-START,counters:counted.values})+'\n';rows++;bytes+=Buffer.byteLength(line);need(rows<=10000&&bytes<=4194304,'NATIVE_PROGRESS_CAP');let at=0;const b=Buffer.from(line);while(at<b.length){const n=fs.writeSync(fd,b,at,b.length-at);need(n>0,'NATIVE_PROGRESS_SHORT_WRITE');at+=n;}fs.fsyncSync(fd);}}
 (async()=>{
  need(process.argv.length===5,'NATIVE_ARGUMENTS');const [configPath,bundlePath,target]=process.argv.slice(2);output=target;
  const configRaw=fs.readFileSync(configPath),c=JSON.parse(configRaw),bundleRaw=fs.readFileSync(bundlePath);need(bundleRaw.length<=16777216,'NATIVE_BUNDLE_CAP');const b=JSON.parse(bundleRaw),{n,q}=geometry(c);maxSeconds=c.native_helper_max_seconds;
  counted=makeCounters(c,progress);Object.assign(result,counted.values,{kind:KIND,phase:c.phase,plan_sha256:c.plan_sha256,native_config_sha256:hash(configRaw)});
  need(b&&Object.keys(b).sort().join(',')==='corpus,epoch,kind,phase,plan_sha256,queries,schema_version'&&b.schema_version===1&&b.epoch===58&&b.kind===KIND&&b.phase===c.phase&&b.plan_sha256===c.plan_sha256,'NATIVE_BUNDLE_BINDING');
  need(b.corpus&&b.queries&&Object.keys(b.corpus).sort().join(',')==='ids,vectors'&&Object.keys(b.queries).sort().join(',')==='ids,vectors','NATIVE_FEATURES_ONLY');
  const sourceIDs=ids(b.corpus.ids,n,'NATIVE_CORPUS_IDS'),queryIDs=ids(b.queries.ids,q,'NATIVE_QUERY_IDS');
  need(JSON.stringify(queryIDs)===JSON.stringify(c.selected_query_ids),'NATIVE_FIXED_QUERY_IDS');
  const vectors=decode(b.corpus.vectors,n),queries=decode(b.queries.vectors,q);
  need(fs.readFileSync('/proc/self/status','utf8').split('\n').some(x=>/^Cpus_allowed_list:\s*0-1$/.test(x)),'CPU_AFFINITY_0_1');
  need(process.env.RAYON_NUM_THREADS==='2'&&process.env.UV_THREADPOOL_SIZE==='2','NATIVE_CPU_THREAD_ENV');
  need(checked(c.native_helper_source)===__filename,'FROZEN_HELPER_SELF');
  const stat=fs.readFileSync('/proc/self/stat','utf8').split(')').slice(1).join(')').trim().split(/\s+/);
  progressPath=path.join(path.dirname(output),'native-progress.jsonl');
  const bindings=newJSON(path.join(path.dirname(output),'native-bindings.json'),{schema_version:1,epoch:58,kind:KIND,phase:c.phase,
   helper_pid:process.pid,helper_startticks:Number(stat[19]),helper_source_sha256:hash(fs.readFileSync(__filename)),plan_sha256:c.plan_sha256,
   native_config_sha256:hash(configRaw),bundle_sha256:hash(bundleRaw),progress_path:progressPath,
   process_clock_origin_milliseconds:START,maximum_progress_bytes:4194304,maximum_progress_rows:10000,helper_max_seconds:maxSeconds});
  result.bindings=bindings;fd=fs.openSync(progressPath,'wx',0o600);
  const deny=()=>{counted.values.Node_network_operations_attempted++;progress('FORBIDDEN_NETWORK');throw Error('NODE_NETWORK_FORBIDDEN');};
  const net=require('node:net'),http=require('node:http'),https=require('node:https'),dns=require('node:dns'),dgram=require('node:dgram');
  net.Socket.prototype.connect=deny;net.connect=deny;net.createConnection=deny;net.Server.prototype.listen=deny;http.request=deny;http.get=deny;https.request=deny;https.get=deny;globalThis.fetch=deny;
  dgram.Socket.prototype.send=deny;dgram.Socket.prototype.connect=deny;dgram.Socket.prototype.bind=deny;
  for(const k of ['lookup','resolve','resolve4','resolve6','reverse']){dns[k]=deny;if(dns.promises&&k in dns.promises)dns.promises[k]=deny;}
  const binary=path.join(c.native_root,'retrieval/node_modules/ruvector-core-linux-x64-gnu/ruvector.node');
  need(c.native_binary.sha256===BINARY_SHA&&c.native_binary.path===binary&&checked(c.native_binary)===binary,'FROZEN_NATIVE_BINARY');
  need(Array.isArray(c.native_closure_sources)&&c.native_closure_sources.length>=4&&c.native_closure_sources.length<=32,'NATIVE_WRAPPER_CLOSURE_REFS');
  const closure=new Set(c.native_closure_sources.map(checked));
  for(const rel of ['@ruvector/core/index.js','@ruvector/core/package.json','ruvector-core-linux-x64-gnu/index.js','ruvector-core-linux-x64-gnu/package.json'])
   need(closure.has(path.join(c.native_root,'retrieval/node_modules',rel)),'NATIVE_WRAPPER_CLOSURE_COMPLETE');
  const originalDlopen=process.dlopen;
  process.dlopen=function(module,filename,...rest){need(path.resolve(filename)===binary,'ONLY_FROZEN_NATIVE_DLOPEN');counted.attempt('native_binary_loads','NATIVE_LOAD_BEFORE');const value=originalDlopen.call(this,module,filename,...rest);counted.complete('native_binary_loads');progress('NATIVE_LOAD_COMPLETE');return value;};
  const core=require(path.join(c.native_root,'retrieval/node_modules/@ruvector/core'));
  need(counted.values.native_binary_loads_completed===1,'SINGLE_ACTUAL_NATIVE_LOAD');
  counted.attempt('native_version_calls','NATIVE_VERSION_BEFORE');const version=core.version();counted.complete('native_version_calls');progress('NATIVE_VERSION_COMPLETE');
  need(version==='2.3.0'&&require(path.join(c.native_root,'retrieval/node_modules/@ruvector/core/package.json')).version==='0.1.33'&&require(path.join(c.native_root,'retrieval/node_modules/ruvector-core-linux-x64-gnu/package.json')).version==='0.1.31','NATIVE_VERSIONS');
  Object.assign(result,{binary_version:version,package_version:'0.1.33',platform_package_version:'0.1.31',native_binary_sha256:BINARY_SHA,bundle_sha256:hash(bundleRaw),helper_source_sha256:hash(fs.readFileSync(__filename)),helper_pid:process.pid,corpus_document_count:n,query_count:q,top_k:n,raw_native_scores_are_not_probabilities:true});
  const folder=path.join(path.dirname(output),'native-database');fs.mkdirSync(folder);
  const observed=await runCorpus(core,c,sourceIDs,queryIDs,vectors,queries,counted,progress,folder);counted.terminal(observed.observedHits);Object.assign(result,observed);
  result.retrieval_miss_count=result.queries.filter(x=>x.retrieval_miss).length;result.resources=process.resourceUsage();result.helper_wall_seconds=(performance.now()-START)/1000;progress('COMPLETE');
  fs.closeSync(fd);fd=undefined;const raw=fs.readFileSync(progressPath);result.progress={path:progressPath,rows,bytes:raw.length,sha256:hash(raw)};newJSON(output,result);
 })().catch(error=>{if(counted)Object.assign(result,counted.values);result.errors.push({type:error.name,code:/^[A-Z0-9_]{1,120}$/.test(error.message)?error.message:'NATIVE_HELPER_FAILED'});try{progress('FAILED');}catch(ignored){}try{if(output)newJSON(output+'.failure.json',result);}catch(ignored){}if(fd!==undefined){try{fs.closeSync(fd);}catch(ignored){}}console.error(error.stack||String(error));process.exitCode=2;});
}

