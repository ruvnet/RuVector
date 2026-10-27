'use strict';
// E57 source-pinned INT8-vector retrieval helper. No model, text, qrels or labels.
// New adaptation of the reviewed E55 helper; actual library/API is unchanged.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {performance}=require('node:perf_hooks');
const MODEL_SHA='6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4';
const GEOMETRY=Object.freeze({engineering:{kind:'REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV',docs:128},
 retrieval_development:{kind:'REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT',docs:5183}});
const BINARY_SHA='ff559e9fac2e3d29a8b32e890fdb047bcd4bd28f389b4731ad72148b221d5a2e';
const KEYS=Object.freeze(['native_index_initializations_attempted','native_index_initializations_completed',
 'native_insertions_attempted','native_insertions_completed','native_queries_attempted',
 'native_queries_completed','native_len_calls_attempted','native_len_calls_completed',
 'native_hit_source_validations','native_binary_loads_attempted','native_binary_loads_completed',
 'native_version_calls_attempted','native_version_calls_completed','Node_network_operations_attempted']);
function geometry(config){
 const g=config&&GEOMETRY[config.phase];
 need(g&&config.schema_version===1&&config.epoch===57&&config.kind===g.kind
  &&config.corpus_document_count===g.docs&&config.query_count===16&&config.top_k===10,
  'NATIVE_EXACT_PLAN_GEOMETRY');
 need(config.bge_model&&config.bge_model.sha256===MODEL_SHA,'INT8_MODEL_PROVENANCE_PIN');
 need(Number.isFinite(config.native_helper_max_seconds)&&config.native_helper_max_seconds>0
  &&config.native_helper_max_seconds<=60,'NATIVE_HELPER_TIME_CAP');
 return g;
}
function expected(config){const n=geometry(config).docs;
 return Object.freeze(Object.fromEntries(KEYS.map((k,i)=>[k,[1,1,n,n,16,16,2,2,160,1,1,1,1,0][i]])));
}
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const need=(x,c)=>{if(!x)throw Error(c);};
const own=(x,k)=>Object.prototype.hasOwnProperty.call(x,k);
const byID=(a,b)=>a<b?-1:a>b?1:0;
function decode(block,rows){
 need(block&&Object.keys(block).sort().join(',')==='bytes,data_base64,dtype,encoding,sha256,shape'&&block.encoding==='base64'&&block.dtype==='float32le'&&Array.isArray(block.shape)
  &&block.shape.length===2&&block.shape[0]===rows&&block.shape[1]===384
  &&block.bytes===rows*1536&&typeof block.data_base64==='string','BUNDLE_F32_SHAPE');
 const b=Buffer.from(block.data_base64,'base64');
 need(b.length===block.bytes&&b.toString('base64')===block.data_base64&&hash(b)===block.sha256,'BUNDLE_F32_IDENTITY');
 const vectors=[];
 for(let row=0;row<rows;row++){
  const v=new Float32Array(384);let norm=0;
  for(let col=0;col<384;col++){
   v[col]=b.readFloatLE(row*1536+col*4);need(Number.isFinite(v[col]),'F32_FINITE');norm+=v[col]*v[col];
  }
  need(Math.abs(Math.sqrt(norm)-1)<=1e-5,'F32_NORMALIZED');vectors.push(v);
 }
 return vectors;
}
function vectorBytes(v){
 need(v&&Object.keys(v).length===384&&Object.keys(v).every((k,i)=>k===String(i)),'NATIVE_RETURNED_VECTOR_SHAPE');
 const b=Buffer.alloc(1536);
 for(let i=0;i<384;i++){
  need(typeof v[i]==='number'&&Number.isFinite(v[i])&&Object.is(Math.fround(v[i]),v[i]),'NATIVE_RETURNED_F32');
  b.writeFloatLE(v[i],i*4);
 }
 return b;
}
function ids(value,n,code){
 need(Array.isArray(value)&&value.length===n&&value.every(x=>typeof x==='string'&&x.length>0
  &&Buffer.byteLength(x)<=256&&!/[\u0000-\u001f\u007f]/.test(x))&&new Set(value).size===n,code);return value;
}
function compact(hit,vectors,indexByID){
 need(hit&&typeof hit.id==='string'&&indexByID.has(hit.id),'NATIVE_KNOWN_SOURCE_ID');
 const index=indexByID.get(hit.id),actual=vectorBytes(hit.vector),source=vectorBytes(vectors[index]);
 need(actual.equals(source),'NATIVE_SOURCE_VECTOR_BITS');
 need(!own(hit,'metadata'),'NATIVE_METADATA_ABSENT');
 need(typeof hit.score==='number'&&Number.isFinite(hit.score),'NATIVE_SCORE_FINITE');
 // Retain only the exact source identity, raw native score and F32 receipt.
 return {id:hit.id,score:hit.score,vector_receipt:{source_vector_index:index,dimensions:384,
  dtype:'float32le',sha256:hash(actual),source_equal_before_stripping:true},metadata_receipt:{present:false}};
}
function returnedTieOrder(hits){
 // Native score orientation is deliberately not reinterpreted. Equal-score
 // groups already returned by the ANN API are ordered by document ID only.
 // Tied documents omitted at k=10 cannot be globally normalized here.
 const out=[];
 for(let i=0;i<hits.length;){let j=i+1;while(j<hits.length&&hits[j].score===hits[i].score)j++;
  out.push(...hits.slice(i,j).sort((a,b)=>byID(a.id,b.id)));i=j;}
 return out;
}
function checked(ref){
 need(ref&&typeof ref.path==='string'&&Number.isInteger(ref.bytes)&&ref.bytes>=0
  &&/^[a-f0-9]{64}$/.test(ref.sha256),'FILE_REFERENCE');
 const p=path.resolve(ref.path);for(let q=p;;q=path.dirname(q)){need(!fs.lstatSync(q).isSymbolicLink(),'ASSET_NOT_SYMLINK');if(path.dirname(q)===q)break;}
 const b=fs.readFileSync(p);need(b.length===ref.bytes&&hash(b)===ref.sha256,'ASSET_HASH');return p;
}
function writeProgressLine(fd,line){
 const written=fs.writeSync(fd,line);
 need(Number.isInteger(written)&&written>0&&written===Buffer.byteLength(line),'SHORT_NATIVE_PROGRESS_WRITE');
 fs.fsyncSync(fd);
}
function makeCounters(onEvent,EXPECTED){
 need(EXPECTED&&Object.keys(EXPECTED).length===KEYS.length&&KEYS.every(k=>Number.isInteger(EXPECTED[k])&&EXPECTED[k]>=0),
  'NATIVE_EXPECTED_VALIDATION_ONLY');
 const counters=Object.fromEntries(KEYS.map(k=>[k,0]));
 return {counters,
  attempt(base,event){const k=base+'_attempted',done=base+'_completed';
   need(own(EXPECTED,k)&&counters[k]<EXPECTED[k]&&counters[k]===counters[done],'NATIVE_OPERATION_CAP_OR_INFLIGHT');
   counters[k]++;if(event!==null)onEvent(event);else need(base==='native_len_calls','ONLY_LEN_MAY_OMIT_EVENT');},
  complete(base){const a=base+'_attempted',k=base+'_completed';
   need(own(EXPECTED,k)&&counters[k]+1===counters[a]&&counters[k]<EXPECTED[k],'NATIVE_COMPLETION_ACCOUNTING');counters[k]++;},
  hit(){need(counters.native_hit_source_validations<160,'NATIVE_HIT_VALIDATION_CAP');counters.native_hit_source_validations++;},
  terminal(){need(KEYS.every(k=>counters[k]===EXPECTED[k]),'NATIVE_EXACT_TERMINAL_COUNTERS');}
 };
}
module.exports={GEOMETRY,MODEL_SHA,KEYS,geometry,expected,decode,vectorBytes,compact,ids,returnedTieOrder,writeProgressLine,makeCounters};
if(require.main===module){
 const START=performance.now();
 const result={schema_version:1,epoch:57,queries:[],errors:[]};
 let output,progressFD,progressPath,progressRows=0,progressBytes=0,counted,maxSeconds=60;
 function newJSON(target,value){const bytes=Buffer.from(JSON.stringify(value)+'\n');
  const fd=fs.openSync(target,'wx',0o600);try{fs.writeFileSync(fd,bytes);fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
  return {path:target,bytes:bytes.length,sha256:hash(bytes)};
 }
 function progress(event){
  need(performance.now()-START<maxSeconds*1000,'NATIVE_HELPER_TIME_BOUND');
  Object.assign(result,counted.counters);result.stage=event;
  if(progressFD!==undefined){
   const line=JSON.stringify({event,monotonic_milliseconds:performance.now()-START,counters:counted.counters})+'\n';
   progressRows++;progressBytes+=Buffer.byteLength(line);
   need(progressRows<=10000&&progressBytes<=4194304,'NATIVE_PROGRESS_BOUND');
   writeProgressLine(progressFD,line);
  }
 }
 (async()=>{
  need(process.argv.length===5,'NATIVE_EXACT_ARGUMENTS');
  const [configPath,bundlePath,target]=process.argv.slice(2);output=target;
  const configBytes=fs.readFileSync(configPath),config=JSON.parse(configBytes),source=fs.readFileSync(bundlePath);
  need(source.length<=16777216,'NATIVE_BUNDLE_BYTE_CAP');const bundle=JSON.parse(source);
  const g=geometry(config),N=g.docs;maxSeconds=config.native_helper_max_seconds;
  counted=makeCounters(progress,expected(config));Object.assign(result,counted.counters,{kind:config.kind,phase:config.phase});
  need(bundle.schema_version===1&&bundle.epoch===57&&bundle.kind===config.kind&&bundle.phase===config.phase
   &&bundle.plan_sha256===hash(configBytes),'NATIVE_EXACT_BUNDLE');
  need(Object.keys(bundle).sort().join(',')==='corpus,epoch,kind,phase,plan_sha256,queries,schema_version',
   'NATIVE_BUNDLE_FEATURES_ONLY');
  need(bundle.corpus&&bundle.queries&&Object.keys(bundle.corpus).sort().join(',')==='ids,vectors'
   &&Object.keys(bundle.queries).sort().join(',')==='ids,vectors','NATIVE_ONLY_VECTOR_ID_PAYLOADS');
  const sourceIDs=ids(bundle.corpus.ids,N,'NATIVE_EXACT_CORPUS_IDS');
  const queryIDs=ids(bundle.queries.ids,16,'NATIVE_EXACT_QUERY_IDS');
  need(JSON.stringify(queryIDs)===JSON.stringify(config.selected_query_ids),'NATIVE_FIXED_QUERY_IDENTITY');
  const vectors=decode(bundle.corpus.vectors,N),queries=decode(bundle.queries.vectors,16);
  const indexByID=new Map(sourceIDs.map((id,index)=>[id,index]));
  need(fs.readFileSync('/proc/self/status','utf8').split('\n').some(x=>/^Cpus_allowed_list:\s*0-1$/.test(x)),
   'CPU_AFFINITY_0_1');
  need(process.env.RAYON_NUM_THREADS==='2'&&process.env.UV_THREADPOOL_SIZE==='2','NATIVE_CPU_THREAD_ENV');
  need(checked(config.native_helper_source)===__filename,'FROZEN_HELPER_SELF_SOURCE');
  need(config.native_helper_source.sha256===hash(fs.readFileSync(__filename)),'HELPER_SOURCE_IDENTITY');
  const statRaw=fs.readFileSync('/proc/self/stat','utf8'),stat=statRaw.slice(statRaw.lastIndexOf(')')+1).trim().split(/\s+/);
  progressPath=path.join(path.dirname(output),'native-progress.jsonl');
  const bindings=newJSON(path.join(path.dirname(output),'native-bindings.json'),{schema_version:1,epoch:57,kind:config.kind,phase:config.phase,
   helper_pid:process.pid,helper_startticks:Number(stat[19]),helper_source_sha256:hash(fs.readFileSync(__filename)),
   plan_sha256:hash(configBytes),bundle_sha256:hash(source),process_clock_origin_milliseconds:START,
   progress_path:progressPath,progress_clock_policy:'milliseconds since helper source start',
   progress_schema:'event,monotonic_milliseconds,counters; canonical14 observed counters',
   maximum_progress_bytes:4194304,maximum_progress_rows:10000,helper_max_seconds:maxSeconds});
  result.bindings=bindings;
  progressFD=fs.openSync(progressPath,'wx',0o600);
  const deny=()=>{counted.counters.Node_network_operations_attempted++;progress('FORBIDDEN_NETWORK');throw Error('NODE_NETWORK_FORBIDDEN');};
  const net=require('node:net'),http=require('node:http'),https=require('node:https'),dns=require('node:dns'),dgram=require('node:dgram');
  net.Socket.prototype.connect=deny;net.connect=deny;net.createConnection=deny;net.Server.prototype.listen=deny;
  http.request=deny;http.get=deny;https.request=deny;https.get=deny;globalThis.fetch=deny;
  dgram.Socket.prototype.send=deny;dgram.Socket.prototype.connect=deny;dgram.Socket.prototype.bind=deny;
  for(const k of ['lookup','resolve','resolve4','resolve6','reverse']){dns[k]=deny;if(dns.promises&&k in dns.promises)dns.promises[k]=deny;}
  const binary=path.join(config.native_root,'retrieval/node_modules/ruvector-core-linux-x64-gnu/ruvector.node');
  need(config.native_binary.sha256===BINARY_SHA&&config.native_binary.path===binary
   &&checked(config.native_binary)===binary,'FROZEN_NATIVE_BINARY');
  const originalDlopen=process.dlopen;
  process.dlopen=function(module,filename,...rest){
   need(path.resolve(filename)===binary,'ONLY_FROZEN_NATIVE_DLOPEN');
   counted.attempt('native_binary_loads','NATIVE_LOAD_BEFORE');
   const value=originalDlopen.call(this,module,filename,...rest);
   counted.complete('native_binary_loads');progress('NATIVE_LOAD_COMPLETE');return value;
  };
  const core=require(path.join(config.native_root,'retrieval/node_modules/@ruvector/core'));
  need(counted.counters.native_binary_loads_completed===1,'ACTUAL_SINGLE_NATIVE_BINARY_LOAD');
  counted.attempt('native_version_calls','NATIVE_VERSION_BEFORE');const version=core.version();counted.complete('native_version_calls');
  need(version==='2.3.0'
   &&require(path.join(config.native_root,'retrieval/node_modules/@ruvector/core/package.json')).version==='0.1.33'
   &&require(path.join(config.native_root,'retrieval/node_modules/ruvector-core-linux-x64-gnu/package.json')).version==='0.1.31',
   'EXACT_NATIVE_VERSION');
  Object.assign(result,{binary_version:version,package_version:'0.1.33',platform_package_version:'0.1.31',
   native_binary_sha256:BINARY_SHA,bundle_sha256:hash(source),helper_source_sha256:hash(fs.readFileSync(__filename)),
   helper_pid:process.pid,plan_sha256:bundle.plan_sha256,corpus_document_count:N,query_count:16,top_k:10,
   raw_native_scores_are_not_probabilities:true,rank_score_policy:'10 through 1 encode returned rank only; do not reinterpret raw native score.'});
  const folder=path.join(path.dirname(output),'native-database');fs.mkdirSync(folder);
  const t0=performance.now();counted.attempt('native_index_initializations','INDEX_BEFORE');
  const db=new core.VectorDb({dimensions:384,distanceMetric:core.JsDistanceMetric.Cosine,storagePath:path.join(folder,'ruvector.db')});
  counted.complete('native_index_initializations');
  counted.attempt('native_len_calls',null);
  need(await db.len()===0,'NATIVE_INDEX_NOT_EMPTY');counted.complete('native_len_calls');
  // INSERT_BEFORE is emitted for insert attempts only; len snapshots are seen
  // by the next legal progress row without introducing an unreviewed event.
  for(let i=0;i<N;i++){
   counted.attempt('native_insertions','INSERT_BEFORE');await db.insert({id:sourceIDs[i],vector:vectors[i]});
   counted.complete('native_insertions');
  }
  // len() is still counted before invoking the native operation. No extra
  // progress event is added to the frozen event vocabulary.
  counted.attempt('native_len_calls',null);
  need(await db.len()===N,'NATIVE_INDEX_INCOMPLETE');counted.complete('native_len_calls');
  result.index_initialization_and_insert_seconds=(performance.now()-t0)/1000;
  for(let i=0;i<16;i++){
   counted.attempt('native_queries','QUERY_BEFORE');const t=performance.now();
   const hits=await db.search({vector:queries[i],k:10});counted.complete('native_queries');
   need(Array.isArray(hits)&&hits.length===10&&new Set(hits.map(h=>h.id)).size===10,'NATIVE_TEN_UNIQUE_SOURCE_HITS');
   const rawHits=hits.map(h=>{const v=compact(h,vectors,indexByID);counted.hit();return v;});
   const ordered=returnedTieOrder(rawHits);
   result.queries.push({query_id:queryIDs[i],ids:ordered.map(h=>h.id),scores:ordered.map(h=>h.score),
    rank_scores:ordered.map((_,j)=>10-j),raw_ids:rawHits.map(h=>h.id),raw_scores:rawHits.map(h=>h.score),
    raw_hits:rawHits,requested_k:10,elapsed_seconds:(performance.now()-t)/1000,
    native_order_used_except_returned_equal_score_ties:true,global_boundary_ties_unverified:true});
   progress('QUERY_COMPLETE');
  }
  counted.terminal();result.resources=process.resourceUsage();
  result.helper_wall_seconds=(performance.now()-START)/1000;
  progress('COMPLETE');fs.closeSync(progressFD);progressFD=undefined;
  const progressRaw=fs.readFileSync(progressPath);
  result.progress={path:progressPath,rows:progressRows,bytes:progressRaw.length,sha256:hash(progressRaw)};
  newJSON(output,result);
 })().catch(error=>{
  if(counted)Object.assign(result,counted.counters);result.errors.push({type:error.name,
   code:/^[A-Z0-9_]{1,120}$/.test(error.message)?error.message:'NATIVE_HELPER_FAILED'});
  try{progress('FAILED');}catch(ignored){}
  try{if(output)newJSON(output+'.failure.json',result);}catch(ignored){}
  if(progressFD!==undefined){try{fs.closeSync(progressFD);}catch(ignored){}}
  console.error(error.stack||String(error));process.exitCode=2;
 });
}
