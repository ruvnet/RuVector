// Definitions only. Root invokes ONE fixed reviewed capsule recovery, then persists its actual receipt.
// No scientific authorization, automatic execution, retry, or reconciliation bridge.
const W59='/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028';
const PUB59=W59+'/publication';
const REPO59='ruvnet/RuVector', BRANCH59='benchmarks/ruvector-sota-2026-10-22';
const DENY59=['486888de9602bbb5d817a55f707b4fcfd11c65e0','3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'];
const READ_DENY59=[...DENY59,...["07bf80160f5d797ccb845b4dffffb787e16ffccf","09cc1a05368eb6deb111d81843ff0ee58a19f04f","1089df824a50b6c95f0cd42b4d9c18b16439f809","15c4e0b1a0a7dc22d76ec2531c7a7648440c8123","18269afd275bb4dd0df46598dddc5589aaf149f6","238aba760ff2674dd5081a33c68aced78a92a6b9","2f49331bc33c489c1acb18469c7db40f9248e6d8","4445a0b19239e85e83384cdc312cbfded999d4a6","493794e759dbea4d05ad30dbac6db0fc990cdfc7","53f13c3ed4c50c521459f186da22fdf7066007db","544d81f36f3c760f17dee1e1cd0266b710486dbb","5a113347c8466d72e346dce8130342a8ac5b89de","6a74fc4a21c59254bb20258a33db17d8b2f69815","76c0a79db634455f48495d859cee1e6d853185dd","784832adc97416932334c075e1edcc1c140d9c4b","8e24326bfbefe1a08c78888c96c86d7e3d98107d","a799498cfe9d9117e47227185e6b28abff47cc53","abc2c1ec1da3cd82fa44efb189202f235718c30e","b1af014cf3a57f40bb30d29df9ee1eec3b69ef9a","b4d9cb2700b8c5044f94052e7f525ea570257728","b541dadc9402b4fc781e712a86db238e4b34b9ac","c8a8757521490a4b0c7f6ac2f26024cc428bbaf9","ca4064e43ccc9c2dca59e5441f234c190db8a851","cdc3a6ce00c3cac70fc6b83261652207bf7d748b","d96c80472b484bc858186f9fa730510c82fa6a34","dc5c5d9cafc35ea4ca0073f2cc061f2bbe987685","ecbeda0c5ff845ed8960e273bb00f1acd13855df","f0329e5f3109afa210a978af044ad08065334c74"]];
const q59=x=>"'"+String(x).replaceAll("'","'\\''")+"'";
const sh59=async cmd=>{const r=await tools.exec_command({cmd,max_output_tokens:100000});if(r.exit_code!==0)throw Error(r.output);return r.output;};
const un59=r=>{if(r.isError)throw Error('Git read failed');return typeof r.structuredContent?.content==='string'?JSON.parse(r.structuredContent.content):r.structuredContent;};
const get59=async url=>{if(READ_DENY59.some(x=>url.includes(x)))throw Error('Retired object query forbidden');return un59(await tools.mcp__codex_apps__github_fetch({url}));};
const rawGet59=async url=>{if(READ_DENY59.some(x=>url.includes(x)))throw Error('Retired object query forbidden');return tools.mcp__codex_apps__github_fetch({url});};
const readText59=async p=>{
  const n=Number((await sh59('taskset -c 0,1 python -I -B -c '+q59('from pathlib import Path;print(len(Path('+JSON.stringify(p)+').read_text()))'))).trim());
  const chunks=[];
  for(let start=0;start<n;start+=160000){
    const positions=[];for(let i=start;i<Math.min(n,start+160000);i+=20000)positions.push(i);
    const rows=await Promise.allSettled(positions.map(async i=>{const r=await tools.exec_command({cmd:'taskset -c 0,1 python -I -B -c '+q59('from pathlib import Path;print(Path('+JSON.stringify(p)+').read_text()['+i+':'+Math.min(n,i+20000)+'],end="")'),max_output_tokens:20000});if(r.exit_code!==0||r.output.startsWith('Warning:'))throw Error('Metadata chunk failed');return r.output;}));
    for(const r of rows){if(r.status!=='fulfilled')throw r.reason;chunks.push(r.value);}
  }
  const result=chunks.join('');if([...result].length!==n)throw Error('Metadata character count');return result;
};
const read59=async p=>JSON.parse(await readText59(p));
const pin59=async p=>JSON.parse(await sh59('taskset -c 0,1 python -I -B -c '+q59('from pathlib import Path;import hashlib,json;p=Path('+JSON.stringify(p)+');b=p.read_bytes();print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))')));
let CONTEXT59=null,SUCCESS59=null;
const guard59=async(mode,extra={})=>{
  if(!CONTEXT59)throw Error('Explicit phase context required');
  let cmd='taskset -c 0,1 python -I -B '+q59(W59+'/publication-helpers-g28/guard.py')+' '+q59(mode)+' --phase '+q59(CONTEXT59.phase)+' --approval '+q59(CONTEXT59.approval)+' --approval-sha '+q59(CONTEXT59.approvalSHA);
  for(const [k,v]of Object.entries(extra))cmd+=' --'+k+' '+q59(v);
  return JSON.parse(await sh59(cmd));
};
const payload59=async value=>(await sh59('taskset -c 0,1 python -I -B -c '+q59('import base64;print(base64.b64encode('+JSON.stringify(JSON.stringify(value))+'.encode()).decode())'))).trim();
const capture59=async(id,raw)=>guard59('capture-raw',{id,payload:await payload59(raw)});
const proof59=async(proof,raw)=>guard59('capture-proof',{proof,payload:await payload59(raw)});
const mutate59=async slot=>{
  if(!/^recovery59-g28-[a-z0-9-]+$/.test(slot.id))throw Error('Fresh recovery59 generation28 ID only');
  await new Promise(resolve=>setTimeout(resolve,2000));
  await guard59('reserve',{id:slot.id,kind:slot.kind});
  const args=await read59(PUB59+'/requests/'+slot.id+'.json');
  await guard59('intent',{id:slot.id,kind:slot.kind});await guard59('owner');
  let raw;
  try{raw=await tools[slot.kind](args);}
  catch(error){await capture59(slot.id,{transport_error:String(error)});throw error;}
  // This capture is awaited BEFORE un59, owner checks, fresh reads or response validation.
  await capture59(slot.id,raw);
  if(slot.kind.endsWith('github_update_ref')){
    await guard59('owner');
    await proof59('ref',await rawGet59('https://api.github.com/repos/'+REPO59+'/git/ref/heads/'+BRANCH59));
  }
  return guard59('response',{id:slot.id,kind:slot.kind});
};
const siblings59=(actual,before,key,sha)=>{
  if(actual.truncated||before.truncated||actual.tree.length!==before.tree.length)throw Error('Sibling count');
  for(const row of before.tree){const r=actual.tree.find(x=>x.path===row.path);if(!r||r.mode!==row.mode||r.type!==row.type||r.sha!==(row.path===key?sha:row.sha))throw Error('Unrelated sibling changed');}
};
const checkLeaves59=(actual,expected,scopeSHA)=>{
  if(actual.truncated||actual.sha!==scopeSHA||expected.scope!=='benchmarks/ruvector-sota-2026-10-22/')throw Error('Full exact scope required');
  const map=new Map(expected.files.map(r=>[r.path,r]));if(map.size!==expected.files.length)throw Error('Duplicate leaf');
  const rows=actual.tree.filter(r=>r.type!=='tree');if(rows.length!==map.size)throw Error('Leaf count');
  const seen=new Set();
  for(const r of rows){const e=map.get(r.path);if(seen.has(r.path)||!e||DENY59.includes(r.sha)||['path','mode','type','sha','size'].some(k=>r[k]!==e[k]))throw Error('Leaf identity differs');seen.add(r.path);}
  return rows.length;
};
const publishRecovery59=async(approvalPath,approvalSHA)=>{
  const phase='recovery';
  if(CONTEXT59||!/^[0-9a-f]{64}$/.test(approvalSHA))throw Error('One explicit fresh controller phase only');
  CONTEXT59={phase,approval:approvalPath,approvalSHA};await guard59('check');
  const stage=W59+'/stage-'+phase,meta=await read59(stage+'/metadata.json'),mp=await pin59(stage+'/metadata.json');
  const leafPin=await pin59(stage+'/'+meta.expected_leaves.file);
  if(leafPin.bytes!==meta.expected_leaves.bytes||leafPin.sha256!==meta.expected_leaves.sha256)throw Error('Expected leaf map changed');
  const expected=await read59(stage+'/'+meta.expected_leaves.file),url='https://api.github.com/repos/'+REPO59;
  const head=await get59(url+'/git/ref/heads/'+BRANCH59),main0=await get59(url+'/git/ref/heads/main');
  if(head.ref!=='refs/heads/'+BRANCH59||head.object.sha!==meta.source_parent)throw Error('Actual scoped parent advanced; STOP');
  if(main0.ref!=='refs/heads/main'||!/^[0-9a-f]{40}$/.test(main0.object?.sha)||DENY59.includes(main0.object.sha))throw Error('Invalid read-only main snapshot');
  const parent=await get59(url+'/git/commits/'+meta.source_parent);if(parent.tree.sha!==meta.expected_source_root)throw Error('Actual parent root');
  const r0=await get59(url+'/git/trees/'+parent.tree.sha),b0=await get59(url+'/git/trees/'+meta.expected_source_bench);
  if(r0.tree.find(r=>r.path==='benchmarks')?.sha!==meta.expected_source_bench||b0.tree.find(r=>r.path==='ruvector-sota-2026-10-22')?.sha!==meta.expected_source_scope)throw Error('Actual root/bench/scope baseline');
  for(const row of meta.blobs)await mutate59(row);
  for(const row of meta.trees)await mutate59(row);
  const r1=await get59(url+'/git/trees/'+meta.expected_final_root),b1=await get59(url+'/git/trees/'+meta.expected_final_bench);
  if(r1.sha!==meta.expected_final_root||b1.sha!==meta.expected_final_bench)throw Error('Actual desired tree SHA');
  siblings59(r1,r0,'benchmarks',meta.expected_final_bench);siblings59(b1,b0,'ruvector-sota-2026-10-22',meta.expected_final_scope);
  const leaves=checkLeaves59(await get59(url+'/git/trees/'+meta.expected_final_scope+'?recursive=1'),expected,meta.expected_final_scope);
  if(leaves!==meta.new_Git_leaves)throw Error('Expected whole-scope count');
  if((await get59(url+'/git/ref/heads/'+BRANCH59)).object.sha!==meta.source_parent)throw Error('Late parent advance; STOP');
  const commitSlot=meta.requests.find(r=>r.kind.endsWith('github_create_commit')),refSlot=meta.requests.find(r=>r.kind.endsWith('github_update_ref'));
  const commit=await mutate59(commitSlot);
  await proof59('commit',await rawGet59(url+'/git/commits/'+commit.ack_sha));await guard59('verify-commit');
  if((await get59(url+'/git/ref/heads/'+BRANCH59)).object.sha!==meta.source_parent)throw Error('Pre-CAS parent advance; STOP');
  await mutate59(refSlot);
  if((await get59(url+'/git/ref/heads/'+BRANCH59)).object.sha!==commit.ack_sha)throw Error('Fresh postCAS mismatch; STOP');
  const main1=await get59(url+'/git/ref/heads/main');if(main1.ref!=='refs/heads/main'||!/^[0-9a-f]{40}$/.test(main1.object?.sha)||DENY59.includes(main1.object.sha))throw Error('Invalid closing read-only main snapshot');
  SUCCESS59={schema_version:1,epoch:59,kind:meta.kind,phase,commit:commit.ack_sha,parent:meta.source_parent,root:meta.expected_final_root,bench:meta.expected_final_bench,scope:meta.expected_final_scope,leaves,CP_rows:meta.CP_rows,verified_actual_full_tree:true,stage_metadata_sha256:mp.sha256,scientific_source_plan_sha256:meta.scientific_plan.sha256,scientific_source_freeze_sha256:meta.scientific_source_freeze.sha256,scientific_epoch_manifest_sha256:meta.science_manifest.sha256,recovery_freeze_sha256:meta.recovery_freeze.sha256,canonical_capsule_envelope_sha256:meta.capsule_envelope.sha256,old_failed_blob_history:'UNKNOWN',old_failed_blob_read_404_now:true,all_28_prior_blob_recreation_requests:0,new_scientific_calls:0,CP_only_transport_count:70,E44_overlay55_remains_separate:true,E59_overlay78_excluded_from_Git_and_CP:true,original_blob5_execution:'UNKNOWN_PERMANENTLY',old_forbidden_Git_objects_not_queried_recreated_or_live_referenced:true,publication_only:true,scientific_authorization_granted:false,read_only_main:{before:main0.object.sha,after:main1.object.sha,unchanged_within_phase:main0.object.sha===main1.object.sha,main_mutation_requests:0,mutation_scope:'ONLY_SCOPED_BRANCH'}};
  return SUCCESS59;
};
const persistRecovery59Receipt=async result=>{
  if(!SUCCESS59||result!==SUCCESS59)throw Error('Only actual successful result from this controller');
  const p=PUB59+'/publication-'+result.phase+'-result.json';
  await sh59('taskset -c 0,1 python -I -B -c '+q59('from pathlib import Path;import os;p=Path('+JSON.stringify(p)+');b='+JSON.stringify(JSON.stringify(result)+'\n')+'.encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);f=os.fdopen(fd,"wb");f.write(b);f.flush();os.fsync(f.fileno());f.close();d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)'));
  return p;
};
