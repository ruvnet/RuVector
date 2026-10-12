import { randomUUID } from 'node:crypto';
import { sweep, reportMarkdown } from './index.mjs';
import { reportHtml } from './visualize.mjs';
const bucket = process.env.REPORT_BUCKET;
if (!bucket || !/^[a-z0-9][a-z0-9._-]+$/.test(bucket)) throw new Error('REPORT_BUCKET is required');
const id = process.env.CLOUD_RUN_EXECUTION ?? randomUUID();
const prefix = `runs/${new Date().toISOString().slice(0,10)}/${id}`;
async function token() {
  const r = await fetch('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',{headers:{'Metadata-Flavor':'Google'},signal:AbortSignal.timeout(10000)});
  if(!r.ok) throw new Error(`GCP identity HTTP ${r.status}`);
  return (await r.json()).access_token;
}
async function storage(path, options={}) {
  return fetch(`https://storage.googleapis.com/${path}`,{...options,headers:{...options.headers,Authorization:`Bearer ${await token()}`},signal:AbortSignal.timeout(30000)});
}
async function readObject(name) {
  const r=await storage(`storage/v1/b/${bucket}/o/${encodeURIComponent(name)}?alt=media`);
  if(r.status===404)return null;
  if(!r.ok)throw new Error(`Read report HTTP ${r.status}`);
  return r.json();
}
async function writeObject(name,content,type='application/json',generation=null) {
  const match=generation===null?'':`&ifGenerationMatch=${generation}`;
  const r=await storage(`upload/storage/v1/b/${bucket}/o?uploadType=media&name=${encodeURIComponent(name)}${match}`,{method:'POST',headers:{'content-type':type},body:typeof content==='string'?content:JSON.stringify(content,null,2)});
  if(!r.ok)throw new Error(`Write report HTTP ${r.status}`);
  return r.json();
}
// GCS create-if-absent prevents overlapping scheduler/manual executions. A crash leaves
// a lock for operator inspection, rather than silently running two mutation controllers.
const lock=await writeObject('controller.lock',{id,startedAt:new Date().toISOString()},'application/json',0);
try {
  const npmToken=process.env.NPM_TOKEN;
  if(!npmToken)throw new Error('NPM_TOKEN secret not configured');
  const identity=await fetch('https://registry.npmjs.org/-/whoami',{headers:{Authorization:`Bearer ${npmToken}`},signal:AbortSignal.timeout(30000)});
  if(!identity.ok)throw new Error(`npm authentication HTTP ${identity.status}`);
  const {username}=await identity.json();
  if(username!=='ruvnet')throw new Error('npm identity is not configured maintainer');
  const previous=await readObject('latest.json');
  const report=await sweep({previous});
  report.runner={execution:id,sourceSha:process.env.SOURCE_SHA??'unknown',npmIdentity:username,publishEnabled:false};
  await writeObject(`${prefix}/inventory.json`,report,'application/json',0);
  await writeObject(`${prefix}/inventory.md`,reportMarkdown(report),'text/markdown',0);
  await writeObject(`${prefix}/improvement-plan.json`,report.plan,'application/json',0);
  await writeObject(`${prefix}/inventory.html`,reportHtml(report),'text/html',0);
  if(!report.complete) throw new Error(`Incomplete sweep; preserved at gs://${bucket}/${prefix}`);
  await writeObject('latest.json',report);
  console.log(JSON.stringify({event:'constellation-sweep-complete',packages:report.packages.length,cycles:report.graph.runtimeCycles.length,report:`gs://${bucket}/${prefix}`,sourceSha:report.runner.sourceSha,publishEnabled:false}));
} finally {
  const r=await storage(`storage/v1/b/${bucket}/o/controller.lock?ifGenerationMatch=${lock.generation}`,{method:'DELETE'});
  if(!r.ok)throw new Error(`Lock release HTTP ${r.status}`);
}
