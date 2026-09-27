// Offline adapter. No model retraining, deployment, network, or hidden-test claims.
import {readFileSync,writeFileSync,mkdtempSync,rmSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {stripTypeScriptTypes} from 'node:module';
import {pathToFileURL} from 'node:url';
const [firmwareArg, metaharnessArg, outputArg]=process.argv.slice(2);
if (!firmwareArg || !metaharnessArg || !outputArg) throw Error('Usage: node tools/workspace_gate.mjs FIRMWARE_ROOT METAHARNESS_ROOT OUTPUT_JSON');
const fw=resolve(firmwareArg), mh=resolve(metaharnessArg), output=resolve(outputArg), evidence={};
const sha=b=>createHash('sha256').update(b).digest('hex');
function read(p){const b=readFileSync(join(fw,p));evidence[p]=sha(b);return b;}
function json(p){return JSON.parse(read(p).toString());}
function requireTrue(b,s){if(!b)throw Error(s);}
let tmp;
try {
const experiment=json('build-workspace/experiment.json');
for(const [path,expected] of Object.entries(experiment.frozen)) requireTrue(sha(read(path))===expected,'Frozen file changed: '+path);
const host=read('build-workspace/host-tests.log').toString();
requireTrue(/Ran 5 tests in [\d.]+s\s+OK\s*$/.test(host),'Workspace tests missing or failed');
for(const name of ['test_exact_decisions_for_all_heads_precisions_and_kernel_profiles','test_sensor_replay_and_memory_reduction','test_invalid_capacities_fail_compilation','test_model_larger_than_capacity_rejected_before_access','test_cmake_literal_metadata_and_legacy_fallback'])requireTrue(new RegExp(name+'.* \\.\\.\\. ok').test(host),'Workspace check missing: '+name);
const lab=read('build-workspace/lab.log').toString();
const summary=JSON.parse(lab.trim().split('\n').at(-1));
requireTrue(summary.software_acceptance==='pass','Software acceptance missing or failed');
requireTrue(!/FAILED \(|ERROR:|AddressSanitizer:|runtime error:/.test(lab),'Lab failure diagnostic');
requireTrue(/Ran 27 tests in [\d.]+s\s+OK/.test(lab),'Lab tests missing');
for(let n=1;n<=11;n++)requireTrue(lab.includes(`Pair ${n}/11: exact decisions True`),'Exact pair missing: '+n);
const rows=json('build-sensor/validation.json');requireTrue(Array.isArray(rows)&&rows.length>0,'Empty regression corpus');
const model=read('build-workspace/frozen-model/model.h').toString();
const targets=[];
for(const target of ['esp32s3','esp32c6']){
 const r=json(`build-workspace/memory-${target}.json`);requireTrue(r.execution==='linked ELF analysis','Wrong evidence class');
 for(const variant of ['generic','compact']){
  const v=r[variant];requireTrue(v.target===target,'Wrong target');
  requireTrue(read(`build-workspace/build-${target}-${variant}.log`).toString().includes('Project build complete.'),'Build incomplete');
  for(const [file,field] of [['merged-binary.bin','image_sha256'],['ruvector_decision.elf','elf_sha256']])requireTrue(sha(read(`build-workspace/${target}-${variant}/${file}`))===v[field],'Artifact digest mismatch: '+file);
  requireTrue(read(`build-workspace/${target}-${variant}/ruvector_decision.bin`).length===v.application_bytes,'Application size mismatch');
  requireTrue(model.includes(v.model_sha256),'Model digest not bound to frozen header');
 }
 requireTrue(r.generic.model_sha256===r.compact.model_sha256,'Model changed');
 const base=r.generic.static_symbols.ctx+r.generic.static_symbols.workspace,child=r.compact.static_symbols.ctx+r.compact.static_symbols.workspace;
 requireTrue(Number.isSafeInteger(base)&&Number.isSafeInteger(child)&&base>child&&child>0,'Invalid memory reduction');
 requireTrue(base-child===r.static_symbol_bytes_recovered,'Symbol reduction mismatch');
 requireTrue(r.generic.sections['.dram0.bss']-r.compact.sections['.dram0.bss']===base-child,'BSS reduction mismatch');
 requireTrue(r.generic.application_bytes-r.compact.application_bytes===r.application_bytes_recovered&&r.application_bytes_recovered>=0,'Application size regressed or inconsistent');
 targets.push({target,parent_static_bytes:base,child_static_bytes:child,static_bytes_recovered:base-child,application_bytes_recovered:r.application_bytes_recovered,score:1-child/base});
}
const s3=json('build-workspace/memory-esp32s3.json');
const emulator=s3.emulator;
requireTrue(emulator?.exact_replay_match===true,'S3 exact emulator replay missing');
requireTrue(emulator.free_heap_recovered===s3.static_symbol_bytes_recovered,'S3 heap recovery mismatch');
for(const variant of ['generic','compact']) {
 const e=emulator[variant];
 requireTrue(e.replay_rows===rows.length && e.warmup_rows===rows.length,'S3 replay incomplete');
 requireTrue(e.heap_before===e.heap_after,'S3 warmed heap changed');
 requireTrue(e.meta.selftest_pass && e.meta.model_sha256===s3[variant].model_sha256,'S3 model or selftest mismatch');
 requireTrue(e.meta.kernel_sha256===experiment.frozen['components/rvdecision/rvdecision.c'],'S3 kernel mismatch');
}
requireTrue(emulator.generic.answers_sha256===emulator.compact.answers_sha256,'S3 reply digests differ');
const pins={promotion:'828be79b48e94dd3bf9f0d51d5dbb60d53663d54bf40baae6991c522ba2b81af',stats:'4b23c82cf811e0546e37a45efa92505b6f8cf79260eb0d1e9df0267f4fea9bf2'};
tmp=mkdtempSync(join(tmpdir(),'esp32-metaharness-'));writeFileSync(join(tmp,'package.json'),'{"type":"module"}');
for(const [name,digest]of Object.entries(pins)){const source=readFileSync(join(mh,'packages/darwin-mode/src/bench',name+'.ts'),'utf8');requireTrue(sha(source)===digest,'MetaHarness gate source changed');writeFileSync(join(tmp,name+'.js'),stripTypeScriptTypes(source));}
const {decidePromotion}=await import(pathToFileURL(join(tmp,'promotion.js')).href);
const result=(t,child)=>({taskId:t.target,variantId:child?'compact':'generic',repoCommit:experiment.parent,parentId:experiment.parent,solved:true,publicTestPassed:true,hiddenTestPassed:false,regressionPassed:true,finalScore:child?t.score:0,safetyViolations:[],blockedFileTouches:[]});
const decision=decidePromotion({parentResults:targets.map(t=>result(t,false)),childResults:targets.map(t=>result(t,true)),cleanReplay:true,minDelta:0.05,seed:20260926});
requireTrue(decision.promote,'MetaHarness rejected static reduction');
const receipt={schema:1,accepted_for:'PR review only',promote:true,physical_hardware:false,metaharness_commit:'d5833dc6512ac1adeeef91a331c29055cd8a4dbb',source_sha256:pins,parent:experiment.parent,targets,frozen_regression_rows:rows.length,decision,evidence_sha256:evidence,mappings:{finalScore:'Fraction of linked workspace and context symbol bytes removed. Generic baseline score is zero.',solved:'Existing frozen regression suite passed. Not an unseen heldout result.',cleanReplay:'Generic and compact host semantic replies matched by recorded workspace tests and all 574 S3 emulator rows matched with stable warmed heap.',safetyViolations:'No frozen arithmetic/evaluator hash changes found; not a probability or broad security certification.'},limitations:['No hidden or unseen evaluation. hiddenTestPassed=false is intentional; this calls decidePromotion, not evaluateGates.','The two targets are deterministic linked artifacts, not independent random trials. MetaHarness lower95 is not a paired timing confidence interval.','No speed, power, model accuracy or deployment safety improvement asserted.','Tests are consumed from local evidence receipts; this adapter does not rerun the whole lab.','Autogenous deployment hard gates are not evaluated without domain appropriate safety and latency evidence.']};
writeFileSync(output,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify({promote:true,accepted_for:receipt.accepted_for,targets}));
}catch(error){writeFileSync(output,JSON.stringify({promote:false,error:String(error),evidence_sha256:evidence},null,2)+'\n');console.error(String(error));process.exitCode=1;}finally{if(tmp)rmSync(tmp,{recursive:true,force:true});}
