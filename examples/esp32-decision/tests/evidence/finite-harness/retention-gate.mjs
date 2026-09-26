// Deterministic offline adapter to pinned MetaHarness decidePromotion.
// Inputs are public adversarial regression fixtures, not hidden measurements.
import {readFileSync,writeFileSync,mkdtempSync,rmSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {stripTypeScriptTypes} from 'node:module';
import {pathToFileURL} from 'node:url';
const [firmwareArg,mhArg,outArg]=process.argv.slice(2);
const fw=resolve(firmwareArg);
const receiptArg=join(fw,'build-finite/evaluator.json'),rigArg=join(fw,'tools/rig.py');
if (!outArg) throw Error('Usage: node retention-gate.mjs FIRMWARE_ROOT METAHARNESS_ROOT OUTPUT_JSON');
const sha=x=>createHash('sha256').update(x).digest('hex');
const check=(v,m)=>{if(!v)throw Error(m)};
const pins={promotion:'828be79b48e94dd3bf9f0d51d5dbb60d53663d54bf40baae6991c522ba2b81af',stats:'4b23c82cf811e0546e37a45efa92505b6f8cf79260eb0d1e9df0267f4fea9bf2'};
let tmp;
try {
 const rb=readFileSync(resolve(receiptArg)), r=JSON.parse(rb);
 check(r.parent_sha==='6fdb5f12ac7fe16752eeb60405e52567081706db','Unexpected parent');
 check(r.candidate_sha256===sha(readFileSync(resolve(rigArg))),'Candidate source digest mismatch');
 check(r.finite_legacy_equal===true,'Finite legacy behavior not preserved');
 const evaluatorDigest='cc5c17ecb2c9513167bb1a040c0e3854c063d5842578511d3c0985ecd70e498a';
 check(r.evaluator_sha256===evaluatorDigest&&sha(readFileSync(join(fw,'tests/test_finite_benchmark.py')))===evaluatorDigest,'Frozen evaluator changed');
 const experimentBytes=readFileSync(join(fw,'build-finite/experiment.json')), experiment=JSON.parse(experimentBytes);
 check(experiment.parent===r.parent_sha&&experiment.seed===719&&experiment.candidate_budget===1,'Experiment boundary changed');
 for (const [p,h] of Object.entries(experiment.frozen))check(sha(readFileSync(join(fw,p)))===h,'Invariant source changed: '+p);
 const labBytes=readFileSync(join(fw,'build-finite/lab.log')),lab=labBytes.toString();
 const labSummary=JSON.parse(lab.trim().split('\n').at(-1));
 check(labSummary.software_acceptance==='pass','Integrated lab evidence absent or failed');
 check(!/FAILED \(|ERROR:|AddressSanitizer:|runtime error:/.test(lab),'Lab failure diagnostic');
 check(r.candidate_failing_subtests===0&&r.candidate_test_methods===9,'Independent evaluator missing or failed');
 check(Array.isArray(r.malformed)&&r.malformed.length>0,'Missing malformed input fixtures');
 check(new Set(r.malformed.map(x=>x.name)).size===r.malformed.length,'Duplicate fixture names');
 for (const f of r.malformed) {
  check(typeof f.name==='string'&&f.name.length>0,'Missing fixture name');
  check(typeof f.parent_rejected==='boolean'&&f.candidate_rejected===true,'Malformed fixture result invalid or candidate unsafe');
 }
 tmp=mkdtempSync(join(tmpdir(),'retention-metaharness-'));
 writeFileSync(join(tmp,'package.json'),'{"type":"module"}');
 for(const [name,digest] of Object.entries(pins)) {
  const src=readFileSync(join(resolve(mhArg),'packages/darwin-mode/src/bench',name+'.ts'),'utf8');
  check(sha(src)===digest,'Pinned MetaHarness source mismatch');
  writeFileSync(join(tmp,name+'.js'),stripTypeScriptTypes(src));
 }
 const {decidePromotion}=await import(pathToFileURL(join(tmp,'promotion.js')).href);
 const result=(f,child)=>({taskId:f.name,variantId:child?'finite-guard':'baseline',repoCommit:r.parent_sha,parentId:r.parent_sha,solved:child?f.candidate_rejected:f.parent_rejected,publicTestPassed:child?f.candidate_rejected:f.parent_rejected,hiddenTestPassed:false,regressionPassed:true,finalScore:Number(child?f.candidate_rejected:f.parent_rejected),safetyViolations:[],blockedFileTouches:[]});
 const input={parentResults:r.malformed.map(f=>result(f,false)),childResults:r.malformed.map(f=>result(f,true)),cleanReplay:r.finite_legacy_equal,minDelta:0.05,seed:719};
 const decision=decidePromotion(input);
 check(decision.promote===true,'Measured malformed rejection gain not accepted by MetaHarness');
 const badReplay=decidePromotion({...input,cleanReplay:false});
 check(badReplay.promote===false,'MetaHarness failed replay veto');
 const regressed=structuredClone(input);regressed.childResults.forEach(x=>x.regressionPassed=false);
 check(decidePromotion(regressed).promote===false,'MetaHarness failed regression veto');
 const receipt={schema:1,parent_sha:r.parent_sha,candidate_sha256:r.candidate_sha256,independent_receipt_sha256:sha(rb),frozen_evaluator_sha256:evaluatorDigest,experiment_sha256:sha(experimentBytes),lab_sha256:sha(labBytes),lab_pass:true,metaharness_commit:'d5833dc6512ac1adeeef91a331c29055cd8a4dbb',source_sha256:pins,seed:719,decision,fixtures:r.malformed.length,parent_rejected:r.malformed.filter(x=>x.parent_rejected).length,candidate_rejected:r.malformed.length,negative_gates:{failed_replay_veto:true,regression_veto:true},accepted_for:'PR review only',deployment_promoted:false,mappings:{finalScore:'Observed malformed fixture rejection: 1 rejected, 0 not rejected.',solved:'Correct fixture rejection only.',cleanReplay:'Independent finite legacy result equality.',regressionPassed:'Independent lab pass and finite legacy equality.',safetyViolations:'No observed violation of declared narrow fixture invariants; not deployment safety.'},limitations:['Public deterministic fixtures, not unseen generalization evidence.','MetaHarness bootstrap over fixture scores is not a timing confidence interval or broader statistical safety claim.','No physical hardware, speed, energy, model or quantization improvement measured.','Independent receipt claims must be checked against evaluator source and raw outputs by coordinator.']};
 writeFileSync(resolve(outArg),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
} catch(e) {
 writeFileSync(resolve(outArg),JSON.stringify({promote:false,error:String(e)},null,2)+'\n');console.error(String(e));process.exitCode=1;
} finally { if(tmp)rmSync(tmp,{recursive:true,force:true}); }
