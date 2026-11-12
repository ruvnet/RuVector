// Offline MetaHarness adapter for the frozen energy provenance experiment.
import {createHash} from 'node:crypto';
import {mkdtempSync,readFileSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';

const [firmwareArg,metaharnessArg,outputArg]=process.argv.slice(2);
if(!firmwareArg||!metaharnessArg||!outputArg)throw Error('Usage: node tools/provenance_gate.mjs FIRMWARE_ROOT METAHARNESS_ROOT OUTPUT_JSON');
const fw=resolve(firmwareArg),mh=resolve(metaharnessArg),output=resolve(outputArg);
const sha=value=>createHash('sha256').update(value).digest('hex');
const requireTrue=(value,message)=>{if(!value)throw Error(message);};
const read=path=>readFileSync(join(fw,path));
let temporary;
try{
  const experiment=JSON.parse(read('tests/evidence/provenance-experiment.json'));
  requireTrue(experiment.parent_sha==='21fd8753ea1d54473ef797e9c6ebaaa62c7749bf','Unexpected parent');
  requireTrue(experiment.candidate_budget===1&&experiment.seed===719,'Experiment budget or seed changed');
  for(const [path,digest] of Object.entries(experiment.frozen))requireTrue(sha(read(path))===digest,'Frozen file changed: '+path);
  requireTrue(sha(read('tools/energy_compare.py'))===experiment.candidate.energy_compare_sha256,'Candidate changed');
  requireTrue(sha(read('tests/test_energy_provenance.py'))===experiment.evaluator_sha256,'Evaluator changed');
  const validation=JSON.parse(read('tests/evidence/provenance-validation.json'));
  requireTrue(validation.parent.tamper_cases_rejected===0&&validation.candidate.tamper_cases_rejected===7,'Tamper result mismatch');
  requireTrue(validation.valid_retention_result_equal&&validation.integrated_lab.software_acceptance==='pass','Regression gate failed');

  const pins={promotion:'828be79b48e94dd3bf9f0d51d5dbb60d53663d54bf40baae6991c522ba2b81af',stats:'4b23c82cf811e0546e37a45efa92505b6f8cf79260eb0d1e9df0267f4fea9bf2'};
  temporary=mkdtempSync(join(tmpdir(),'esp32-provenance-metaharness-'));
  writeFileSync(join(temporary,'package.json'),'{"type":"module"}');
  for(const [name,digest] of Object.entries(pins)){
    const source=readFileSync(join(mh,'packages/darwin-mode/src/bench',name+'.ts'),'utf8');
    requireTrue(sha(source)===digest,'MetaHarness source changed: '+name);
    writeFileSync(join(temporary,name+'.js'),stripTypeScriptTypes(source));
  }
  const {decidePromotion}=await import(pathToFileURL(join(temporary,'promotion.js')).href);
  const result=(index,child)=>({taskId:'tamper-'+index,variantId:child?'bound':'parent',repoCommit:experiment.parent_sha,
    solved:true,publicTestPassed:true,hiddenTestPassed:false,regressionPassed:true,finalScore:child?1:0,
    safetyViolations:[],blockedFileTouches:[]});
  const parentResults=Array.from({length:7},(_,i)=>result(i,false));
  const childResults=Array.from({length:7},(_,i)=>result(i,true));
  const decision=decidePromotion({parentResults,childResults,cleanReplay:true,minDelta:.05,seed:719});
  requireTrue(decision.promote,'MetaHarness rejected observed provenance improvement');
  const negative={
    replay:decidePromotion({parentResults,childResults,cleanReplay:false,minDelta:.05,seed:719}).promote,
    regression:decidePromotion({parentResults,childResults:childResults.map((r,i)=>i? r:{...r,regressionPassed:false}),cleanReplay:true,minDelta:.05,seed:719}).promote,
    safety:decidePromotion({parentResults,childResults:childResults.map((r,i)=>i? r:{...r,safetyViolations:['frozen evaluator changed']}),cleanReplay:true,minDelta:.05,seed:719}).promote,
  };
  requireTrue(Object.values(negative).every(value=>value===false),'MetaHarness negative gate promoted');
  const receipt={schema:1,campaign:experiment.campaign,run:experiment.run,accepted_for:'PR review only',promote:true,
    physical_hardware:false,metaharness_commit:'d5833dc6512ac1adeeef91a331c29055cd8a4dbb',source_sha256:pins,
    parent:experiment.parent_sha,evaluator_sha256:experiment.evaluator_sha256,candidate_sha256:experiment.candidate.energy_compare_sha256,
    metric:'fraction of seven named provenance tamper cases rejected',parent_score:0,candidate_score:1,decision,negative,
    mappings:{solved:'The frozen evaluator produced the expected reject or valid result.',regressionPassed:'The valid eleven pair result and integrated software lab passed.',cleanReplay:'An independent worker reproduced all evaluator and finite benchmark tests.'},
    limitations:['Synthetic tamper fixtures are software evidence, not physical energy measurements.','MetaHarness bootstrap scores are deterministic case outcomes, not paired device timing confidence.','hiddenTestPassed=false; no unseen dataset or deployment claim.']};
  writeFileSync(output,JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({promote:true,tamper_cases:7,negative_gates:negative}));
}catch(error){writeFileSync(output,JSON.stringify({promote:false,error:String(error)},null,2)+'\n');console.error(String(error));process.exitCode=1;}
finally{if(temporary)rmSync(temporary,{recursive:true,force:true});}
