// Offline MetaHarness adapter for the frozen CR/LF transport-framing check.
// This gate consumes recorded software evidence; it does not run hardware or
// claim a hidden/held-out evaluation.
import {createHash} from 'node:crypto';
import {mkdtempSync,readFileSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';

const USAGE='Usage: node tools/transport_gate.mjs FIRMWARE_ROOT METAHARNESS_ROOT OUTPUT_JSON';
if(process.argv.includes('--help')||process.argv.includes('-h')){console.log(USAGE);process.exit(0);}
const [firmwareArg,metaharnessArg,outputArg]=process.argv.slice(2);
if(!firmwareArg||!metaharnessArg||!outputArg)throw Error(USAGE);

const fw=resolve(firmwareArg),mh=resolve(metaharnessArg),output=resolve(outputArg);
const POLICY=Object.freeze({schemaVersion:1,framingCases:4,baselineScore:0,candidateScore:1,minDelta:0,seed:20260926});
const METAHARNESS_COMMIT='d5833dc6512ac1adeeef91a331c29055cd8a4dbb';
const SOURCE_SHA256=Object.freeze({
  promotion:'828be79b48e94dd3bf9f0d51d5dbb60d53663d54bf40baae6991c522ba2b81af',
  stats:'4b23c82cf811e0546e37a45efa92505b6f8cf79260eb0d1e9df0267f4fea9bf2',
});
const sha=value=>createHash('sha256').update(value).digest('hex');
const isDigest=value=>typeof value==='string'&&/^[0-9a-f]{64}$/i.test(value);
const veto=(name,pass,reason)=>({name,pass,reason});
let temporary;

try{
  const evidenceBytes=readFileSync(join(fw,'tests/evidence/transport-framing.json'));
  const parsedEvidence=JSON.parse(evidenceBytes);
  const recordPresent=parsedEvidence!==null&&typeof parsedEvidence==='object'&&!Array.isArray(parsedEvidence);
  const evidence=recordPresent?parsedEvidence:{};
  const regressions=Array.isArray(evidence.valid_command_regressions)?evidence.valid_command_regressions:[];
  const tools=evidence.tool_sha256;
  const toolHashesPresent=tools!==null&&typeof tools==='object'&&!Array.isArray(tools)&&
    Object.keys(tools).length>0&&Object.keys(tools).every(name=>name.length>0)&&Object.values(tools).every(isDigest);
  const regressionsPass=regressions.length>0&&regressions.every(row=>
    row!==null&&typeof row==='object'&&typeof row.name==='string'&&row.name.length>0&&row.pass===true);
  const labPass=evidence.lab?.status==='pass';
  const candidateRejected=Number.isInteger(evidence.candidate?.rejected_before_write)
    ? evidence.candidate.rejected_before_write:0;

  const hardVetoes=[
    veto('record',recordPresent,'evidence must be a JSON object'),
    veto('schema',evidence.schema_version===POLICY.schemaVersion,'schema_version must equal 1'),
    veto('parent',typeof evidence.parent_sha==='string'&&/^[0-9a-f]{40}$/i.test(evidence.parent_sha),'parent_sha must be a full 40-hex commit hash'),
    veto('evaluator_hash',isDigest(evidence.evaluator_sha256),'evaluator_sha256 must be present'),
    veto('tool_hashes',toolHashesPresent,'tool_sha256 must contain at least one named SHA-256 digest'),
    veto('baseline_total',evidence.baseline?.framing_cases_total===POLICY.framingCases,'baseline framing_cases_total must equal 4'),
    veto('baseline_score',evidence.baseline?.rejected_before_write===POLICY.baselineScore,'baseline rejected_before_write must equal 0'),
    veto('baseline_stale',evidence.baseline?.stale_followups===3,'baseline stale_followups must equal 3'),
    veto('baseline_cr_normalization',evidence.baseline?.carriage_return_normalizations===1,'baseline carriage_return_normalizations must equal 1'),
    veto('candidate_total',evidence.candidate?.framing_cases_total===POLICY.framingCases,'candidate framing_cases_total must equal 4'),
    veto('candidate_score',evidence.candidate?.rejected_before_write===POLICY.framingCases,'candidate rejected_before_write must equal 4'),
    veto('no_stale_followups',evidence.candidate?.stale_followups===0,'candidate stale_followups must equal 0'),
    veto('no_rejected_writes',evidence.candidate?.write_count_for_rejected_inputs===0,'candidate write_count_for_rejected_inputs must equal 0'),
    veto('valid_commands',regressionsPass,'all named valid_command_regressions must pass'),
    veto('lab',labPass,"lab.status must equal 'pass'"),
  ];

  temporary=mkdtempSync(join(tmpdir(),'esp32-transport-metaharness-'));
  writeFileSync(join(temporary,'package.json'),'{"type":"module"}');
  for(const [name,digest] of Object.entries(SOURCE_SHA256)){
    const source=readFileSync(join(mh,'packages/darwin-mode/src/bench',name+'.ts'),'utf8');
    if(sha(source)!==digest)throw Error('MetaHarness source changed: '+name);
    writeFileSync(join(temporary,name+'.js'),stripTypeScriptTypes(source));
  }
  const {decidePromotion}=await import(pathToFileURL(join(temporary,'promotion.js')).href);
  const candidateScores=Array.from({length:POLICY.framingCases},(_,index)=>index<candidateRejected?1:0);
  const transportSafe=evidence.candidate?.stale_followups===0&&evidence.candidate?.write_count_for_rejected_inputs===0;
  const result=(index,child)=>({
    taskId:'crlf-framing-'+(index+1),variantId:child?'candidate':'baseline',parentId:child?'baseline':null,
    repoCommit:typeof evidence.parent_sha==='string'?evidence.parent_sha:'missing-parent',
    solved:child?candidateScores[index]===POLICY.candidateScore:true,
    publicTestPassed:child?candidateScores[index]===POLICY.candidateScore:true,
    hiddenTestPassed:false,regressionPassed:regressionsPass&&labPass,
    durationMs:0,costUsd:0,changedFiles:[],blockedFileTouches:[],
    safetyViolations:child&&!transportSafe?['transport hard veto failed']:[],
    hallucinatedFileRefs:false,traceQuality:1,patchPath:'',tracePath:'',baseScore:child?candidateScores[index]:POLICY.baselineScore,
    finalScore:child?candidateScores[index]:POLICY.baselineScore,
  });
  const parentResults=Array.from({length:POLICY.framingCases},(_,index)=>result(index,false));
  const childResults=Array.from({length:POLICY.framingCases},(_,index)=>result(index,true));
  const decision=decidePromotion({parentResults,childResults,cleanReplay:regressionsPass&&labPass,
    minDelta:POLICY.minDelta,seed:POLICY.seed});
  const failedVetoes=hardVetoes.filter(check=>!check.pass);
  const promote=decision.promote&&failedVetoes.length===0;
  const reasons=promote
    ? ['MetaHarness promotion passed',...hardVetoes.map(check=>check.name+' passed')]
    : [...(decision.promote?[]:decision.reasons),...failedVetoes.map(check=>check.reason)];
  const receipt={schema_version:1,accepted_for:'PR review only',promote,reasons,decision,hard_vetoes:hardVetoes,
    policy:POLICY,metaharness_commit:METAHARNESS_COMMIT,metaharness_source_sha256:SOURCE_SHA256,
    evidence_sha256:sha(evidenceBytes),parent_sha:evidence.parent_sha??null,evaluator_sha256:evidence.evaluator_sha256??null,
    tool_sha256:toolHashesPresent?tools:null,hidden_labels_used:false,
    mappings:{finalScore:'Observed fraction of four forbidden CR/LF framing inputs rejected before a write; immutable baseline is 0 and required candidate is 1.',
      solved:'The corresponding visible framing input was rejected before write. No hidden labels were used.',
      regressionPassed:'Every named valid-command regression and the recorded software lab passed.',
      cleanReplay:'Every named valid-command regression and the recorded software lab passed.'},
    limitations:['Recorded software evidence only; no physical-hardware claim.','No hidden or unseen evaluation; hiddenTestPassed=false for every MetaHarness result.','The four deterministic framing cases are not independent random trials.']};
  writeFileSync(output,JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({promote,reasons,decision}));
  if(!promote)process.exitCode=1;
}catch(error){
  writeFileSync(output,JSON.stringify({schema_version:1,promote:false,reasons:[String(error)]},null,2)+'\n');
  console.error(String(error));process.exitCode=1;
}finally{if(temporary)rmSync(temporary,{recursive:true,force:true});}
