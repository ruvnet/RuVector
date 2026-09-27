// Offline MetaHarness adapter for the frozen concurrent transport check.
// This gate consumes deterministic software evidence; it does not run hardware
// or claim a statistical/hidden evaluation.
import {createHash} from 'node:crypto';
import {mkdtempSync,readFileSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';

const USAGE=`Usage: node tools/concurrency_gate.mjs FIRMWARE_ROOT METAHARNESS_ROOT OUTPUT_JSON

Reads FIRMWARE_ROOT/tests/evidence/concurrency-transport.json and evaluates the
frozen eight-pair baseline/candidate comparison with the pinned MetaHarness
Darwin promotion API. The command exits nonzero unless every evidence invariant
and every MetaHarness promotion clause passes.`;
if(process.argv.includes('--help')||process.argv.includes('-h')){
  console.log(USAGE);
  process.exit(0);
}
const [firmwareArg,metaharnessArg,outputArg]=process.argv.slice(2);
if(!firmwareArg||!metaharnessArg||!outputArg)throw Error(USAGE);

const fw=resolve(firmwareArg),mh=resolve(metaharnessArg),output=resolve(outputArg);
const POLICY=Object.freeze({
  schemaVersion:1,
  pairsTotal:8,
  baselineMisassociated:8,
  candidateMisassociated:0,
  baselineScore:0,
  candidateScore:1,
  minDelta:0,
  seed:20260926,
});
const METAHARNESS_COMMIT='d5833dc6512ac1adeeef91a331c29055cd8a4dbb';
const PARENT_SHA='47b042815052be2263498d0ca9ae2ae1f31cd7e1';
const EVALUATOR_SHA256='3496a962085d6183b4823dd7642f14054af1e7eb81d4b6091ad10d22c3fb07f5';
const CANDIDATE_TRANSPORT_SHA256='21465fc62d47e0a84c0fe3a3e048d8896c497cc04955c393c21a2d62e3e8c4a2';
const SOURCE_SHA256=Object.freeze({
  promotion:'828be79b48e94dd3bf9f0d51d5dbb60d53663d54bf40baae6991c522ba2b81af',
  stats:'4b23c82cf811e0546e37a45efa92505b6f8cf79260eb0d1e9df0267f4fea9bf2',
});
const REQUIRED_TOOL_SHA256=Object.freeze({
  parent_transport:'dc9ce59bba35793427b2dba3b7e7bf58334742a2a8ed579bdc272f7d8713eb5f',
  candidate_transport:CANDIDATE_TRANSPORT_SHA256,
  metaharness_promotion:SOURCE_SHA256.promotion,
  metaharness_stats:SOURCE_SHA256.stats,
  native_pair_report:'d5abfeb89908cb0e2d7cc03c3568c4246645fbfa4abd667c568344def8652fd6',
  sensor_report:'176192d88ad2a753ac795c61f522f3ebacf2c9f40a492194b1a5731a97ad4812',
});
const SEQUENTIAL_NAMES=Object.freeze(['alpha','beta','meta','predict_sample_7']);
const sha=value=>createHash('sha256').update(value).digest('hex');
const isDigest=value=>typeof value==='string'&&/^[0-9a-f]{64}$/i.test(value);
const veto=(name,pass,reason)=>({name,pass,reason});
let temporary;

try{
  const evidenceBytes=readFileSync(join(fw,'tests/evidence/concurrency-transport.json'));
  const parsedEvidence=JSON.parse(evidenceBytes);
  const recordPresent=parsedEvidence!==null&&typeof parsedEvidence==='object'&&!Array.isArray(parsedEvidence);
  const evidence=recordPresent?parsedEvidence:{};
  const toolHashes=evidence.tool_sha256;
  const requiredToolNames=[...Object.keys(REQUIRED_TOOL_SHA256),'concurrency_gate'].sort();
  const toolHashesPresent=toolHashes!==null&&typeof toolHashes==='object'&&!Array.isArray(toolHashes)&&
    JSON.stringify(Object.keys(toolHashes).sort())===JSON.stringify(requiredToolNames)&&
    Object.values(toolHashes).every(isDigest);
  const fixedToolHashesMatch=toolHashesPresent&&Object.entries(REQUIRED_TOOL_SHA256)
    .every(([name,digest])=>toolHashes[name]===digest);
  const evaluatorFileSha=sha(readFileSync(join(fw,'tests/test_transport_concurrency.py')));
  const candidateTransportFileSha=sha(readFileSync(join(fw,'tools/transport.py')));
  const gateFileSha=sha(readFileSync(new URL(import.meta.url)));
  const nativeReportFileSha=sha(readFileSync(join(fw,'build-lab/native.json')));
  const sensorReportFileSha=sha(readFileSync(join(fw,'build-sensor/report.json')));
  const sequential=Array.isArray(evidence.sequential_regressions)?evidence.sequential_regressions:[];
  const sequentialNames=sequential.map(row=>row?.name).sort();
  const sequentialPass=sequential.length===SEQUENTIAL_NAMES.length&&
    JSON.stringify(sequentialNames)===JSON.stringify([...SEQUENTIAL_NAMES].sort())&&sequential.every(row=>
    row!==null&&typeof row==='object'&&!Array.isArray(row)&&
    typeof row.name==='string'&&row.name.length>0&&row.pass===true);
  const framingPass=evidence.framing_regression!==null&&
    typeof evidence.framing_regression==='object'&&!Array.isArray(evidence.framing_regression)&&
    evidence.framing_regression.pass===true&&evidence.framing_regression.cases===4;
  const labPass=evidence.lab!==null&&typeof evidence.lab==='object'&&!Array.isArray(evidence.lab)&&
    evidence.lab.status==='pass'&&evidence.lab.rust_tests===57&&
    evidence.lab.python_tests===53&&evidence.lab.exact_native_pairs===11&&
    Array.isArray(evidence.lab.built_targets)&&evidence.lab.built_targets.length===0&&
    evidence.lab.physical_acceptance==='not performed';

  const hardVetoes=[
    veto('record',recordPresent,'evidence must be a JSON object'),
    veto('schema',evidence.schema_version===POLICY.schemaVersion,'schema_version must equal 1'),
    veto('parent',evidence.parent_sha===PARENT_SHA,'parent_sha must equal the frozen parent'),
    veto('evaluator_hash',evidence.evaluator_sha256===EVALUATOR_SHA256&&evaluatorFileSha===EVALUATOR_SHA256,'evaluator hash must equal the frozen digest and current evaluator file'),
    veto('tool_hashes',toolHashesPresent&&fixedToolHashesMatch,'tool_sha256 must contain exactly the frozen named digests'),
    veto('candidate_source',candidateTransportFileSha===CANDIDATE_TRANSPORT_SHA256&&toolHashes?.candidate_transport===candidateTransportFileSha,'candidate transport must match its frozen source digest'),
    veto('gate_source',toolHashes?.concurrency_gate===gateFileSha,'concurrency gate must match its recorded source digest'),
    veto('native_report',nativeReportFileSha===REQUIRED_TOOL_SHA256.native_pair_report&&toolHashes?.native_pair_report===nativeReportFileSha,'native pair report must match its frozen digest'),
    veto('sensor_report',sensorReportFileSha===REQUIRED_TOOL_SHA256.sensor_report&&toolHashes?.sensor_report===sensorReportFileSha,'sensor report must match its frozen digest'),
    veto('baseline_total',evidence.baseline?.pairs_total===POLICY.pairsTotal,'baseline.pairs_total must equal 8'),
    veto('baseline_misassociated',evidence.baseline?.misassociated===POLICY.baselineMisassociated,'baseline.misassociated must equal 8'),
    veto('candidate_total',evidence.candidate?.pairs_total===POLICY.pairsTotal,'candidate.pairs_total must equal 8'),
    veto('candidate_misassociated',evidence.candidate?.misassociated===POLICY.candidateMisassociated,'candidate.misassociated must equal 0'),
    veto('sequential_regressions',sequentialPass,'sequential_regressions must contain exactly the four frozen named passing checks'),
    veto('framing_regression',framingPass,'framing_regression must pass exactly four cases'),
    veto('lab',labPass,'lab must bind the exact 57 Rust, 53 Python and 11-pair software result with no target or physical run'),
  ];

  temporary=mkdtempSync(join(tmpdir(),'esp32-concurrency-metaharness-'));
  writeFileSync(join(temporary,'package.json'),'{"type":"module"}');
  for(const [name,digest] of Object.entries(SOURCE_SHA256)){
    const source=readFileSync(join(mh,'packages/darwin-mode/src/bench',name+'.ts'),'utf8');
    if(sha(source)!==digest)throw Error('MetaHarness source changed: '+name);
    writeFileSync(join(temporary,name+'.js'),stripTypeScriptTypes(source));
  }
  const {decidePromotion}=await import(pathToFileURL(join(temporary,'promotion.js')).href);
  const allRegressionsPass=sequentialPass&&framingPass&&labPass;
  const observedScores=record=>{
    const total=record?.pairs_total;
    const misassociated=record?.misassociated;
    if(total!==POLICY.pairsTotal||!Number.isInteger(misassociated)||misassociated<0||misassociated>total){
      return Array(POLICY.pairsTotal).fill(0);
    }
    return Array.from({length:total},(_,index)=>index<total-misassociated?1:0);
  };
  const baselineScores=observedScores(evidence.baseline);
  const candidateScores=observedScores(evidence.candidate);
  const result=(index,child)=>({
    taskId:'concurrent-transport-pair-'+(index+1),
    variantId:child?'candidate':'baseline',
    parentId:child?'baseline':null,
    repoCommit:typeof evidence.parent_sha==='string'?evidence.parent_sha:'missing-parent',
    solved:true,
    publicTestPassed:(child?candidateScores[index]:baselineScores[index])===1,
    hiddenTestPassed:false,
    regressionPassed:allRegressionsPass,
    durationMs:0,
    costUsd:0,
    changedFiles:[],
    blockedFileTouches:[],
    safetyViolations:[],
    hallucinatedFileRefs:false,
    traceQuality:1,
    patchPath:'',
    tracePath:'',
    baseScore:child?candidateScores[index]:baselineScores[index],
    finalScore:child?candidateScores[index]:baselineScores[index],
  });
  const parentResults=Array.from({length:POLICY.pairsTotal},(_,index)=>result(index,false));
  const childResults=Array.from({length:POLICY.pairsTotal},(_,index)=>result(index,true));
  const decision=decidePromotion({
    parentResults,
    childResults,
    cleanReplay:allRegressionsPass,
    minDelta:POLICY.minDelta,
    seed:POLICY.seed,
  });
  const failedVetoes=hardVetoes.filter(check=>!check.pass);
  const promote=decision.promote&&failedVetoes.length===0;
  const reasons=promote
    ? ['MetaHarness promotion passed',...hardVetoes.map(check=>check.name+' passed')]
    : [...(decision.promote?[]:decision.reasons),...failedVetoes.map(check=>check.reason)];
  const receipt={
    schema_version:1,
    accepted_for:'PR review only',
    promote,
    reasons,
    decision,
    hard_vetoes:hardVetoes,
    policy:POLICY,
    metaharness_commit:METAHARNESS_COMMIT,
    metaharness_source_sha256:SOURCE_SHA256,
    evidence_sha256:sha(evidenceBytes),
    parent_sha:evidence.parent_sha??null,
    evaluator_sha256:evidence.evaluator_sha256??null,
    tool_sha256:toolHashesPresent?toolHashes:null,
    hidden_labels_used:false,
    mappings:{
      finalScore:'Observed fraction of eight deterministic request/reply pairs correctly associated: baseline 0, candidate 1.',
      solved:'The deterministic evaluator ran for the corresponding visible pair; association quality is represented by finalScore.',
      regressionPassed:'Every named sequential regression, the framing regression, and the recorded software lab passed.',
      cleanReplay:'Every named sequential regression, the framing regression, and the recorded software lab passed.',
    },
    limitations:[
      'Recorded software evidence only; no physical-hardware, timing, or energy claim.',
      'No hidden or unseen evaluation; hiddenTestPassed=false for every MetaHarness result.',
      'The eight deterministic request/reply cases are not independent statistical physical trials.',
    ],
  };
  writeFileSync(output,JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({promote,reasons,decision}));
  if(!promote)process.exitCode=1;
}catch(error){
  const failure={schema_version:1,promote:false,reasons:[String(error)]};
  try{writeFileSync(output,JSON.stringify(failure,null,2)+'\n');}catch{}
  console.error(String(error));
  process.exitCode=1;
}finally{
  if(temporary)rmSync(temporary,{recursive:true,force:true});
}
