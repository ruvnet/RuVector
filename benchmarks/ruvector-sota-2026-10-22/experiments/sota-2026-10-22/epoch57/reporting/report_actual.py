"""E57 metadata-only actual reporting and immutable closure; no scientific calls.

Independent reports supply observations. Frozen plans never fill missing actuals.
"""
from pathlib import Path
import copy,hashlib,json,os,sys
W=Path(__file__).resolve().parent;R=W/'project'
P='experiments/sota-2026-10-22/epoch57/';A='experiments/sota-2026-10-22/audit/epoch57/'
ROOTS=(P,A,'experiments/sota-2026-10-22/runtime/epoch57/')
BASE='5e426edd8ada8c0fdcec38ec8807054affc89e3c'
KIND='REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV'
def encoded(v):return(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def ref(p,project=False):
 b=p.read_bytes();return{'path':p.relative_to(R).as_posix()if project else str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checked(row,project=False):
 p=R/row['path']if project else Path(row['path']);assert p.is_file()and not p.is_symlink();b=p.read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];return json.loads(b)
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);b=encoded(v)
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p,p.is_relative_to(R))
def independent(row):
 v=checked(row,True);assert v['epoch']==57 and v['independent_review_complete']is True and v['all_owned_processes_closed']is True
 assert type(v['actual_execution_independently_derived'])is dict
 return v

def summarize(c):
 old=checked(c['initial_CONT']);src=checked(c['source_receipt'],True);e=independent(c['engineering_audit'])
 assert src['epoch']==57 and src['phase']=='source'and src['parent']==BASE and src['verified_actual_full_tree']is True
 assert type(e['engineering_qualified'])is bool and type(e['throughput_gate_passed'])is bool
 assert type(e['native_counter_accounting'])is dict
 assert e['actual_execution_independently_derived'].get('paid_API_calls')==0 and e['actual_execution_independently_derived'].get('GPU_calls')==0
 raw=checked(c['engineering_result'],True);assert raw['epoch']==57 and e['actual_results_ref']==c['engineering_result']
 pil=None;pa=None;pilotrefs=c.get('retrieval_DEV_pilot');newfeature=[];newscored=[];labelonly=[];namespace=None
 if pilotrefs is not None:
  assert set(pilotrefs)=={'results','independent_acceptance','frozen_plan'}
  pil=checked(pilotrefs['results'],True);pa=independent(pilotrefs['independent_acceptance']);plan=checked(pilotrefs['frozen_plan'],True)
  assert pil['epoch']==plan['epoch']==57 and pil['actual_pilot_attempted']is True
  assert pa['actual_results_ref']==pilotrefs['results'] and pil['candidate_promoted']is False and pil['benchmark_generalization_claim']is False and pil['used_confirmation_identities']==[]
  assert pa['actual_execution_independently_derived'].get('paid_API_calls')==0 and pa['actual_execution_independently_derived'].get('GPU_calls')==0
  assert pil['reserved_development_query_ids']==plan['selected_query_ids']
  namespace=pil['identity_namespace'];assert namespace==plan['identity_namespace'] and type(namespace)is str
  newfeature=pil['actual_feature_query_ids'];newscored=pil['actual_scored_query_ids'];labelonly=pil['label_only_exposed_query_ids']
  assert len(set(newfeature))==len(newfeature)and set(newfeature)<=set(plan['selected_query_ids'])
  assert len(set(newscored))==len(newscored)and set(newscored)<=set(newfeature)
  assert len(set(labelonly))==len(labelonly)and all(type(x)is str for x in labelonly)
 else:
  no=checked(c['retrieval_nonexecution_review'],True)
  assert no['epoch']==57 and no['independent_review_complete']is True and no['retrieval_CLI_invocations']==0 and no['Gold_scoring_calls']==0 and no['all_owned_processes_closed']is True
 common={'schema_version':1,'epoch':57,'kind':KIND,'source_checkpoint':BASE,'actual_SOURCE_commit':src['commit'],'native104_champion_preserved':True,'candidate_promoted':False,'official_goal_verified':False,'official_rank':None,'benchmark_generalization_claim':False,'new_submission':False,'used_confirmation_identities':[]}
 status=e['status'];reason=e['reason']
 if pil is not None and not pa.get('pilot_completed',False):status='ENGINEERING_'+('QUALIFIED'if e['engineering_qualified']else'REJECTED')+'_RETRIEVAL_DEV_PILOT_REJECTED_PARTIAL'
 if pil is not None and pa.get('pilot_completed',False):status='ENGINEERING_'+('QUALIFIED'if e['engineering_qualified']else'REJECTED')+'_RETRIEVAL_DEV_PILOT_COMPLETED_UNPROMOTED'
 result={**common,'status':status,'engineering_qualified':e['engineering_qualified'],'throughput_gate_passed':e['throughput_gate_passed'],'independent_reason':reason,'engineering_result':c['engineering_result'],'engineering_audit':c['engineering_audit'],'actual_engineering_execution':e['actual_execution_independently_derived'],'native_counter_accounting':e['native_counter_accounting'],'engineering_latency_and_resources':e['latency_and_resources'],'engineering_throughput_measurements':e['throughput_measurements'],'retrieval_DEV_pilot':pilotrefs,'retrieval_nonexecution_review':c.get('retrieval_nonexecution_review'),'actual_retrieval_execution':pa['actual_execution_independently_derived']if pa else None,'quality_metrics':{'classification_accuracy':None,'Brier':None,'NLL':None},'metric_scope':'Authored engineering speed/numerical/native fidelity is not benchmark quality. A separately frozen actual SciFact PROJECT_DEV pilot may have relevance metrics only; no typed NLI classification, fresh confirmation or official rank is measured.','model_and_data_qualification':c['qualification'],'official_refresh':c['official_refresh'],'source_frozen_metadata_reporter_executed':True,'all_epoch57_scientific_slots_closed':True,'pre_science_preparation':c['pre_science_preparation']}
 used={**common,'roles':{'train':[],'development':sorted(set(namespace+':'+x for x in(newfeature+labelonly)))if namespace else[],'calibration':[],'fresh_confirmation':[]},'engineering_identities':c['engineering_identities'],'engineering_identity_scope':'New authored CC0 inputs; engineering only, never benchmark or confirmation.','actual_scored_DEV_identities':sorted(namespace+':'+x for x in newscored)if namespace else[],'actual_label_only_exposed_DEV_identities':sorted(namespace+':'+x for x in labelonly)if namespace else[],'inherited_used_identities':old['epoch56_final']['used_identities'],'inherited_used_confirmation':old['epoch56_final']['used_confirmation'],'all_previous_roles_and_protected_confirmation_unchanged':True,'all_SciFact16_prepared_DEV_and1109_pool_freshness_caveat_inherited':True,'existing_confirmation_status':'UNQUALIFIED_HOLD_UNUSED','confirmation_payloads_features_or_labels_read':False}
 priority=('Preserve SubmittedNative104. Retain the exact E57 engineering throughput/numerical/native verdict and any separately audited retrieval DEV result, including every failure and pending operation uncertainty. Same INT8 vectors must feed native and exact cosine; FP32 is only a separate engineering throughput/drift control. Authored tests and public DEV relevance cannot establish benchmark generalization, NLI accuracy, fresh confirmation or official placement. Do not repeat an unchanged rejected runtime/time plan or exposed-question grid. Any next attempt requires a new source/identity/resource freeze and credible whole-epoch time reserve. Stronger typed NLI remains the quality priority when licensed aligned data/scorer and exact model/training-test provenance are available; do not repeat unchanged FinNLI author-access or ModernBERT lineage probes. All easy48/Legal96 and prepared SciFact16 are DEV forever; the entire1109 query pool remains unqualified for fresh confirmation. Existing confirmation stays UNQUALIFIED_HOLD and unused. No paid API/GPU authorization or publisher submission follows; official success requires the matching publisher result/rank and eligibility.')
 if not e['throughput_gate_passed']:priority+=' E57 engineering did not admit the full-corpus throughput gate; do not dispatch or retry the pilot under this unchanged plan.'
 if pa is not None and not pa.get('pilot_completed',False):priority+=' E57 actual pilot rejected/partial: '+str(pa['reason'])
 learning={**common,'status':status,'engineering_qualified':e['engineering_qualified'],'throughput_gate_passed':e['throughput_gate_passed'],'next_research_priority':priority,'results_scope':result['metric_scope'],'engineering_audit':c['engineering_audit'],'retrieval_DEV_pilot':pilotrefs}
 ledger=copy.deepcopy(checked(c['initial_rejection_ledger'],True));assert not any(x['epoch']==57 for x in ledger['rejections'])
 rr={'path':P+'RESULTS.json','bytes':len(encoded(result)),'sha256':hashlib.sha256(encoded(result)).hexdigest()}
 ledger['rejections'].append({**common,'status':status,'reason':reason,'engineering_qualified':e['engineering_qualified'],'throughput_gate_passed':e['throughput_gate_passed'],'results':rr,'independent_acceptance':c['engineering_audit'],'retrieval_DEV_pilot':pilotrefs,'actual_execution_independently_derived':e['actual_execution_independently_derived']})
 outputs=[put(R/(P+'RESULTS.json'),result),put(R/(P+'USED_IDENTITIES.json'),used),put(R/(P+'USED_CONFIRMATION.json'),used),put(R/(P+'LEARNINGS.json'),learning),put(R/(P+'REJECTION_LEDGER.json'),ledger)]
 assert outputs[0]==rr
 return{'outputs':outputs,'status':status,'results':outputs[0],'learning':outputs[3],'ledger':outputs[4],'next_research_priority':priority}

def close(c):
 a=checked(c['independent_public_review']);assert a['epoch']==57 and a['checks_passed']is True and a['public_inventory_review_passed']is True
 files=sorted([ref(p,True)for root in ROOTS for p in(R/root).rglob('*')if p.is_file()],key=lambda r:r['path'])
 assert files==a['files']
 target=R/(A+'FINAL_PUBLIC_SUMMARY_AND_INVENTORY_INDEPENDENT_REVIEW.v1.json');source=Path(c['independent_public_review']['path']);target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb')as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
 files=sorted([*files,ref(target,True)],key=lambda r:r['path'])
 return put(R/(P+'EPOCH_MANIFEST.json'),{'schema_version':1,'epoch':57,'kind':KIND,'source_checkpoint':BASE,'files':files,'no_project_additions_after_manifest':True,'used_confirmation_identities':[],'official_goal_verified':False,'independent_public_review':ref(target,True)})
if __name__=='__main__':
 mode,path=sys.argv[1:];cfg=json.loads(Path(path).read_bytes());assert mode in('summarize','close');print(json.dumps(summarize(cfg)if mode=='summarize'else close(cfg),indent=2,sort_keys=True))
