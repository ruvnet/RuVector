"""Root-only, once-only DEV scoring after full three-arm forecast seal and independent physical closure.

This source executes exact pinned BEIR stdlib MRR/top-k-accuracy/Rcap functions.
It never loads a model, changes rankings or decodes labels outside selected16QIDs.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,time
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
KIND='REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT';ARMS=('native','exact_cosine','bm25')
OWNER=W/'owner.json';OWNER_SHA='a98d0e355925de7cf2fbe8711dc35afc4c3765e07fe5dfab8bd59f0309f733e5'
OFFICIAL_SCORER_SHA='3b315c1473115dc4e41d3038423fe5f1963868c3b9c92d5386839bde33472157'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(r):
 p=Path(r['path']);assert p.is_file()and not p.is_symlink()and ref(p)==r;return p

def parsed(r):return json.loads(check(r).read_bytes())
def put(p,v):
 b=(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
def percentile(xs,q):
 assert xs and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in xs);a=sorted(xs);p=(len(a)-1)*q;lo=int(math.floor(p));hi=int(math.ceil(p));return a[lo]+(a[hi]-a[lo])*(p-lo)
def fence():
 assert sha(OWNER)==OWNER_SHA;o=json.loads(OWNER.read_bytes());assert time.time()<o['expires_unix'] and not(W/'release.request').exists();p=Path('/proc')/str(o['pid']);assert int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==o['startticks'];s=(p/'fd/3').stat();assert(s.st_dev,s.st_ino)==(o['lock_device'],o['lock_inode']);assert 'FLOCK'in(p/'fdinfo/3').read_text()
def write_gold_progress(fd,value):
 b=(json.dumps(value)+'\n').encode();n=os.write(fd,b)
 if n!=len(b):raise RuntimeError('SHORT_GOLD_PROGRESS_WRITE')
 os.fsync(fd)
def load_selected_qrels(gold,selected,gold_progress):
  qrels={q:{}for q in selected};row_decodes=0;metadata_rows=0;selected_set=set(selected)
  with gold.open('rb')as f:
   assert f.readline().rstrip(b'\r\n')==b'query-id\tcorpus-id\tscore'
   for line in f:
    first,sep,remainder=line.rstrip(b'\r\n').partition(b'\t');assert sep;metadata_rows+=1;qid=first.decode('utf8')
    if qid not in selected_set:continue
    # Only these selected16 rows' remaining fields may be decoded, after seal+closure.
    write_gold_progress(gold_progress,{'query_id':qid,'event':'GOLD_FIELDS_DECODE_BEFORE','prior_completed_row_decodes':row_decodes})
    doc_bytes,sep,grade_bytes=remainder.partition(b'\t');assert sep and b'\t'not in grade_bytes;doc=doc_bytes.decode('utf8');grade=int(grade_bytes.decode('ascii'));assert doc and grade>=0
    if doc in qrels[qid]:assert qrels[qid][doc]==grade
    qrels[qid][doc]=grade;row_decodes+=1;write_gold_progress(gold_progress,{'query_id':qid,'event':'GOLD_FIELDS_DECODE_AFTER','completed_row_decodes':row_decodes})
  assert all(v and any(g>0 for g in v.values())for v in qrels.values())
  return qrels,row_decodes,metadata_rows
def main():
 if sys.flags.optimize!=0:raise ValueError('NONOPTIMIZED_PYTHON_REQUIRED')
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);a=p.parse_args();assert sorted(os.sched_getaffinity(0))==[0,1];cp=Path(a.config);assert sha(cp)==a.config_sha256;cfg=json.loads(cp.read_bytes());assert cfg['epoch']==57 and cfg['kind']==KIND
 fence();plan=parsed(cfg['plan']);assert sha(Path(__file__))==plan['scoring_source']['sha256'];raw=parsed(cfg['raw']);outer=parsed(cfg['outer']);cli=parsed(cfg['root_CLI_wall']);ind=parsed(cfg['independent_before_Gold_closure']);forecasts=parsed(cfg['forecasts']);selected=plan['selected_query_ids'];assert selected==sorted(set(selected))and len(selected)==16 and plan['metrics_ks']==[1,10]
 assert raw['epoch']==outer['epoch']==ind['epoch']==forecasts['epoch']==57 and raw['kind']==forecasts['kind']==KIND
 assert raw['actual_pilot_attempted']is True and raw['status']=='RETRIEVAL_DEVELOPMENT_FORECASTS_SEALED_NOT_SCORED'and raw['errors']==[]and raw['semantic_gold_decodes']==0
 assert raw['forecast_seal']==cfg['forecasts']and raw['plan_sha256']==cfg['plan']['sha256']==forecasts['plan_sha256']and raw['native_state']['accounting']['qualified']is True
 assert outer['checks_passed']is True and outer['all_owned_experiment_processes_physically_closed']is True and not outer['live_owned_processes']and outer['worker_exit_code']==0 and outer['within_600_seconds']is True
 assert cli['exit_code']==0 and cli['timed_out']is False and cli['within_600_seconds']is True
 assert ind['checks_passed']is True and ind['independent_review_complete']is True and ind['all_owned_processes_closed']is True and ind['forecasts_sha256']==cfg['forecasts']['sha256']and ind['raw_sha256']==cfg['raw']['sha256']
 assert forecasts['selected_query_ids']==selected==raw['actual_feature_query_ids']and forecasts['semantic_gold_decodes_before_seal']==0 and forecasts['candidate_promoted']is False and forecasts['used_confirmation_identities']==[]
 assert set(forecasts['rankings'])==set(ARMS) and len(forecasts['rows'])==48
 for arm in ARMS:
  assert set(forecasts['rankings'][arm])==set(selected)
  for qid,row in forecasts['rankings'][arm].items():assert len(row)==10 and sorted(row.values())==list(range(1,11))and all(type(d)is str for d in row)
 out=W/'private-run/retrieval_development';assert out.is_dir()and not(out/'SCORING_ATTEMPT.actual.v1.json').exists();put(out/'SCORING_ATTEMPT.actual.v1.json',{'epoch':57,'config_sha256':a.config_sha256,'forecasts_ref':cfg['forecasts'],'independent_physical_closure_ref':cfg['independent_before_Gold_closure'],'maximum_scoring_invocations':1,'selected_query_ids':selected,'model_native_paid_GPU_calls_authorized':0,'time_unix':time.time()})
 started=time.monotonic();gold_progress=os.open(out/'SCORING_GOLD_PROGRESS.actual.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);net={'attempted':0}
 def no_network(event,args):
  if event.startswith('socket.'):
   net['attempted']+=1;raise RuntimeError('DEV_SCORER_NETWORK_FORBIDDEN')
 sys.addaudithook(no_network)
 scorer=check(plan['official_metrics_source']);assert sha(scorer)==OFFICIAL_SCORER_SHA;spec=importlib.util.spec_from_file_location('frozen_official_beir_custom_metrics',scorer);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 gold=check(plan['qrels_asset_ref']);assert gold==check(cfg['qrels_asset_ref']) and gold.stat().st_size==5389
 qrels,row_decodes,metadata_rows=load_selected_qrels(gold,selected,gold_progress);os.close(gold_progress);fence()
 corpus_file=check(plan['corpus_features']);corpus_ids={json.loads(line)['_id']for line in corpus_file.read_text().splitlines()};assert len(corpus_ids)==5183 and all(doc in corpus_ids for rows in qrels.values()for doc in rows)
 metrics={};latency={}
 for arm in ARMS:
  ranks=forecasts['rankings'][arm];metrics[arm]={**module.mrr(qrels,ranks,[1,10]),**module.top_k_accuracy(qrels,ranks,[1,10]),**module.recall_cap(qrels,ranks,[1,10])};xs=[r['query_search_seconds']for r in forecasts['rows']if r['arm']==arm];assert len(xs)==16;latency[arm]={'observations':16,'p50_seconds':percentile(xs,.5),'p95_seconds':percentile(xs,.95)}
 exact=forecasts['rankings']['exact_cosine'];native=forecasts['rankings']['native'];agreements={qid:len(set(native[qid])&set(exact[qid]))/10 for qid in selected};ann_recall=sum(agreements.values())/16
 assert net['attempted']==0 and not any(n in sys.modules for n in ['torch','transformers','onnxruntime','tokenizers'])
 result={'schema_version':1,'epoch':57,'kind':KIND,'status':'RETRIEVAL_DEV_PILOT_SCORED_NOT_OFFICIAL_BENCHMARK_PLACED','actual_pilot_attempted':True,'benchmark_generalization_claim':False,'candidate_promoted':False,'used_confirmation_identities':[],'identity_namespace':plan['identity_namespace'],'reserved_development_query_ids':selected,'actual_feature_query_ids':raw['actual_feature_query_ids'],'actual_scored_query_ids':selected,'label_payload_exposed_query_ids':selected,'label_only_exposed_query_ids':[],'semantic_gold_query_decodes':16,'selected_qrel_row_label_decodes':row_decodes,'query_ID_metadata_rows_read':metadata_rows,'unselected_qrel_documentID_or_relevance_decodes':0,'train_or_calibration_or_fresh_confirmation_queries':[],'metrics':metrics,'exact_scorer':plan['official_metrics_source'],'metrics_ks':[1,10],'metric_scope':'ExactpinnedBEIRcustom_metrics functions on a16queryPROJECT_DEV subset; nocoreEvaluateRetrieval/nDCG/pytrec_evalorfullbenchmarkplacementclaim. No semanticNLIaccuracyorproperclassificationloss.','self_ID_filter':False,'metric_input_score_encoding':'Ordinalrankcodes10..1 reflectfrozenranks; notprobabilities orinterpretedrawnativecosinedistances.','tie_policy':forecasts['tie_policy'],'ANN_top10_set_recall_against_exact_same_F32_vectors':{'per_query':agreements,'mean':ann_recall,'boundary_tie_uncertainty':'Nativeonlyreturnedtop10tiescanonicalized;outsidecutofftiesunverified.'},'per_arm_query_search_latency':latency,'percentile_definition':'Linearinterpolation q*(N-1),N16perarm. SharedBGEencoding/indexbuildandCPUinitializationexcluded; inclusiveRootCLIwall reportedseparately.','classification_quality_metrics':{'accuracy':None,'Brier':None,'NLL':None},'actual_execution_raw_counters':raw['counters'],'native_accounting':raw['native_state']['accounting'],'resources':raw['resources'],'Root_CLI_wall':cli,'outer_execution':cfg['outer'],'raw_forecast_execution':cfg['raw'],'sealed_forecasts':cfg['forecasts'],'independent_pre_Gold_closure':cfg['independent_before_Gold_closure'],'frozen_plan':cfg['plan'],'qrels_asset_ref':plan['qrels_asset_ref'],'Gold_scoring_calls':1,'model_native_training_calibration_paid_GPU_calls_in_scorer':0,'network_audit':net,'scoring_wall_seconds':time.monotonic()-started,'official_goal_verified':False,'official_rank':None,'full_benchmark_placement':None,'model_training_test_nonuse_lineage':'UNQUALIFIED; projectDEVonly, cannotpromotechampionorclaimfreshconfirmation.'};output=put(out/'scored-results.json',result);print(json.dumps({'results':output,'selected_gold_queries':16,'label_only_exposed_query_ids':[],'native_batches_in_scorer':0}))
if __name__=='__main__':main()
