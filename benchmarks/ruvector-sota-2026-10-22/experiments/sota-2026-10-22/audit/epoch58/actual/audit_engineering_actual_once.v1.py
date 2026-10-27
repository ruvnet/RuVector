"""Post-run read-only independent actual engineering audit; no model/native execution."""
import json,hashlib,os,math,base64,struct,importlib.util,statistics
from pathlib import Path
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');R=W/'project';Q=Path('/dev/shm/ruvector-e58-audit');C=Path('/dev/shm/ruvector-e58-candidate')
def load(p):return json.loads(Path(p).read_bytes())
def pin(p):b=Path(p).read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checked(r):assert pin(r['path'])==r;return load(r['path'])
def closed(x):
 p=Path('/proc')/str(x['pid'])
 if not p.exists():return True
 try:
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  if int(f[19])!=x['startticks']:return True
  return f[0] in ('Z','X') and not list((p/'fd').iterdir()) and not any(int(y) for y in (p/'statm').read_text().split())
 except FileNotFoundError:return True
rawref=pin(W/'engineering-worker/RESULT.actual.json');raw=checked(rawref);outer=load(W/'engineering-cli/OUTER_RESULT.actual.json');sup=load(W/'engineering-cli/SUPERVISOR_RESULT.actual.json')
assert raw['phase']=='engineering' and raw['errors']==[] and outer['errors']==[] and sup['errors']==[] and outer['supervisor_exit_code']==0 and sup['worker_exit_code']==0
assert outer['inclusive_limit_passed'] is True and outer['wall_seconds_inclusive']<=600 and sup['children_physically_closed'] is True
identities={(x['pid'],x['startticks']):x for x in outer['owned_process_identities']+sup['owned_process_identities']+raw['owned_process_identities']};assert all(closed(x) for x in identities.values())
groups={(x['pgid'],x['sid']) for x in identities.values()}
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   f=(p/'stat').read_text().rsplit(')',1)[1].split()
   if (int(f[2]),int(f[3])) in groups:assert f[0] in ('Z','X') and not list((p/'fd').iterdir()) and not any(int(y) for y in (p/'statm').read_text().split())
  except (FileNotFoundError,ProcessLookupError):pass
assert outer['kernel_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and outer['PID_namespace']==os.readlink('/proc/self/ns/pid')
helper=Q/'derive_actual_phase_events.v1.py';assert pin(helper)['sha256']=='8861d7b2ed861113ba210d807d00a2267ca0b907a150b74673579826981e0376'
sp=importlib.util.spec_from_file_location('derive',helper);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);schema=load(C/'COUNTER_SCHEMA.v1.json');events=m.derive(raw,schema);counts=events['actual_execution_independently_derived']
assert all(counts[k]==0 for k in ['paid_API_calls','GPU_calls','Gold_labels_decoded','Python_network_operations_attempted','Node_network_operations_attempted'])
native=checked(raw['native_results_ref']);state=raw['native_adapter_state'];bundle=load(W/'engineering-worker/native/native-bundle.json') if (W/'engineering-worker/native/native-bundle.json').exists() else None
# Discover exact native IO bundle from immutable once-only NATIVE_ATTEMPT binding.
attempts=list((W/'engineering-worker').rglob('NATIVE_ATTEMPT.json'));assert len(attempts)==1;attempt=load(attempts[0]);bundle=checked(attempt['bundle_ref']);nativeconf=checked(attempt['config_ref'])
assert native['bundle_sha256']==attempt['bundle_ref']['sha256'] and native['native_config_sha256']==attempt['config_ref']['sha256'] and native['errors']==[] and native['stage']=='COMPLETE'
assert native['native_binary_sha256']=='ff559e9fac2e3d29a8b32e890fdb047bcd4bd28f389b4731ad72148b221d5a2e' and native['binary_version']=='2.3.0' and native['package_version']=='0.1.33' and native['platform_package_version']=='0.1.31'
def vectors(x):
 z=x['vectors'];b=base64.b64decode(z['data_base64'],validate=True);assert len(b)==z['bytes'] and hashlib.sha256(b).hexdigest()==z['sha256'];v=list(struct.iter_unpack('<384f',b));assert len(v)==len(x['ids']);assert all(all(math.isfinite(y) for y in r) and abs(sum(y*y for y in r)-1)<=1e-4 for r in v);return b,v
cb,cv=vectors(bundle['corpus']);qb,qv=vectors(bundle['queries']);assert len(cv)==128 and len(qv)==16
sourceids=bundle['corpus']['ids'];sidmap={x:i for i,x in enumerate(sourceids)};assert [x['query_id'] for x in native['queries']]==bundle['queries']['ids']
hits=0
for i,row in enumerate(native['queries']):
 assert len(row['raw_hits'])==len(row['ids']) and len(set(row['ids']))==len(row['ids']) and row['requested_k']==128
 for h in row['raw_hits']:
  j=sidmap[h['id']];assert h['vector_receipt']['sha256']==hashlib.sha256(cb[j*1536:(j+1)*1536]).hexdigest() and h['vector_receipt']['source_equal_before_stripping'] is True;hits+=1
 assert row['missing_source_ids']==[x for x in sourceids if x not in row['ids']]
 for d in row['exact_cosine_diagnostic']:
  v=cv[sidmap[d['id']]];u=qv[i];cos=math.fsum(x*y for x,y in zip(v,u))/math.sqrt(math.fsum(x*x for x in v)*math.fsum(y*y for y in u));assert abs(cos-d['cosine'])<=1e-10
assert hits==native['observedHits']==counts['native_hit_source_validations']
t=raw['throughput_measurements'];f=t['full8_NLI_forward_seconds'];assert len(f)==4 and all(type(x) is float and x>0 and math.isfinite(x) for x in f)
observed=[x for x in [json.loads(x) for x in Path(raw['progress_ref']['path']).read_bytes().splitlines()] if x['event']=='NLI_FORWARD_OBSERVED'];assert [x['forward_seconds'] for x in observed]==f and all(x['batch_size']==8 and x['sequence_length']==512 for x in observed)
assert len(raw['engineering_pairs'])==counts['NLI_pair_inferences_completed']==32 and counts['NLI_forward_calls_completed']==4
for row in raw['engineering_pairs']:
 l=row['logits'];assert len(l)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in l);normal=max(l)+math.log(math.fsum(math.exp(x-max(l)) for x in l));logs=[x-normal for x in l];probs=[math.exp(x) for x in logs];assert all(abs(x-y)<=1e-12 for x,y in zip(logs,row['log_probabilities'])) and all(abs(x-y)<=1e-12 for x,y in zip(probs,row['probabilities']));assert row['prediction']==['Contradiction','Entailment','NotMentioned'][max(range(3),key=lambda i:l[i])]
projection=2*math.ceil(3*17*9/8)*max(f)+t['measured_modelinit_total']+90;assert projection==t['projection_seconds'];gate=projection<=420
public='experiments/sota-2026-10-22/epoch58/actual/engineering/RESULT.actual.json';assert (R/public).read_bytes()==Path(rawref['path']).read_bytes();publicref={'path':public,'bytes':rawref['bytes'],'sha256':rawref['sha256']}
proof={'schema_version':1,'epoch':58,'kind':raw['kind'],'phase':'engineering','checks_passed':True,'independent_review_complete':True,'all_owned_processes_closed':True,'actual_results_ref':publicref,'private_actual_results_origin_ref':rawref,'actual_execution_independently_derived':counts,'worker_event_derivation':events,'engineering_qualified':gate,'throughput_gate_passed':gate,'projection_seconds':projection,'plan_ref':load(C/'ENGINEERING_CONFIGURATION_FIXED_FIELDS.proposal.v1.json')['worker_fixed_fields']['plan'],'status':'ENGINEERING_QUALIFIED_FOR_ONE_FROZEN_DEV_BATCH' if gate else 'REJECTED_FROZEN_THROUGHPUT_GATE','reason':'Actual full8 fixed512 NLI forwards and real native IO validated; conservative projection <=420.' if gate else 'Conservative frozen projection exceeds420; no DEV batch or Gold authorized.','throughput_measurements':t,'latency_and_resources':raw['latency_and_resources'],'native_counter_accounting':{'actual_native_version':native['binary_version'],'actual_source_vectors':len(cv),'actual_queries':len(qv),'actual_hits_revalidated':hits,'cosine_diagnostics_recomputed_without_replacing_native_order':True,'native_query_latency_seconds':native['query_latency_seconds'],'native_resources':native['resources']},'physical_process_closure':{'identities':list(identities.values()),'closed_PID_PGID_SID_pairs':[list(x) for x in groups],'kernel_boot_id':outer['kernel_boot_id'],'PID_namespace':outer['PID_namespace'],'outer_result_ref':pin(W/'engineering-cli/OUTER_RESULT.actual.json'),'supervisor_result_ref':pin(W/'engineering-cli/SUPERVISOR_RESULT.actual.json'),'whole_CLI_seconds_inclusive':outer['wall_seconds_inclusive']},'accuracy':None,'Brier':None,'NLL':None,'benchmark_generalization_measured':False,'fresh_confirmation_ids':[],'confirmation_status':'UNQUALIFIED_HOLD','champion_preserved':True}
p=Q/'ENGINEERING_ACTUAL_INDEPENDENT_AUDIT.v1.json';b=(json.dumps(proof,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,b);os.fsync(fd);os.close(fd);print(json.dumps(pin(p)))
