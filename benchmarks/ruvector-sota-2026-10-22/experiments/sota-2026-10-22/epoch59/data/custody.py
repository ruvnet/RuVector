"""E59 originalTRAIN473 as PROJECT_DEV: blind custody and strictly late Gold.
No fitting, officialDEV/test generalization, confirmation or promotion claim.

DATA decodes exactly one selected NDA and the17 public hypothesis strings.
Gold choices are copied as opaque JSON-token bytes, never interpreted in DATA.
No model, tokenizer, native, network, training or confirmation entrypoint exists.
"""
import argparse,base64,fcntl,hashlib,importlib.util,io,json,math,os,re,stat,time,traceback,unicodedata,zipfile
from pathlib import Path
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983');R=W/'project';PRIVATE=W/'private-data'
BASE='9b677a91698352bd563f189fee861d6ff2a1ad3a'
P='experiments/sota-2026-10-22/epoch59/data/'
OPERATIONAL=Path('/dev/shm/ruvector-e59-data/data-v1')
OWNER={'path':str(W/'guardian27/owner.json'),'bytes':831,'sha256':'28f34b4b0af71fbd39d94a05766ba7bf61798ae540419d1838ea3ca21d5cce9f'}
OWNER_GENERATION=27
OWNER_PID=198313
OWNER_STARTTICKS=46302984
OWNER_FENCE={'path':str(W/'guardian27/owner_fence.py'),'bytes':3376,'sha256':'cc0f6f56e14998820dfad60752e3579ce09f1b89deae0ef6c57e47cdbe1dbbc5'}
ARCHIVE_BYTES=65362913;ARCHIVE_SHA='e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b';ARCHIVE_GIT='757fd1dafd29a997fba00c60c6d40b2930a36159'
MEMBER='contract-nli/train.json';MEMBER_BYTES=7608211;MEMBER_SHA='dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5'
DOC_SHA='d9c341c8757f6616004a579576c4ae29e64f7346a0bf4e98c182ad431730fd37';FILENAME_SHA='f2c24152484c19386c32caac5c3a1f593fdecff2d4a3534c95a4d65b05f60406'
HYPOTHESES=('nda-1','nda-10','nda-11','nda-12','nda-13','nda-15','nda-16','nda-17','nda-18','nda-19','nda-2','nda-20','nda-3','nda-4','nda-5','nda-7','nda-8')
IDS=tuple('contractnli:eced6528:train:473:'+h for h in HYPOTHESES)
LABELS=('Contradiction','Entailment','NotMentioned');ARMS=('native_order','source_order','full_reference')
PROJECT_ROLE_SEMANTICS='PROJECT_DEV_DIAGNOSTIC_NO_FITTING_NO_CONFIRMATION'
FILENAME_QUALIFICATION_REF={'path':str(W/'private-identity/FILENAME_QUALIFICATION.actual.v1.json'),'bytes':2358,'sha256':'1fbd84cd2d8a9b42d16e041602640853064458c78e03eb5d67e43d0eb6f078e9'}
CODEC_SHA='9c63102a01576c230da94695f593fd7bd4fe2d0b20672b74e8a7f62aba426fc4'
COUNTER_SCHEMA_SHA='a5480f7fbaf1fa06a6955702bc7655813ef426e9600a3113f8a3792b48f83960'
def need(v,code):
 if not v:raise ValueError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def unique(rows):
 out={}
 for k,v in rows:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
 return out
def parse(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in()).throw(ValueError('NONFINITE_JSON')))
def safe(path):
 p=Path(path);need(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in[p,*p.parents]),'CANONICAL_NON_SYMLINK_PATH');return p
def body(path):
 p=safe(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb')as f:a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode),'REGULAR_FILE');b=f.read();z=os.fstat(f.fileno())
 need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'STABLE_FILE');return b
def ref(path):
 p=safe(path);b=body(p);return{'path':str(p),'bytes':len(b),'sha256':sha(b)}
def read_ref(r):
 need(type(r)is dict and set(r)=={'path','bytes','sha256'}and type(r['bytes'])is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256'])is not None,'EXACT_TYPED_REF');b=body(r['path']);need(len(b)==r['bytes']and sha(b)==r['sha256'],'REF_BYTES_SHA');return b
def load_ref(r):return parse(read_ref(r))
def put(path,value):
 p=safe(path);p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d);return ref(p)
def owner_guard():
 read_ref(OWNER_FENCE);module=imported(OWNER_FENCE['path'],'e59_shared_owner27_fence');return module.verify_owner(OWNER)
def imported(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def admission(path,expected,mode):
 need(mode in('DATA','GOLD'),'ONLY_DEV_DATA_OR_GOLD');r=ref(path);need(r['sha256']==expected,'EXACT_ROOT_ADMISSION');a=load_ref(r)
 need(a['schema_version']==1 and a['epoch']==59 and a['mode']==mode and a['allowed_ids']==list(IDS)and a['owner_ref']==OWNER,'FROZEN_DOC473_ALL17_DEV_ONLY')
 need(a['train_ids']==[]and a['calibration_ids']==[]and a['fresh_confirmation_ids']==[]and a['confirmation_materialization_authorized']is False,'EMPTY_TRAIN_CAL_CF_HOLD')
 owner_guard();plan=load_ref(a['plan_ref']);f=load_ref(a['source_freeze_ref'])
 need(a['filename_qualification_ref']==FILENAME_QUALIFICATION_REF,'EXACT_ACTUAL_FILENAME_QUALIFICATION_REF');fq=load_ref(a['filename_qualification_ref']);need(fq['checks_passed']is True and fq['status']=='READY_TRAIN473_FILENAME_ALIAS_QUALIFIED_NO_FEATURES'and fq['selected_document_id']==473 and fq['source_split']=='train'and fq['selected_document_filename_key_sha256']==FILENAME_SHA and fq['source_member_sha256']==MEMBER_SHA and fq['source_member_bytes']==MEMBER_BYTES and fq['source_text_decodes']==fq['hypothesis_value_decodes']==fq['semantic_Gold_decodes']==0,'ACTUAL_FILENAME_ONLY_QUALIFICATION_REQUIRED')
 fqr=load_ref(a['independent_filename_review_ref']);need(fqr['checks_passed']is True and fqr['actual_qualification_report_ref']==FILENAME_QUALIFICATION_REF,'INDEPENDENT_ACTUAL_FILENAME_QUALIFICATION_REQUIRED')
 need(plan['epoch']==59 and plan['kind']==a['kind']and plan['source_checkpoint']==BASE,'EXACT_PLAN_SOURCE')
 files={r['path']:r for r in f['files']};need(len(files)==len(f['files']),'UNIQUE_SOURCE_FILES')
 need(Path(__file__).resolve()==OPERATIONAL/'custody.py','EXACT_OPERATIONAL_CUSTODY_LOCATION')
 need(sha(body(OPERATIONAL/'official_codec.source.py'))==CODEC_SHA,'EXACT_OFFICIAL_CODEC')
 if mode=='DATA':
  need(a['phase']=='BLIND_CATALOG_PREPARATION_ONLY'and plan['phase']==a['phase'],'LOCAL_BLIND_PREPARATION_ONLY')
  q=load_ref(a['independent_preparation_approval_ref']);need(q['checks_passed']is True and q['phase']==a['phase']and q['source_checkpoint']==BASE and q['owner_ref']==OWNER and q['preparation_plan_sha256']==a['plan_ref']['sha256']and q['preparation_source_freeze_sha256']==a['source_freeze_ref']['sha256'],'EXACT_INDEPENDENT_LOCAL_PREPARATION_SOURCE_APPROVAL')
  for name in['custody.py','span_json.py','source_catalog.py','official_codec.source.py','DATA_PROTOCOL.v1.json','COUNTER_SCHEMA.v1.json']:
   q=ref(OPERATIONAL/name);need(files.get(q['path'])==q,'PREPARATION_HELPER_IN_FROZEN_LOCAL_SOURCE')
  need(files.get(OWNER_FENCE['path'])==OWNER_FENCE,'SHARED_OWNER_FENCE_IN_PREPARATION_FREEZE')
  need(files.get(a['plan_ref']['path'])==a['plan_ref'],'PREPARATION_PLAN_IN_LOCAL_FREEZE')
  return a,r
 need(a['phase']=='dev','ONLY_LATE_DEV_GOLD')
 pub=load_ref(a['source_publication_ref']);need(pub['checks_passed']is True and pub['verified_actual_remote_readback']is True and pub['actual_parent']==BASE and pub['commit']!=BASE and re.fullmatch('[0-9a-f]{40}',pub['commit'])is not None,'ACTUAL_INDEPENDENT_FULL_SOURCE_BEFORE_MEASUREMENTS_AND_GOLD')
 need(pub['source_plan_sha256']==a['plan_ref']['sha256']and pub['source_freeze_sha256']==a['source_freeze_ref']['sha256'],'EXACT_FULL_SOURCE_PLAN_FREEZE_BINDING')
 for name in['custody.py','span_json.py','source_catalog.py','official_codec.source.py','COUNTER_SCHEMA.v1.json']:
  q=ref(OPERATIONAL/name);need(files.get(P+name)=={'path':P+name,'bytes':q['bytes'],'sha256':q['sha256']}and body(R/P/name)==read_ref(q),'CUSTODY_HELPER_IN_FROZEN_PUBLISHED_SOURCE')
 need(sha(body(OPERATIONAL/'official_codec.source.py'))==CODEC_SHA,'EXACT_OFFICIAL_CODEC')
 need(files.get(str(Path(a['plan_ref']['path']).relative_to(R)))=={'path':str(Path(a['plan_ref']['path']).relative_to(R)),'bytes':a['plan_ref']['bytes'],'sha256':a['plan_ref']['sha256']},'PLAN_IN_FREEZE')
 return a,r
def extract(member,span_module,catalog,progress,before_document_decode=None,after_document_decode=None):
 need(len(member)==MEMBER_BYTES and sha(member)==MEMBER_SHA,'EXACT_DEV_MEMBER')
 s=span_module.SpanJSON(member);progress['value_decoder_counts']=s.decoded;root=s.value(0);need(root[2]=='object'and s.ws(root[1])==len(member),'ONE_SOURCE_OBJECT');top=s.members(root);labels=s.members(top['labels']);need(tuple(sorted(labels))==HYPOTHESES,'EXACT_OFFICIAL17_HYPOTHESIS_CATALOG')
 claims={}
 for cid,h in zip(IDS,HYPOTHESES):
  claims[h]=s.decode_string(s.members(labels[h])['hypothesis'],'selected_hypothesis');progress['actual_hypothesis_exposed_case_ids'].append(cid)
 need(all(x.strip()and len(x.encode())<=65536 for x in claims.values()),'BOUNDED_ORIGINAL_CLAIMS')
 seen=set();selected=None;tokens=[]
 for ordinal,entry in enumerate(s.items(top['documents'])):
  fields=s.members(entry);need(fields['id'][2]=='number','INTEGER_ID');did=parse(s.raw(fields['id']));need(type(did)is int and did not in seen,'UNIQUE_SOURCE_ID');seen.add(did)
  if did!=473:continue
  name=s.decode_string(fields['file_name'],'metadata');key=' '.join(unicodedata.normalize('NFKC',name.replace('\\','/').rsplit('/',1)[-1]).casefold().split());actual_filename=sha(key.encode());progress['selected_document_filename_key_sha256']=actual_filename;need(actual_filename==FILENAME_SHA,'EXACT_DOC473_FILENAME_ALIAS')
  progress['source_text_exposed_case_ids']=list(IDS)
  if before_document_decode is not None:before_document_decode()
  text=s.decode_string(fields['text'],'selected_document')
  if after_document_decode is not None:after_document_decode()
  actual_normalized=sha(' '.join(text.casefold().split()).encode());progress['selected_normalized_document_sha256']=actual_normalized;need(actual_normalized==DOC_SHA,'EXACT_DOC473_NORMALIZED_SOURCE');units=catalog.sentence_catalog(text);catalog.validate_spans(units);need(''.join(x['sentence']for x in units)==text,'EXACT_SOURCE_COVERAGE')
  sets=list(s.items(fields['annotation_sets']));need(len(sets)==1,'ONE_ANNOTATION_SET');annotations=s.members(s.members(sets[0])['annotations']);need(set(annotations)==set(HYPOTHESES),'EXACT17_ANNOTATIONS')
  for h in HYPOTHESES:
   token=s.members(annotations[h])['choice'];need(token[2]=='string','OPAQUE_CHOICE_STRING_TOKEN');tokens.append(base64.b64encode(s.raw(token)).decode('ascii'))
  selected={'document_id':473,'record_ordinal':ordinal,'source_group_sha256':sha(('ContractNLI|'+ARCHIVE_SHA+'|train|473').encode()),'source_document_sha256':sha(text.encode()),'normalized_document_sha256':actual_normalized,'source_filename_key_sha256':actual_filename,'evidence':units}
 need(len(seen)==423 and selected is not None and len(tokens)==17 and s.decoded['selected_document']==1 and s.decoded['selected_hypothesis']==17 and s.gold_values_decoded==0,'EXACT_ONE_DOC473_17_CLAIMS_NO_GOLD')
 text=''.join(u['sentence']for u in selected['evidence']);cases=[{'id':cid,'document_id':473,'hypothesis_id':h,'claim':claims[h],'input_sha256':sha(canonical({'claim':claims[h],'document':text}))}for cid,h in zip(IDS,HYPOTHESES)]
 return selected,cases,tokens,{'document_IDs_seen':len(seen),'selected_document_text_decodes':1,'selected_hypothesis_text_decodes':17,'unselected_document_or_choice_decodes':0,'semantic_gold_decodes':0,'opaque_selected_choice_tokens_copied':17,'source_units':len(selected['evidence']),'value_decoder_counts':dict(s.decoded)}
def data(a,admission_ref,archive):
 PRIVATE.mkdir(mode=0o700,exist_ok=True);need(not any((PRIVATE/n).exists()for n in['DATA_ATTEMPT.json','FEATURES.json','GOLD.vault.json','DATA_REPORT.json','SOURCE_TEXT_ACCESS_BEFORE.json','SOURCE_TEXT_ACCESS_AFTER.json']),'DATA_ONCE_OUTPUTS_ABSENT');put(PRIVATE/'DATA_ATTEMPT.json',{'epoch':59,'mode':'DATA','admission':admission_ref,'reserved_development_case_ids':list(IDS),'automatic_retry':False})
 report={'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'phase':'dev','source_split':'train','project_role_semantics':PROJECT_ROLE_SEMANTICS,'filename_qualification_ref':a['filename_qualification_ref'],'selected_document_filename_key_sha256':None,'selected_normalized_document_sha256':None,'status':'DATA_FAILED_CLOSED_NO_RETRY','errors':[],'selected_ids':list(IDS),'identity_namespace':'contractnli:eced6528:train:473','reserved_development_case_ids':list(IDS),'reserved_role':'DEV_FOREVER','actual_feature_case_ids':[],'source_text_exposed_case_ids':[],'actual_hypothesis_exposed_case_ids':[],'value_decoder_counts':{},'semantic_gold_decodes':0,'all17_DOC473_hypotheses_DEV_forever_on_access':True,'used_confirmation_identities':[],'source_archive_sha256':ARCHIVE_SHA,'source_member_sha256':MEMBER_SHA,'official_codec_source_sha256':CODEC_SHA}
 try:
  owner_guard();b=body(archive);need(len(b)==ARCHIVE_BYTES and sha(b)==ARCHIVE_SHA and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==ARCHIVE_GIT,'EXACT_SOURCE_ARCHIVE')
  with zipfile.ZipFile(io.BytesIO(b))as z:
   infos=[i for i in z.infolist()if i.filename==MEMBER and not i.is_dir()];need(len(infos)==1 and infos[0].file_size==MEMBER_BYTES,'ONE_EXACT_DEV_MEMBER');member=z.read(infos[0])
  def exposure(stage):
   report['source_text_access_'+stage.lower()+'_ref']=put(PRIVATE/('SOURCE_TEXT_ACCESS_'+stage+'.json'),{'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'stage':stage,'reserved_development_case_ids':list(IDS),'all17_retired_as':'DEV_FOREVER','confirmed_document_text_decodes':1 if stage=='AFTER'else 0,'uncertain_if_process_dies_before_AFTER':stage=='BEFORE'})
  span=imported(OPERATIONAL/'span_json.py','e59_span_json');catalog=imported(OPERATIONAL/'source_catalog.py','e59_source_catalog');doc,cases,tokens,counts=extract(member,span,catalog,report,lambda:exposure('BEFORE'),lambda:exposure('AFTER'));owner_guard()
  bank={'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'role':'dev','source_split':'train','project_role_semantics':PROJECT_ROLE_SEMANTICS,'source_archive_sha256':ARCHIVE_SHA,'source_member_sha256':MEMBER_SHA,'documents':[doc],'cases':cases}
  features=put(PRIVATE/'FEATURES.json',bank);report['actual_feature_case_ids']=[row['id']for row in cases];report['features']=features
  vault=put(PRIVATE/'GOLD.vault.json',{'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'phase':'dev','source_archive_sha256':ARCHIVE_SHA,'features_sha256':features['sha256'],'ids':[row['id']for row in cases],'label_encoding':'OPAQUE_RAW_JSON_STRING_TOKENS_NOT_DECODED','official_codec_source_sha256':CODEC_SHA,'opaque_raw_choice_tokens_base64':tokens})
  report.update(status='READY_LABEL_BLIND_ONE_DOC473_ALL17',features=features,private_vault=vault,counts=counts,source_document_sha256=doc['source_document_sha256'],source_plan_sha256=a['plan_ref']['sha256'],source_freeze_sha256=a['source_freeze_ref']['sha256'])
 except Exception as e:report['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 put(PRIVATE/'DATA_REPORT.json',report);return report
def physically_closed(q):
 need(q['checks_passed']is True and q['all_owned_processes_closed']is True and type(q['actual_execution_independently_derived'])is dict and bool(q['actual_execution_independently_derived']),'INDEPENDENT_OBSERVED_COUNTS_AND_CLOSURE');o=owner_guard();need(q['kernel_boot_id']==o['kernel_boot_id']and q['PID_namespace']==o['PID_namespace'],'CLOSURE_CURRENT_BOOT_NAMESPACE')
 identities=q['owned_process_identities'];need(type(identities)is list and identities,'ALL_OWNED_PROCESS_IDENTITIES');groups=set();sessions=set()
 for x in identities:
  need(all(type(x[k])is int and x[k]>1 for k in['pid','startticks','pgid','sid'])and x['pid']!=OWNER_PID,'OWNED_CLOSURE_IDENTITY');groups.add(x['pgid']);sessions.add(x['sid']);p=Path('/proc')/str(x['pid'])
  if not p.exists():continue
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  if int(s[19])!=x['startticks']:continue
  need(s[0]=='Z'and not any(int(v)for v in(p/'statm').read_text().split())and not list((p/'fd').iterdir()),'OWNED_PROCESS_NOT_PHYSICALLY_CLOSED')
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:s=(p/'stat').read_text().rsplit(')',1)[1].split()
  except FileNotFoundError:continue
  need(s[0]in('Z','X')or(int(s[2])not in groups and int(s[3])not in sessions),'OWNED_PROCESS_GROUP_OR_SESSION_STILL_LIVE')
def validate_forecasts(f,a):
 expected={'schema_version','epoch','kind','phase','plan_sha256','source_freeze_sha256','labels','arms','selected_query_ids','reserved_ids','rows'}
 need(type(f)is dict and set(f)==expected and f['schema_version']==1 and f['epoch']==59 and f['kind']==a['kind']and f['phase']=='dev','EXACT_FROZEN_FORECAST_SCHEMA')
 need(f['plan_sha256']==a['plan_ref']['sha256']and f['source_freeze_sha256']==a['source_freeze_ref']['sha256'],'FORECAST_EXACT_PLAN_FREEZE')
 need(f['labels']==list(LABELS)and f['arms']==list(ARMS)and f['selected_query_ids']==list(IDS)and f['reserved_ids']==list(IDS),'FROZEN_CLASS_ARM_AND_CASE_IDENTITIES')
 rows=f['rows'];need(type(rows)is list and len(rows)==51,'ALL51_FORECASTS_REQUIRED')
 windows_total=a['frozen_window_count'];need(type(windows_total)is int and 1<=windows_total<=24,'EXACT_FROZEN_BLIND_WINDOW_COUNT')
 rowfields={'case_id','arm','probabilities','log_probabilities','elapsed_seconds','actual_model_forward_calls','actual_pair_inferences','context_window_ids','all_source_windows_covered','early_exit','numeric_flags','prediction'}
 for r,(cid,arm)in zip(rows,((cid,arm)for cid in IDS for arm in ARMS)):
  need(type(r)is dict and set(r)==rowfields and r['case_id']==cid and r['arm']==arm,'EXACT_ORDERED_CASE_ARM_ROW_NO_DUPLICATES')
  p=r['probabilities'];lp=r['log_probabilities'];need(type(p)is list and len(p)==3 and all(type(x)in(int,float)and math.isfinite(x)and 0<x<=1 for x in p)and abs(sum(p)-1)<=1e-6,'VALID_POSITIVE_THREEWAY_PROBABILITIES')
  need(type(lp)is list and len(lp)==3 and all(type(x)in(int,float)and math.isfinite(x)and x<=0 for x in lp)and all(abs(math.exp(x)-v)<=1e-12 for x,v in zip(lp,p)),'VALID_UNCLIPPED_CONSISTENT_LOG_PROBABILITIES')
  need(type(r['elapsed_seconds'])in(int,float)and math.isfinite(r['elapsed_seconds'])and r['elapsed_seconds']>=0 and all(type(r[k])is int and r[k]>=0 for k in['actual_model_forward_calls','actual_pair_inferences']),'OBSERVED_NONNEGATIVE_RUNTIME_COUNTS')
  need(type(r['all_source_windows_covered'])is bool and type(r['early_exit'])is bool,'EXACT_COVERAGE_FLAGS')
  windows=r['context_window_ids'];need(type(windows)is list and 1<=len(windows)<=windows_total and len(set(windows))==len(windows)and all(type(x)is int and 0<=x<windows_total for x in windows),'BOUNDED_ACTUAL_UNIQUE_CONTEXT_WINDOW_IDS')
  need(r['actual_pair_inferences']==len(windows)and math.ceil(len(windows)/8)<=r['actual_model_forward_calls']<=len(windows),'ACTUAL_PAIR_AND_FORWARD_COUNTS_MATCH_CONSUMED_CONTEXT')
  need(not r['all_source_windows_covered']or set(windows)==set(range(windows_total)),'FULL_COVERAGE_REQUIRES_ALL_FROZEN_WINDOWS')
  flags=r['numeric_flags'];allowed=('claim_has_numerals','claim_numeral_not_in_supplied_context');need(type(flags)is list and flags==[x for x in allowed if x in flags]and(not flags or not r['early_exit']),'FROZEN_NUMERIC_DIAGNOSTICS_DISABLE_EARLY_EXIT')
  need(r['prediction']==LABELS[max(range(3),key=p.__getitem__)],'EXACT_OFFICIAL_LABEL_ARGMAX')
  need(r['prediction']!='NotMentioned'or r['all_source_windows_covered'],'INSUFFICIENT_EVIDENCE_REQUIRES_ALL_SUPPLIED_CONTEXT')
  need(arm!='full_reference'or(r['all_source_windows_covered']and not r['early_exit']),'FULL_REFERENCE_REQUIRES_ALL_WINDOWS_NO_EARLY_EXIT')
 return rows
def validate_catalog(catalog,bank,window_count,kind):
 fields={'schema_version','epoch','kind','role','source_split','project_role_semantics','selected_query_ids','reserved_ids','document_id','source_document_sha256','frozen_window_count','windows','cases','all17_pair_capacity_checked','BGE_query_prefix','max_tokens','max_windows','no_label_or_model_output_access'}
 need(type(catalog)is dict and set(catalog)==fields and catalog['schema_version']==1 and catalog['epoch']==59 and catalog['kind']==kind and catalog['role']=='dev'and catalog['source_split']=='train'and catalog['project_role_semantics']==PROJECT_ROLE_SEMANTICS and catalog['document_id']==473,'EXACT_BLIND_CATALOG_SCHEMA')
 need(type(window_count)is int and 1<=window_count<=24 and catalog['frozen_window_count']==window_count and catalog['selected_query_ids']==list(IDS)and catalog['reserved_ids']==list(IDS)and catalog['max_tokens']==512 and catalog['max_windows']==24,'EXACT_BLIND_CATALOG_IDENTITIES_AND_LIMITS')
 need(catalog['all17_pair_capacity_checked']is True and catalog['no_label_or_model_output_access']is True and type(catalog['BGE_query_prefix'])is str,'EXACT_PREPARATION_CAPACITY_AND_BLINDNESS_ASSERTIONS')
 need(bank['epoch']==59 and bank['kind']==kind and bank['source_checkpoint']==BASE and bank['role']=='dev'and bank['source_split']=='train'and bank['project_role_semantics']==PROJECT_ROLE_SEMANTICS and len(bank['documents'])==1 and [r['id']for r in bank['cases']]==list(IDS)and catalog['cases']==bank['cases'],'CATALOG_EXACT_ORIGINAL17_FEATURE_CASES')
 doc=bank['documents'][0];need(doc['document_id']==473 and doc['source_document_sha256']==catalog['source_document_sha256'],'CATALOG_SOURCE_DOCUMENT_IDENTITY');text=''.join(r['sentence']for r in doc['evidence']);need(sha(text.encode())==doc['source_document_sha256'],'ORIGINAL_SOURCE_BANK_TEXT_HASH')
 windows=catalog['windows'];need(type(windows)is list and len(windows)==window_count,'EXACT_FROZEN_WINDOW_COUNT');end=0
 for index,w in enumerate(windows):
  need(type(w)is dict and set(w)=={'window_id','start','end','source_unit_ids','text','text_sha256'}and type(w['window_id'])is int and w['window_id']==index and type(w['start'])is int and type(w['end'])is int and w['start']==end and w['end']>end and w['end']<=len(text),'CONTIGUOUS_EXACT_SOURCE_WINDOW_OFFSETS')
  need(type(w['text'])is str and w['text']==text[w['start']:w['end']]and sha(w['text'].encode())==w['text_sha256'],'EXACT_UNREWRITTEN_WINDOW_TEXT_HASH')
  units=w['source_unit_ids'];need(type(units)is list and bool(units)and all(type(x)is int and 0<=x<len(doc['evidence'])for x in units)and units==sorted(set(units)),'VALID_SOURCE_UNIT_REFERENCES')
  need(units==[u['unit_id']for u in doc['evidence']if u['start']<w['end']and u['end']>w['start']],'EXACT_SOURCE_UNITS_OVERLAPPING_WINDOW');end=w['end']
 need(end==len(text),'CATALOG_COVERS_ALL_SUPPLIED_SOURCE');return windows
def validate_observed_execution(q,a,forecasts):
 need(q['report_type']=='JOINT_LATE_GOLD_CLOSURE'and q['epoch']==59 and q['kind']==a['kind'],'EXPLICIT_JOINT_LATE_GOLD_AUDIT_SCHEMA')
 observed=q['actual_execution_independently_derived'];need(type(observed)is dict and set(observed)=={'engineering','dev'},'EXACT_TWO_PHASE_OBSERVED_COUNTS')
 rawrefs=q['raw_worker_results'];native_refs=q['native_result_refs'];need(type(rawrefs)is dict and set(rawrefs)==set(observed)and type(native_refs)is dict and set(native_refs)==set(observed),'EXACT_TWO_PHASE_RAW_AND_NATIVE_REFS')
 need(q['plan_ref']==a['plan_ref']and q['source_freeze_ref']==a['source_freeze_ref'],'JOINT_AUDIT_EXACT_MASTER_PLAN_AND_SOURCE_FREEZE')
 raw=body(OPERATIONAL/'COUNTER_SCHEMA.v1.json');need(len(raw)==5115 and sha(raw)==COUNTER_SCHEMA_SHA,'EXACT_CANONICAL42_COUNTER_SCHEMA');schema=parse(raw);need(len(schema['keys'])==42 and len(set(schema['keys']))==42,'EXACT42_UNIQUE_COUNTER_KEYS')
 identities={(x['pid'],x['startticks'],x['pgid'],x['sid'])for x in q['owned_process_identities']}
 for phase in('engineering','dev'):
  counters=observed[phase];need(type(counters)is dict and set(counters)==set(schema['keys'])and all(type(x)is int and x>=0 for x in counters.values()),'EXACT42_NAMED_NONNEGATIVE_INTEGER_OBSERVATIONS')
  need(all(counters[attempt]>=counters[complete]for attempt,complete in schema['attempt_completion_pairs']),'OBSERVED_COMPLETIONS_NEVER_EXCEED_ATTEMPTS')
  need(all(counters[k]==0 for k in schema['worker_zero_keys'])and all(counters[k]>0 for k in schema['positive_actual_'+phase+'_completed_keys']),'ACTUAL_NATIVE_NLI_REQUIRED_NO_NETWORK_PAID_GPU_OR_GOLD')
  worker=load_ref(rawrefs[phase]);rawcounts=worker['counters'];need(type(rawcounts)is dict and set(rawcounts)==set(schema['keys'])and all(type(x)is int and x>=0 for x in rawcounts.values()),'RAW_WORKER_EXACT42_STRICT_INTEGER_COUNTERS')
  need(worker['schema_version']==1 and worker['epoch']==59 and worker['kind']==a['kind']and worker['phase']==phase and worker['errors']==[]and worker['plan_sha256']==a['plan_ref']['sha256']and worker['source_freeze_sha256']==a['source_freeze_ref']['sha256']and rawcounts==counters,'INDEPENDENT_COUNTS_EXACTLY_EQUAL_PHYSICAL_RAW_WORKER')
  need(worker['native_results_ref']==native_refs[phase],'EXACT_INDEPENDENT_NATIVE_PROOF_REF');read_ref(native_refs[phase])
  need(all((x['pid'],x['startticks'],x['pgid'],x['sid'])in identities for x in worker['owned_process_identities']),'JOINT_CLOSURE_COVERS_ALL_RAW_OWNED_PROCESS_IDENTITIES')
  if phase=='engineering':need(worker['status']=='ENGINEERING_EXECUTED_PENDING_INDEPENDENT_AUDIT','ACTUAL_ENGINEERING_COMPLETED_STATUS')
  else:
   need(worker['status']=='DEV_FORECASTS_SEALED_NO_GOLD'and worker['sealed_forecasts_ref']==a['sealed_forecasts_ref']and worker['catalog_ref']==a['catalog_ref']and worker['rows']==forecasts['rows']and worker['selected_query_ids']==list(IDS)and worker['reserved_ids']==list(IDS)and counters['forecasts_completed']==51,'ALL51_PHYSICAL_DEV_FORECASTS_SEALED_MATCH_RAW')
 need(sum(observed[p]['native_processes_attempted']for p in observed)<=2,'AT_MOST_TWO_ACTUAL_NATIVE_CORPUS_BATCH_ATTEMPTS')
def gold(a,admission_ref):
 report={'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'phase':'dev','status':'GOLD_FAILED_CLOSED_NO_RETRY','errors':[],'identity_namespace':'contractnli:eced6528:train:473','reserved_development_case_ids':list(IDS),'reserved_role':'DEV_FOREVER','actual_label_decoded_case_ids':[],'semantic_labels_decoded':0,'semantic_decode_count_uncertain':False,'actual_decoded_label_class_counts':{label:0 for label in LABELS},'actual_decoded_unrecognized_labels':0,'used_confirmation_identities':[]}
 need(not any((PRIVATE/n).exists()for n in['GOLD_ATTEMPT.json','GOLD_JOINED.json','GOLD_REPORT.json']),'GOLD_ONCE_OUTPUTS_ABSENT');put(PRIVATE/'GOLD_ATTEMPT.json',{'epoch':59,'mode':'GOLD','admission':admission_ref,'automatic_retry':False})
 try:
  d=load_ref(a['data_report_ref']);need(d['epoch']==59 and d['kind']==a['kind']and d['source_checkpoint']==BASE and d['status']=='READY_LABEL_BLIND_ONE_DOC473_ALL17'and d['errors']==[]and d['selected_ids']==list(IDS)and d['actual_feature_case_ids']==list(IDS)and d['source_text_exposed_case_ids']==list(IDS)and d['semantic_gold_decodes']==0,'EXACT_BLIND_DATA_REPORT')
  need(a['vault_ref']==d['private_vault'],'DATA_ISSUED_VAULT_REF');q=load_ref(a['independent_actual_audit_ref']);need(q['sealed_forecasts_ref']==a['sealed_forecasts_ref']and q['catalog_ref']==a['catalog_ref'],'EXACT_Q_AND_ADMISSION_FORECAST_AND_CATALOG_REFS')
  catalog=load_ref(a['catalog_ref']);bank=load_ref(d['features']);validate_catalog(catalog,bank,a['frozen_window_count'],a['kind']);need(catalog['source_document_sha256']==d['source_document_sha256'],'EXACT_BLIND_FROZEN_CATALOG_SOURCE')
  forecasts=load_ref(a['sealed_forecasts_ref']);validate_forecasts(forecasts,a);validate_observed_execution(q,a,forecasts);physically_closed(q)
  vault=load_ref(a['vault_ref']);need(vault['epoch']==59 and vault['kind']==a['kind']and vault['source_checkpoint']==BASE and vault['phase']=='dev'and vault['source_archive_sha256']==ARCHIVE_SHA and vault['ids']==list(IDS)and vault['official_codec_source_sha256']==CODEC_SHA and vault['features_sha256']==d['features']['sha256']and vault['label_encoding']=='OPAQUE_RAW_JSON_STRING_TOKENS_NOT_DECODED','EXACT_OPAQUE_VAULT_SOURCE');tokens=vault['opaque_raw_choice_tokens_base64'];need(len(tokens)==17,'EXACT17_OPAQUE_TOKENS');labels={}
  for i,(cid,encoded)in enumerate(zip(IDS,tokens)):
   owner_guard();raw=base64.b64decode(encoded,validate=True);put(PRIVATE/f'GOLD_{i:02d}_BEFORE.json',{'id':cid,'semantic_labels_decoded':report['semantic_labels_decoded'],'token_sha256':sha(raw)});value=parse(raw);report['semantic_labels_decoded']+=1;report['actual_label_decoded_case_ids'].append(cid);known=type(value)is str and value in LABELS
   if known:report['actual_decoded_label_class_counts'][value]+=1
   else:report['actual_decoded_unrecognized_labels']+=1
   put(PRIVATE/f'GOLD_{i:02d}_AFTER.json',{'id':cid,'semantic_labels_decoded':report['semantic_labels_decoded'],'supported_exact_literal':type(value)is str and value in LABELS});need(type(value)is str and value in LABELS,'UNSUPPORTED_OFFICIAL_LABEL_NO_REMAP');labels[cid]=value
  need(len(labels)==report['semantic_labels_decoded']==sum(report['actual_decoded_label_class_counts'].values())==17 and report['actual_decoded_unrecognized_labels']==0,'EXACT17_LATE_GOLD');report.update(status='GOLD_JOIN_COMPLETE_AFTER_ALL_FORECASTS_AND_PHYSICAL_CLOSURE',sealed_forecasts_ref=a['sealed_forecasts_ref'],joined_gold=put(PRIVATE/'GOLD_JOINED.json',{'schema_version':1,'epoch':59,'kind':a['kind'],'source_checkpoint':BASE,'phase':'dev','ids':list(IDS),'label_order':list(LABELS),'rows':[{'case_id':cid,'label':labels[cid]}for cid in IDS],'semantic_labels_decoded':17,'actual_decoded_label_class_counts':dict(report['actual_decoded_label_class_counts']),'official_codec_source_sha256':CODEC_SHA,'sealed_forecasts_ref':a['sealed_forecasts_ref'],'all_DOC473_claims_DEV_forever':True,'used_confirmation_identities':[]}))
 except Exception as e:report['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});report['semantic_decode_count_uncertain']=len(list(PRIVATE.glob('GOLD_*_BEFORE.json')))!=len(list(PRIVATE.glob('GOLD_*_AFTER.json')))
 put(PRIVATE/'GOLD_REPORT.json',report);return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['DATA','GOLD']);ap.add_argument('--admission',required=True);ap.add_argument('--expected-admission-sha256',required=True);ap.add_argument('--archive');a=ap.parse_args();x,r=admission(a.admission,a.expected_admission_sha256,a.mode)
 if a.mode=='DATA':need(a.archive is not None,'EXACT_ARCHIVE_REQUIRED');out=data(x,r,a.archive)
 else:need(a.archive is None,'NO_ARCHIVE_OR_CONFIRMATION_ACCESS_IN_GOLD');out=gold(x,r)
 print(json.dumps({'epoch':59,'status':out['status'],'errors':out['errors'],'semantic_gold_decodes':out.get('semantic_gold_decodes',out.get('semantic_labels_decoded'))},sort_keys=True));return 1 if out['errors']else 0
if __name__=='__main__':raise SystemExit(main())
