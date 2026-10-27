"""One frozen metadata-only filename-alias qualification for originalTRAIN473.

Byte-frame the one hash-pinned train member and decode423 filenames only.
Never decode source text, hypothesis, choice or evidence labels. No replacement.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, os, re, stat, traceback, unicodedata, zipfile

W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983')
OUT=W/'private-identity'
BASE='9b677a91698352bd563f189fee861d6ff2a1ad3a'
ARCHIVE_BYTES=65362913
ARCHIVE_SHA='e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b'
ARCHIVE_GIT='757fd1dafd29a997fba00c60c6d40b2930a36159'
MEMBER='contract-nli/train.json'
MEMBER_SHA='dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5'
MEMBER_CAP=16777216
def need(v,c):
 if not v:raise ValueError(c)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(rows):
 d={}
 for k,v in rows:need(k not in d,'DUPLICATE_JSON_KEY');d[k]=v
 return d
def parse(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in()).throw(ValueError('NONFINITE_JSON')))
def body(p):
 p=Path(p);need(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in[p,*p.parents]),'CANONICAL_REGULAR_SOURCE');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb')as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode),'REGULAR_SOURCE');b=f.read();z=os.fstat(f.fileno())
 need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'STABLE_SOURCE');return b
def ref(p):
 b=body(p);return{'path':str(p),'bytes':len(b),'sha256':sha(b)}
def checked(r):
 need(type(r)is dict and set(r)=={'path','bytes','sha256'}and type(r['bytes'])is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256'])is not None,'EXACT_TYPED3_REF');b=body(r['path']);need(len(b)==r['bytes']and sha(b)==r['sha256'],'REF_BYTES_SHA');return b
def put(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 q=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(q);os.close(q);return ref(p)
def imported(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def admission(p,h):
 b=body(p);need(sha(b)==h,'ROOT_ADMISSION_SHA');a=parse(b);need(a['epoch']==59 and a['phase']=='FILENAME_IDENTITY_ONLY'and a['selected_document_id']==473 and a['source_split']=='train'and a['project_role']=='PROJECT_DEV','ONE_EXACT_PROJECT_DEV_IDENTITY')
 need(a['archive_text_hypothesis_choice_value_decodes_allowed']=={'filename_metadata':423,'source_text':0,'hypothesis':0,'choice':0},'EXPLICIT_FILENAME_ONLY_VALUE_BUDGET')
 plan=parse(checked(a['plan_ref']));sf=parse(checked(a['source_freeze_ref']));q=parse(checked(a['independent_preparation_approval_ref']))
 need(plan['epoch']==59 and plan['phase']==a['phase']and plan['source_checkpoint']==BASE and plan['selected_document_id']==473 and plan['no_replacement_on_collision_or_failure']is True,'EXACT_FROZEN_METADATA_PLAN')
 need(q['checks_passed']is True and q['filename_metadata_qualification_allowed']is True and q['plan_ref']==a['plan_ref']and q['source_freeze_ref']==a['source_freeze_ref'],'INDEPENDENT_METADATA_SOURCE_ADMISSION')
 files={r['path']:r for r in sf['files']};need(len(files)==len(sf['files']),'UNIQUE_FROZEN_METADATA_FILES')
 for r in [plan['code'],plan['span_module'],plan['alias_policy'],plan['fence_source'],a['plan_ref']]:need(files.get(r['path'])==r,'METADATA_SOURCE_IN_FREEZE');checked(r)
 need(Path(plan['code']['path'])==Path(__file__).resolve(),'EXACT_OPERATIONAL_METADATA_CODE');f=imported(plan['fence_source']['path'],'e59_filename_shared_fence');f.verify_owner(plan['owner'])
 return a,plan,lambda:f.verify_owner(plan['owner'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--admission',required=True);ap.add_argument('--expected-admission-sha256',required=True);args=ap.parse_args();need(not __debug__ is False and os.sched_getaffinity(0)=={0,1},'NORMAL_CPU01_METADATA_EXECUTION')
 a,p,fence=admission(args.admission,args.expected_admission_sha256);need(not OUT.exists(),'ONCE_METADATA_OUTPUTS_ABSENT');OUT.mkdir(mode=0o700)
 put(OUT/'FILENAME_ATTEMPT.actual.json',{'schema_version':1,'epoch':59,'phase':'FILENAME_IDENTITY_ONLY','Root_admission':ref(args.admission),'selected_document_id':473,'source_split':'train','automatic_retry':False})
 report={'schema_version':1,'epoch':59,'phase':'FILENAME_IDENTITY_ONLY','source_checkpoint':BASE,'status':'FILENAME_IDENTITY_FAILED_CLOSED','errors':[],'selected_document_id':473,'source_split':'train','project_role':'PROJECT_DEV','source_text_decodes':0,'hypothesis_value_decodes':0,'semantic_Gold_decodes':0,'filename_metadata_decodes':0,'no_replacement':True,'used_confirmation_identities':[]}
 try:
  fence();policy=parse(checked(p['alias_policy']));need(policy['selected_document_id']==473 and policy['no_replacement_on_collision_or_failure']is True,'EXACT_SELECTED_ALIAS_POLICY')
  b=body(p['source_archive_path']);need(len(b)==ARCHIVE_BYTES and sha(b)==ARCHIVE_SHA and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==ARCHIVE_GIT,'EXACT_CURRENT_ARCHIVE_BYTES_HASH')
  with zipfile.ZipFile(io.BytesIO(b))as z:
   infos=[x for x in z.infolist()if x.filename==MEMBER and not x.is_dir()];need(len(infos)==1 and 0<infos[0].file_size<=MEMBER_CAP,'ONE_BOUNDED_TRAIN_MEMBER');raw=z.read(infos[0]);need(len(raw)==infos[0].file_size and sha(raw)==MEMBER_SHA,'EXACT_CURRENT_TRAIN_MEMBER')
  report.update(source_archive_bytes=len(b),source_archive_sha256=ARCHIVE_SHA,source_member=MEMBER,source_member_bytes=len(raw),source_member_sha256=MEMBER_SHA)
  span=imported(p['span_module']['path'],'e59_filename_only_span');s=span.SpanJSON(raw);root=s.value(0);need(root[2]=='object'and s.ws(root[1])==len(raw),'ONE_SOURCE_MEMBER_OBJECT');top=s.members(root);seen=set();keys={}
  for row in s.items(top['documents']):
   fields=s.members(row);need(fields['id'][2]=='number','NUMERIC_ID_METADATA');did=parse(s.raw(fields['id']));need(type(did)is int and did not in seen,'UNIQUE_INTEGER_ID');seen.add(did)
   name=s.decode_string(fields['file_name'],'metadata');key=' '.join(unicodedata.normalize('NFKC',name.replace('\\','/').rsplit('/',1)[-1]).casefold().split());need(key and len(key.encode())<=65536,'BOUNDED_FILENAME_METADATA');keys[did]=sha(key.encode());report['filename_metadata_decodes']+=1
  need(seen==set(policy['source_train_document_ids'])and len(seen)==423 and s.decoded['metadata']==423 and s.decoded['selected_document']==s.decoded['selected_hypothesis']==s.decoded['legacy_dev_text']==s.gold_values_decoded==0,'EXACT423_FILENAMES_ZERO_SOURCE_OR_GOLD_VALUES')
  selected=keys[473];aliases=sorted(k for k,v in keys.items()if v==selected);blocked=sorted(set(aliases)&set(policy['blocked_exposed_or_reserved_source_TRAIN_document_ids']));prior=selected in set(policy['prior_DEV_TEST_Legal96_filename_key_sha256'])
  report.update(document_IDs_seen=len(seen),selected_document_filename_key_sha256=selected,selected_normalized_document_sha256_from_frozen_metadata=policy['selected_normalized_document_sha256'],all_source_TRAIN_filename_alias_document_ids=aliases,blocked_or_reserved_TRAIN_filename_alias_document_ids=blocked,prior_DEV_TEST_Legal96_filename_collision=prior,value_decoder_counts=dict(s.decoded),byte_framing_scope='Full train member traversed as bytes; decoded only JSON object field names, numeric document IDs, and423 file_name metadata values.',source_text_and_hypothesis_values_decoded=0,Gold_choice_values_decoded=0)
  need(not blocked and not prior,'PRIOR_OR_PROTECTED_FILENAME_COLLISION_ABORT_NO_REPLACEMENT');need(set(aliases)<=set(policy['unassigned_source_TRAIN_document_ids']),'ALL_NEW_FILENAME_ALIASES_UNASSIGNED')
  report.update(status='READY_TRAIN473_FILENAME_ALIAS_QUALIFIED_NO_FEATURES',checks_passed=True,all_filename_alias_document_ids_retired_as_PROJECT_DEV_forever=aliases,train_ids=[],calibration_ids=[],fresh_confirmation_ids=[],confirmation_qualification='UNQUALIFIED_HOLD',source_plan_ref=a['plan_ref'],source_freeze_ref=a['source_freeze_ref'])
  fence()
 except Exception as e:
  report.update(checks_passed=False);report['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 r=put(OUT/'FILENAME_QUALIFICATION.actual.v1.json',report);print(json.dumps(r,sort_keys=True));need(report['checks_passed']is True,'METADATA_QUALIFICATION_FAILED_CLOSED')
if __name__=='__main__':main()
