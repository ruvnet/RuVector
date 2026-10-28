"""Bounded official model source metadata only. No weights, runtime, data, or inference."""
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
import os,json,hashlib,time,datetime,re
R=Path('/dev/shm/ruvector-oct28-model-research');CAP=1048576
REPOS=('FacebookAI/roberta-large-mnli','tasksource/deberta-base-long-nli')
class NoRedirect(HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):return None
opener=build_opener(NoRedirect());calls=[]
def put(p,b):
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def obj(p,d):put(p,(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def fetch(repo,kind,url):
 assert len(calls)<8 and url.startswith('https://huggingface.co/') and kind in('api','README','config','LICENSE')
 stem=repo.replace('/','--')+'--'+kind
 intent={'repo':repo,'kind':kind,'url':url,'attempt':1,'timeout_seconds':45,'redirect_follow':False,'retry':False,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 obj(R/(stem+'.intent.json'),intent);t=time.monotonic();raw=b'';status=None;error=None
 try:
  with opener.open(Request(url,headers={'User-Agent':'ruvector-bounded-model-source-research/1.0'}),timeout=45)as response:
   status=response.status;raw=response.read(CAP+1)
 except HTTPError as e:status=e.code;raw=e.read(CAP+1);error='HTTPError'
 except Exception as e:error=type(e).__name__
 row={**intent,'status':status,'error_type':error,'elapsed_seconds':time.monotonic()-t,'captured_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'over_limit':len(raw)>CAP,'raw_path':str(R/(stem+'.raw'))}
 put(Path(row['raw_path']),raw);obj(R/(stem+'.actual.json'),row);calls.append(row)
 if status!=200 or len(raw)>CAP:return None
 return raw
models=[]
for repo in REPOS:
 raw=fetch(repo,'api','https://huggingface.co/api/models/'+repo+'?blobs=true')
 if raw is None:models.append({'repository':repo,'status':'SOURCE_METADATA_BLOCKED','queries':len(calls)});continue
 try:a=json.loads(raw)
 except Exception:models.append({'repository':repo,'status':'API_SCHEMA_UNRECOGNIZED'});continue
 revision=a.get('sha');assert type(revision)is str and re.fullmatch('[0-9a-f]{40}',revision)
 siblings=a.get('siblings');assert type(siblings)is list
 names={x['rfilename']:x for x in siblings};record={'repository':repo,'revision':revision,'api_id':a.get('id'),'license_declaration':(a.get('cardData')or{}).get('license'),'gated':a.get('gated'),'private':a.get('private'),'createdAt':a.get('createdAt'),'lastModified':a.get('lastModified'),'source_file_rows':siblings,'cardData':a.get('cardData'),'safetensors':a.get('safetensors'),'fetched_source_paths':[]}
 for kind,name in [('README','README.md'),('config','config.json')]:
  if name not in names:record[kind+'_status']='NOT_LISTED';continue
  data=fetch(repo,kind,'https://huggingface.co/'+repo+'/raw/'+revision+'/'+name)
  if data is not None:
   record['fetched_source_paths'].append(str(R/(repo.replace('/','--')+'--'+kind+'.raw')))
   if kind=='config':
    try:record['config']=json.loads(data)
    except Exception:record['config_status']='UNRECOGNIZED_JSON'
 candidates=[name for name in names if name.lower()in('license','license.txt','license.md')]
 if candidates:
  name=sorted(candidates)[0];data=fetch(repo,'LICENSE','https://huggingface.co/'+repo+'/raw/'+revision+'/'+name)
  record['standalone_LICENSE_file']=name;record['standalone_LICENSE_success']=data is not None
 else:record['standalone_LICENSE_file']='NOT_LISTED';record['standalone_LICENSE_success']=False
 record['status']='METADATA_REVIEW_REQUIRED_NO_EXECUTION_AUTHORITY';models.append(record)
obj(R/'METADATA_HTTP_ACTUAL.v1.json',{'schema_version':1,'kind':'OFFICIAL_MODEL_SOURCE_METADATA_ONLY','actual_GETs':len(calls),'repos_examined':len(REPOS),'redirects_followed':0,'retries':0,'weight_tokenizer_model_native_Gold_paid_GPU_calls':0,'calls':calls,'models':models})
print(json.dumps({'actual_GETs':len(calls),'model_revisions':[{k:x.get(k)for k in['repository','revision','status','license_declaration','standalone_LICENSE_file']}for x in models],'actual_result':str(R/'METADATA_HTTP_ACTUAL.v1.json')}))
