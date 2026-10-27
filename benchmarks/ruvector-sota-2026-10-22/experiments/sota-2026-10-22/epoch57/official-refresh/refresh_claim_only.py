"""Independent ClaimBench step after bounded Jev stream; two GETs, no retry."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import urljoin,urlparse
import urllib.request,ssl,os,json,re,hashlib
OUT=Path('/dev/shm/ruvector-e57-data/official-refresh')
prior=json.loads((OUT/'donors/benchmark-refresh.safe.json').read_bytes())
assert hashlib.sha256((OUT/'donors/benchmark-refresh.safe.json').read_bytes()).hexdigest()=='dc3aa9ff96ad0e88357ccb41c75d13637326eece04287af4b823ab365a6cc1f4'
def now():return datetime.now(timezone.utc).isoformat()
def put(name,obj):
 with(OUT/name).open('x')as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):raise RuntimeError('Unexpected redirect; no extra GET authorized')
ca=os.environ.get('SSL_CERT_FILE')or os.environ.get('REQUESTS_CA_BUNDLE')
opener=urllib.request.build_opener(urllib.request.ProxyHandler(),urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca if ca else None)),NoRedirect())
receipts=[]
def get(url):
 assert len(receipts)<2
 start=now();req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 official-source-read-only','Accept-Encoding':'identity'})
 with opener.open(req,timeout=45)as response:
  assert response.status==200 and response.url==url
  cap=2*1024*1024;decl=response.headers.get('Content-Length')
  assert decl is None or int(decl)<=cap
  b=response.read(cap+1);assert len(b)<=cap
  r={'url':url,'final_url':response.url,'status':response.status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content_type':response.headers.get('Content-Type'),'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified'),'request_started_at':start,'retrieved_at':now(),'read_cap_bytes':cap,'TLS_verification_enabled':True,'raw_body_persisted':False}
 receipts.append(r);put('claim-source-receipt-%d.json'%len(receipts),r);return b,r
class Page(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'href'in a:self.links.append(a['href'])
put('CLAIM_REFRESH_ATTEMPT.json',{'epoch':57,'started_at':now(),'maximum_GETs':2,'nonretryable':True,'independent_step_after_Jev_header_cap':'Separate fixed2GET Claim source step after new bounded Jev stream; no repeated access probes','no_Git_source_data_model_Gold_calls':True})
b,receipt=get(prior['ClaimBench']['source']['url']);page=Page();page.feed(b.decode())
links=sorted(set(urljoin(receipt['url'],s)for s in page.links if re.search(r'/_astro/benchmark_view_claimbench\.[A-Za-z0-9_-]+\.json(?:$|\?)',s)))
assert len(links)==1 and urlparse(links[0]).scheme=='https'and urlparse(links[0]).netloc=='www.vals.ai'
a,aggregate_receipt=get(links[0]);d=json.loads(a);m=d['metadata'];overall=d['tasks']['overall']
decisions=sorted([{'model':n,'accuracy_percent':v['accuracy']}for n,v in overall.items()if n.startswith('decision/')],key=lambda r:r['accuracy_percent'],reverse=True)
assert len(decisions)>=2
facts={'schema_version':1,'epoch':57,'kind':'FRESH_OFFICIAL_CLAIM_PUBLIC_SOURCE_AND_LINKED_AGGREGATE_READ_ONLY','created_at':now(),'source_receipts':receipts,
 'ClaimBench':{'source':receipt,'official_aggregate':aggregate_receipt,'aggregate_URL_verified_from_current_official_page':True,'total_models':m['total_models'],'publisher_version':m['version'],'publisher_updated':m['updated'],'publisher_metadata_dataset_type':m['dataset_type'],'decision_leader':decisions[0],'decision_runner_up':decisions[1],'language_models_at100':[{'model':n,'accuracy_percent':v['accuracy']}for n,v in overall.items()if not n.startswith('decision/')and v['accuracy']==100],'RuVector_in_aggregate':any('ruvector'in n.lower()for n in overall),'RuVector_score':None,'source_bytes_equal_epoch56':receipt['sha256']==prior['ClaimBench']['source']['sha256'],'aggregate_bytes_equal_epoch56':aggregate_receipt['sha256']==prior['ClaimBench']['official_aggregate']['sha256'],'full_dataset_scorer_license_submission_access':'UNVERIFIED. Public examples, metadata and aggregate JSON do not establish full labelled cases, exact scorer, redistribution licence or submission access.','verified_dataset_access_scope':'Current official public page examples and linked aggregate only.','retrieval_scope':'ONLY supplied 10-K excerpts. A retrieval miss cannot establish insufficient evidence or NEI.','pilot_executed':False,'historical_access_probes_repeated':False,'access_candidate_links_on_current_page_not_followed':sorted(set(s for s in page.links if re.search(r'dataset|scorer|licen[cs]e|submit|huggingface|github',s,re.I)))[:30]},
 'direct_public_GETs_in_this_step':2,'earlier_Jev_stream_GETs':1,'total_epoch_official_public_GETs':3,'raw_public_bodies_persisted':False,'inherited_proxy_preserved':True,'TLS_verification_enabled':True,'model_native_gold_or_paid_GPU_calls':0,'project_or_Git_mutations':0,'outgoing_messages_submissions_or_author_contacts':0,'goal_verified':False,'official_SOTA_verified':False}
payload=(json.dumps(facts,sort_keys=True,indent=2)+'\n').encode();assert b'@'not in payload and len(payload)<100000
put('claim-public-facts.json',facts)
print(json.dumps({'decision_leader':decisions[0],'decision_runner_up':decisions[1],'source_bytes_equal_epoch56':facts['ClaimBench']['source_bytes_equal_epoch56'],'aggregate_bytes_equal_epoch56':facts['ClaimBench']['aggregate_bytes_equal_epoch56'],'actual_claim_GETs':2,'total_official_GETs_including_Jev_stream':3}))
