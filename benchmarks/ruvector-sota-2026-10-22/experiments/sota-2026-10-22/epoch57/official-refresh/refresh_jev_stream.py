"""One bounded official Jev GET, streaming hash/parse only; no raw body files."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
import codecs,hashlib,json,os,re,signal,ssl,time,urllib.request

OUT=Path('/dev/shm/ruvector-e57-data/official-refresh')
URL='https://benchmarkheaven.com/jev-models'
CAP=64*1024*1024
SECONDS=30.0
def now():return datetime.now(timezone.utc).isoformat()
def put(name,obj):
 b=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 assert len(b)<262144 and b'@'not in b
 p=OUT/name
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise RuntimeError('UNEXPECTED_REDIRECT_NO_EXTRA_GET')
class Page(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.skip=0;self.text=[];self.visiblechars=0;self.rows=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag in ('script','style'):self.skip+=1
  if 'data-bh-jev14-capability-row'in a:
   self.rows.append({'model_id':a['data-bh-jev14-capability-row'],'Capability':float(a['data-bh-jev14-capability-value'])})
   assert len(self.rows)<=5000,'CAPABILITY_ROWS_BOUND'
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.skip=max(0,self.skip-1)
 def handle_data(self,s):
  if not self.skip:
   self.visiblechars+=len(s);assert self.visiblechars<=2*1024*1024,'VISIBLE_TEXT_BOUND'
   self.text.append(s)
def alarm(signum,frame):raise TimeoutError('OFFICIAL_JEV_WHOLE_OPERATION_30S_DEADLINE')
def main():
 source=Path(__file__).read_bytes()
 plan={'schema_version':1,'epoch':57,'kind':'NEW_BOUNDED_OFFICIAL_JEV_STREAM_CAPTURE_PLAN','url':URL,'maximum_initial_GET_requests':1,'maximum_redirect_GET_requests':0,'maximum_bytes':CAP,'whole_operation_deadline_seconds':SECONDS,'deadline_mechanism':'SIGALRM ITIMER_REAL covers open/read/hash/parse; no retry or fallback endpoint','stream_chunk_bytes':262144,'persisted':'Exact HTTP header/status/Content-Length, byte count, digest, parse fields, timing and safe failure code; no raw body, page text or executable content','inherited_bounded_capture_plan':'Reviewed E54 streaming plan retained:64MiB cap/30s complete operation, one GET without redirect or retry. Old E52 size issue already resolved.','source_sha256':hashlib.sha256(source).hexdigest(),'TLS_verification_enabled':True,'inherited_proxy_preserved':True,'paid_API_GPU_model_native_DATA_Gold_calls':0,'attempt_nonretryable':True,'source_checkpoint':'5e426edd8ada8c0fdcec38ec8807054affc89e3c'}
 put('JEV_STREAM_PLAN.v1.json',plan)
 put('JEV_STREAM_ATTEMPT.v1.json',{'epoch':57,'started_at':now(),'once_only':True,'maximum_GETs':1,'source_sha256':plan['source_sha256']})
 start=time.monotonic();h=hashlib.sha256();count=0;page=Page();decoder=codecs.getincrementaldecoder('utf-8')('strict');r={'schema_version':1,'epoch':57,'url':URL,'status':'ATTEMPT_RESERVED','initial_GET_attempts':0,'redirect_GETs':0,'raw_body_persisted':False,'TLS_verification_enabled':True,'request_started_at':now(),'maximum_bytes':CAP,'deadline_seconds':SECONDS,'complete_body':False,'error_code':None,'live_open_weights_Capability_leader':None}
 previous=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,SECONDS)
 try:
  ca=os.environ.get('SSL_CERT_FILE')or os.environ.get('REQUESTS_CA_BUNDLE')
  opener=urllib.request.build_opener(urllib.request.ProxyHandler(),urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca if ca else None)),NoRedirect())
  req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0 official-source-read-only','Accept-Encoding':'identity'})
  r['initial_GET_attempts']=1
  with opener.open(req,timeout=SECONDS)as response:
   r.update(HTTP_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),Content_Length=response.headers.get('Content-Length'),etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
   assert response.status==200 and response.url==URL,'EXACT_OFFICIAL_HTTP200_URL'
   declared=response.headers.get('Content-Length');assert declared is None or 0<=int(declared)<=CAP,'DECLARED_SIZE_OVER64MIB_NO_BODY_RETRY'
   while True:
    assert time.monotonic()-start<SECONDS,'WHOLE_OPERATION_30S_DEADLINE'
    block=response.read(min(262144,CAP-count+1))
    if not block:break
    count+=len(block);assert count<=CAP,'STREAM_HARD_CAP64MIB'
    h.update(block);page.feed(decoder.decode(block))
   page.feed(decoder.decode(b'',final=True));page.close()
  if declared is not None:assert count==int(declared),'HTTP_CONTENT_LENGTH_MISMATCH'
  r['complete_body']=True
  text=' '.join(' '.join(page.text).split())
  version=re.search(r'Open weights and API offerings \(board (v[\d.]+) \)',text)
  header=re.search(r'method (v[\d.]+) \(([^)]+)\) · updated (\d{4}-\d{2}-\d{2})',text)
  headline=re.search(r'Capability Score averages Intelligence and Calibration \. (.+?) leads the Jev-class systems with (\d+(?:\.\d+)?)',text)
  ranked=re.search(r'(\d+) ranked open-weights systems',text)
  assert version and header and headline and ranked and page.rows,'OFFICIAL_BOARD_FIELDS_CHANGED_NO_RETRY'
  assert len({x['model_id']for x in page.rows})==len(page.rows),'UNIQUE_CAPABILITY_ROWS'
  leader={'model':headline.group(1),**page.rows[0]};assert round(leader['Capability'],1)==float(headline.group(2)),'HEADLINE_ROW_SCORE_CONSISTENCY'
  r.update(status='FRESH_OFFICIAL_JEV_STREAM_COMPLETE',board_version=version.group(1),board_updated=header.group(3),method_revision=header.group(1),method_header=header.group(2),live_open_weights_Capability_leader=leader,headline_display=float(headline.group(2)),ranked_open_weights_systems=int(ranked.group(1)),verified_capability_rows_count=len(page.rows),first_official_capability_rows=page.rows[:3],all_capability_rows_sha256=hashlib.sha256(json.dumps(page.rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),RuVector_page_text_matches=len(re.findall('ruvector',text,re.I)),submission_reference_page_matches=text.count('7FEA2738'),official_RuVector_rank=None,official_RuVector_score=None,official_RuVector_eligibility='UNVERIFIED',current_questions_scorer_license_access='UNVERIFIED. Parsed aggregate board does not establish exact labelled current300 questions/scorer, redistribution licence, sealed full-set access or submission eligibility.',current_queue_progress='UNVERIFIED',historical_original_queue_position=115,queue_is_rank=False,historical_access_probes_repeated=False)
 except Exception as e:
  msg=str(e);r.update(status='OFFICIAL_JEV_STREAM_FAILED_CLOSED_NO_RETRY',error_class=type(e).__name__,error_code=msg if re.fullmatch('[A-Z_0-9]+',msg)else type(e).__name__,live_open_weights_Capability_leader=None)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
 r.update(bytes_streamed=count,body_sha256=h.hexdigest()if r['complete_body']else None,partial_stream_sha256=None if r['complete_body']else h.hexdigest(),wall_seconds=time.monotonic()-start,retrieved_at=now(),raw_page_text_or_body_files_retained=False,model_native_DATA_Gold_paid_GPU_calls=0,outgoing_messages_submissions_author_contacts=0,official_goal_verified=False)
 ref=put('JEV_STREAM_RESULT.actual.v1.json',r)
 print(json.dumps({'result':ref,'status':r['status'],'bytes_streamed':count,'Content_Length':r.get('Content_Length'),'wall_seconds':r['wall_seconds'],'leader':r['live_open_weights_Capability_leader'],'error_code':r['error_code']}))
 return 0 if r['status']=='FRESH_OFFICIAL_JEV_STREAM_COMPLETE'else 1
if __name__=='__main__':raise SystemExit(main())
