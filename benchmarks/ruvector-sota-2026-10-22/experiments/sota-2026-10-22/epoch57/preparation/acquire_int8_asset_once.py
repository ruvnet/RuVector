"""Root-admitted one-shot opaque public ONNX acquisition, no model execution."""
import time
START=time.monotonic()
from pathlib import Path
import hashlib,json,os,sys,urllib.request,urllib.parse,signal
W=Path(__file__).resolve().parent;D=W/'model-assets'
PLAN=Path('/dev/shm/ruvector-e57-candidate/MODEL_SOURCE_QUALIFICATION_AND_ASSET_PLAN.v2.json')
PLAN_SHA='6fdcef0548ebfa3ba4226409618e31263470e1cf4bcb1df6a47ecc90c76eb81a'
OWNER_SHA='a98d0e355925de7cf2fbe8711dc35afc4c3765e07fe5dfab8bd59f0309f733e5'
URL='https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3/onnx/model_quantized.onnx'
SIZE=34014426;SHA='6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4'
def put(p,v):
 with p.open('xb')as f:f.write((json.dumps(v,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
def fence():
 b=(W/'owner.json').read_bytes();assert hashlib.sha256(b).hexdigest()==OWNER_SHA;o=json.loads(b);assert o['epoch']==57 and o['generation']==23 and time.time()+120<o['expires_unix']and not(W/'release.request').exists();q=Path('/proc')/str(o['pid']);assert int((q/'stat').read_text().rsplit(')',1)[1].split()[19])==o['startticks'];s=(q/'fd/3').stat();assert(s.st_dev,s.st_ino)==(33,13626)and'FLOCK'in(q/'fdinfo/3').read_text()
def memory():
 m=int(Path('/sys/fs/cgroup/memory.max').read_text());c=int(Path('/sys/fs/cgroup/memory.current').read_text());return{'max':m,'current':c,'headroom':m-c}
def main():
 assert sys.flags.optimize==0 and sorted(os.sched_getaffinity(0))==[0,1];fence();b=PLAN.read_bytes();assert hashlib.sha256(b).hexdigest()==PLAN_SHA;p=json.loads(b);a=p['Root_only_once_acquisition_plan'];assert a['destination']==str(D/'model_quantized.onnx')and a['maximum_candidate_weight_GETs']==1 and a['expected_weight_sha256']==SHA and a['expected_weight_bytes']==SIZE
 before=memory();assert before['headroom']>=3355443200;assert not D.exists();D.mkdir(mode=0o700);v=os.statvfs(D);assert v.f_bavail*v.f_frsize>=100663296
 put(W/'INT8_ASSET_GET_ATTEMPT.actual.v1.json',{'epoch':57,'plan_sha256':PLAN_SHA,'original_pinned_URL':URL,'expected_bytes':SIZE,'expected_sha256':SHA,'before':before,'logical_weight_GET_maximum':1,'automatic_retry':False,'model_native_Gold_calls':0});partial=D/'model_quantized.onnx.partial';h=hashlib.sha256();n=0;error=None;host=None
 def expired(signum,frame):raise TimeoutError('OPAQUE_ASSET_OPERATION_90_SECOND_DEADLINE')
 previous=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,expired)
 try:
  signal.setitimer(signal.ITIMER_REAL,max(.001,90-(time.monotonic()-START)))
  req=urllib.request.Request(URL,headers={'User-Agent':'RuVector-public-benchmark-research/epoch57'})
  with urllib.request.urlopen(req,timeout=90)as response,partial.open('xb')as f:
   assert response.status==200;host=urllib.parse.urlsplit(response.geturl()).hostname;cl=response.headers.get('Content-Length');assert cl is None or int(cl)==SIZE
   while True:
    assert time.monotonic()-START<90
    chunk=response.read(min(1048576,SIZE-n+1))
    if not chunk:break
    n+=len(chunk);assert n<=SIZE;h.update(chunk);f.write(chunk)
   f.flush();os.fsync(f.fileno())
  assert n==SIZE and h.hexdigest()==SHA;fence();after=memory();assert after['headroom']>=3221225472
  final=D/'model_quantized.onnx';os.link(partial,final,follow_symlinks=False);partial.unlink();fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 except Exception as ex:error={'type':type(ex).__name__,'errno':getattr(ex,'errno',None),'HTTP_status':getattr(ex,'code',None)}
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
 result={'epoch':57,'checks_passed':error is None,'error':error,'original_pinned_URL':URL,'final_host_without_URL_query_or_headers':host,'observed_stream_bytes':n,'observed_stream_sha256':h.hexdigest(),'expected_bytes':SIZE,'expected_sha256':SHA,'before':before,'after':memory(),'wall_seconds':time.monotonic()-START,'partial_retained_on_failure':partial.exists(),'final_asset':{'path':str(D/'model_quantized.onnx'),'bytes':SIZE,'sha256':SHA}if error is None else None,'automatic_retry':False,'model_tokenizer_native_Gold_training_calibration_paid_GPU_calls':0};put(W/'INT8_ASSET_GET_RESULT.actual.v1.json',result);print(json.dumps({'checks_passed':error is None,'observed_bytes':n,'wall_seconds':result['wall_seconds']}));return 0 if error is None else 19
if __name__=='__main__':raise SystemExit(main())
