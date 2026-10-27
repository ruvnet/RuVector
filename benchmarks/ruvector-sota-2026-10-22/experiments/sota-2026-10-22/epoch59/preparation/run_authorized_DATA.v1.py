"""Root-admitted, once-only label-blind DATA supervision; no scientific authority."""
import argparse,hashlib,importlib.util,json,os,signal,stat,subprocess,time
from pathlib import Path
O=Path('/dev/shm/ruvector-e59-data/local-preparation');W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983');S=Path('/dev/shm/ruvector-e59-data/data-v1/custody.py')
SOURCE_SHA='a91780dce6c730cacb5b21f26ebea4684c893229c0821d7724d21abe1e6ac363'
F=W/'guardian27/owner_fence.py';FENCE_SHA='cc0f6f56e14998820dfad60752e3579ce09f1b89deae0ef6c57e47cdbe1dbbc5'
OWNER={'path':str(W/'guardian27/owner.json'),'bytes':831,'sha256':'28f34b4b0af71fbd39d94a05766ba7bf61798ae540419d1838ea3ca21d5cce9f'}
def read(p):
 p=Path(p);assert p.resolve()==p and not any(x.is_symlink()for x in [p,*p.parents]);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb')as f:a=os.fstat(f.fileno());assert stat.S_ISREG(a.st_mode);b=f.read();z=os.fstat(f.fileno())
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 return b
def put(n,v):
 p=O/n;b=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 d=os.open(O,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d);return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--admission',required=True);ap.add_argument('--expected-admission-sha256',required=True);ap.add_argument('--expected-admission-bytes',type=int,required=True);args=ap.parse_args();A=Path(args.admission);expected=args.expected_admission_sha256
 assert A==W/'LOCAL_DATA_ADMISSION.v1.json' and os.sched_getaffinity(0)<={0,1}
 raw=read(A);assert len(raw)==args.expected_admission_bytes and hashlib.sha256(raw).hexdigest()==expected
 a=json.loads(raw);assert a['epoch']==59 and a['mode']=='DATA' and a['phase']=='BLIND_CATALOG_PREPARATION_ONLY' and a['owner_ref']==OWNER
 assert hashlib.sha256(read(S)).hexdigest()==SOURCE_SHA and hashlib.sha256(read(F)).hexdigest()==FENCE_SHA
 spec=importlib.util.spec_from_file_location('e59_data_execution_owner_fence',F);fence=importlib.util.module_from_spec(spec);spec.loader.exec_module(fence);fence.verify_owner(OWNER)
 assert not(O/'DATA_EXECUTION_INTENT.v1.json').exists()and not(W/'private-data/DATA_ATTEMPT.json').exists()
 put('DATA_EXECUTION_INTENT.v1.json',{'epoch':59,'mode':'DATA','root_admission_sha256':expected,'once_only':True,'source_sha256':SOURCE_SHA,'actual_source_text_decodes_before_dispatch':0,'model_native_Gold_calls':0})
 command=['timeout','--signal=TERM','--kill-after=2s','116s','taskset','-c','0,1','/tmp/ruvector-sota-2026-10-22/epoch33-0522a2c521c9/runtime/.venv/bin/python','-I','-B',str(S),'DATA','--admission',str(A),'--expected-admission-sha256',expected,'--archive','/dev/shm/ruvector-sota-2026-10-22/epoch48-37b29244dcf5/assets/data/contract-nli-eced6528.zip']
 started=time.monotonic();env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false');p=None;identity=None;timed_out=False;supervisor_errors=[];stdout=b'';stderr=b''
 try:
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=env)
  f=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split();identity={'pid':p.pid,'startticks':int(f[19]),'pgid':os.getpgid(p.pid),'sid':os.getsid(p.pid)};assert identity['pid']==identity['pgid']==identity['sid'];put('DATA_PROCESS_STARTED.v1.json',{'epoch':59,'mode':'DATA','identity':identity,'root_admission_sha256':expected})
  stdout,stderr=p.communicate(timeout=max(.1,119-(time.monotonic()-started)))
 except Exception as exc:
  supervisor_errors.append({'type':type(exc).__name__,'code':'DATA_SUPERVISION_EXCEPTION_NO_RETRY'});timed_out=isinstance(exc,subprocess.TimeoutExpired)
 finally:
  if p is not None and p.poll()is None:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   stdout,stderr=p.communicate(timeout=1)
 live=[]
 if p is not None:
  for x in Path('/proc').iterdir():
   if not x.name.isdigit():continue
   try:f=(x/'stat').read_text().rsplit(')',1)[1].split()
   except FileNotFoundError:continue
   if f[0]not in('Z','X')and(int(f[2])==p.pid or int(f[3])==p.pid):live.append({'pid':int(x.name),'pgid':int(f[2]),'sid':int(f[3]),'state':f[0]})
 try:fence.verify_owner(OWNER)
 except Exception as exc:supervisor_errors.append({'type':type(exc).__name__,'code':'OWNER_INVALID_AFTER_DATA'})
 report={'schema_version':1,'epoch':59,'phase':'BLIND_CATALOG_PREPARATION_ONLY','mode':'DATA','root_admission_sha256':expected,'exit_code':None if p is None else p.returncode,'wall_seconds':time.monotonic()-started,'outer_fallback_timeout':timed_out,'identity':identity,'direct_timeout_process_physically_reaped':p is not None and p.poll()is not None,'live_processes_in_owned_group_or_session':live,'supervisor_errors':supervisor_errors,'checks_passed':p is not None and p.returncode==0 and not timed_out and not live and not supervisor_errors,'stdout_bytes':len(stdout),'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_bytes':len(stderr),'stderr_sha256':hashlib.sha256(stderr).hexdigest(),'new_NLI_native_Gold_paid_GPU_calls':0}
 r=put('DATA_ROOT_AUTHORIZED_EXECUTION.actual.v1.json',report);print(json.dumps({'execution_receipt':r,**report},sort_keys=True));print(stdout.decode()[:12000]);print(stderr.decode()[:4000]);return 0 if report['checks_passed']else 1
if __name__=='__main__':raise SystemExit(main())
