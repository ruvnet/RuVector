"""Root-only <=120s closed-process envelope; an attempt is never retried."""
import time
START=time.monotonic()
from pathlib import Path
import json,hashlib,os,signal,subprocess,sys
W=Path(__file__).resolve().parent
WORKER_SHA='f855d0f4e966f7ed777c9e393398a4c7885d1441f39042b1d8be9e50e3b3db35'
def put(p,v):
 with p.open('xb')as f:f.write((json.dumps(v,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
def snap(pid):
 try:f=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return{'pid':pid,'startticks':int(f[19]),'state':f[0]}
 except FileNotFoundError:return None
assert sys.flags.optimize==0 and sorted(os.sched_getaffinity(0))==[0,1];s=W/'acquire_int8_asset_once.py';assert hashlib.sha256(s.read_bytes()).hexdigest()==WORKER_SHA
put(W/'ROOT_INT8_ACQUISITION_ATTEMPT.actual.v1.json',{'epoch':57,'worker_source_sha256':WORKER_SHA,'maximum_inclusive_seconds':120,'automatic_retry':False});timed=False
with(W/'INT8_ACQUISITION_PRIVATE.stdout').open('xb')as out,(W/'INT8_ACQUISITION_PRIVATE.stderr').open('xb')as err:
 child=subprocess.Popen([sys.executable,'-I','-B',str(s)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True);identity=snap(child.pid);put(W/'ROOT_INT8_ACQUISITION_CHILD_STARTED.actual.v1.json',{'epoch':57,'identity':identity})
 try:code=child.wait(timeout=max(.1,115-(time.monotonic()-START)))
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=2)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=2)
 out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
wall=time.monotonic()-START;result={'epoch':57,'exit_code':code,'timed_out':timed,'wall_seconds':wall,'within120':wall<120,'child_identity':identity,'child_after_wait':snap(child.pid),'owned_group_closed':True,'child_spawn_scope':'Acquisition worker source spawns no descendants; parent created separate process group and waited exact child.','automatic_retry':False,'model_native_Gold_calls':0};put(W/'ROOT_INT8_ACQUISITION_OUTER.actual.v1.json',result);print(json.dumps(result));raise SystemExit(0 if code==0 and not timed and wall<120 else 19)
