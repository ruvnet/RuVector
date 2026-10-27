"""One bounded detached guardian launch; never retries or dispatches science."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
G=Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983/guardian27')
SCRIPT=G/'lock_supervisor.py'
SCRIPT_SHA='04846475a4df178f3a1b97398beecf464cd104aac033c19ac4f3fcc54318c9bc'
def put(p,x):
 raw=(json.dumps(x,indent=2,sort_keys=True)+'\n').encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 d=os.open(G,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
def main():
 assert not sys.flags.optimize and len(sys.argv)==1 and os.sched_getaffinity(0)=={0,1}
 assert Path(__file__).resolve()==G/'launch_once.py'
 assert not any(p.is_symlink()for p in[SCRIPT,*SCRIPT.parents])
 b=SCRIPT.read_bytes();assert len(b)==4358 and hashlib.sha256(b).hexdigest()==SCRIPT_SHA
 for name in['owner.json','launch.intent.json','launch.actual.json','release.request','released.json','guardian.stdout.log','guardian.stderr.log']:
  assert not os.path.lexists(G/name),'existing launch evidence: STOP, no retry'
 put(G/'launch.intent.json',{'epoch':59,'generation':27,'guardian_source_sha256':SCRIPT_SHA,'monotonic':time.monotonic(),'automatic_retry':False,'science_authority':False})
 out=os.open(G/'guardian.stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 err=os.open(G/'guardian.stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  child=subprocess.Popen([sys.executable,'-I','-B',str(SCRIPT)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
 finally:os.close(out);os.close(err)
 deadline=time.monotonic()+10;owner=None
 while time.monotonic()<deadline:
  if (G/'owner.json').exists():
   try:owner=json.loads((G/'owner.json').read_text())
   except json.JSONDecodeError:owner=None
   if owner is not None:break
  if child.poll()is not None:break
  time.sleep(.05)
 ready=owner is not None and owner.get('pid')==child.pid and owner.get('generation')==27 and owner.get('launch_mode')=='DETACHED_BOUNDED_SESSION'
 result={'epoch':59,'generation':27,'pid':child.pid,'status':'OWNER_READY'if ready else 'STOP_RECONCILE_EXISTING_LAUNCH_NO_RETRY','launcher_observed_child_exit_code':child.poll(),'launcher_parent_exits_after_result':True,'automatic_retry':False,'native_model_Gold_Git_calls':0}
 if ready:
  stat=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
  assert int(stat[19])==owner['startticks']and stat[0]not in('Z','X')and os.getsid(child.pid)==os.getpgid(child.pid)==child.pid
  raw=(G/'owner.json').read_bytes();result['owner_ref']={'path':str(G/'owner.json'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 put(G/'launch.actual.json',result);print(json.dumps(result),flush=True)
 return 0 if ready else 73
if __name__=='__main__':raise SystemExit(main())
