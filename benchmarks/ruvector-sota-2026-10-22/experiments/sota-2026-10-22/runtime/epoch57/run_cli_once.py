"""Root-only once-per-phase CLI clock; no scientific authorization or retry."""
import time
START=time.monotonic()
import hashlib,importlib.util,json,os,signal,subprocess,sys
from pathlib import Path
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
ENGINEERING_KIND='REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV'
PILOT_KIND='REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT'
def write(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
 fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def snap(pid):
 try:
  f=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'startticks':int(f[19]),'state':f[0]}
 except(FileNotFoundError,ProcessLookupError):return None
if sys.flags.optimize!=0:raise ValueError('NONOPTIMIZED_PYTHON_REQUIRED')
config_path=Path(sys.argv[1]);config_bytes=config_path.read_bytes();assert hashlib.sha256(config_bytes).hexdigest()==sys.argv[2];c=json.loads(config_bytes)
assert c['epoch']==57 and c['CPU_affinity']==[0,1] and sorted(os.sched_getaffinity(0))==[0,1]
phase=c['phase'];assert phase in('engineering','retrieval_development');kind=ENGINEERING_KIND if phase=='engineering'else PILOT_KIND;assert c['kind']==kind
prefix='ENGINEERING'if phase=='engineering'else'RETRIEVAL';argv=c['argv'];assert len(argv)==18 and argv[1:3]==['-I','-B']
assert argv[0]=='/tmp/ruvector-sota-2026-10-22/epoch33-0522a2c521c9/runtime/.venv/bin/python'
assert argv[3]==str(W/'project/experiments/sota-2026-10-22/runtime/epoch57/supervisor.py') and argv[4:6]==['--phase',phase]
assert argv[6::2]==['--plan','--expected-plan-sha256','--admission','--expected-admission-sha256','--approval','--expected-approval-sha256']
pbytes=Path(argv[7]).read_bytes();assert hashlib.sha256(pbytes).hexdigest()==argv[9];p=json.loads(pbytes);assert p['epoch']==57 and p['kind']==kind and p['phase']==phase
assert p['supervisor_source']['path']==argv[3]and c['environment']==p['launch_environment'];assert hashlib.sha256(Path(argv[3]).read_bytes()).hexdigest()==p['supervisor_source']['sha256']
spec=importlib.util.spec_from_file_location('e57_root_preflight_supervisor',argv[3]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.fence(p['owner_ref'])
assert time.time()<=module.LATEST_START_UNIX
limit=(Path('/sys/fs/cgroup/memory.max')).read_text().strip();current=int((Path('/sys/fs/cgroup/memory.current')).read_text());assert limit!='max'and int(limit)-current>=3221225472
write(W/('ROOT_'+prefix+'_CLI_LAUNCH.actual.v1.json'),{'epoch':57,'kind':kind,'phase':phase,'started_monotonic':START,'argv':argv,'launch_config_sha256':sys.argv[2],'actual_pre_CLI_cgroup_headroom_bytes':int(limit)-current,'automatic_retry':False})
env=os.environ.copy();env.update(c['environment']);timed_out=False
with(W/(prefix+'_SUPERVISOR.stdout.txt')).open('xb')as stdout,(W/(prefix+'_SUPERVISOR.stderr.txt')).open('xb')as stderr:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,close_fds=True);identity=snap(child.pid)
 write(W/('ROOT_'+prefix+'_CLI_STARTED.actual.v1.json'),{'epoch':57,'kind':kind,'phase':phase,'supervisor_identity':identity})
 try:code=child.wait(timeout=max(.1,590-(time.monotonic()-START)))
 except subprocess.TimeoutExpired:
  timed_out=True;child.send_signal(signal.SIGTERM)
  try:code=child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=2)
 stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
wall=time.monotonic()-START
write(W/('ROOT_'+prefix+'_CLI_WALL.actual.v1.json'),{'epoch':57,'kind':kind,'phase':phase,'exit_code':code,'supervisor_identity':identity,'supervisor_after_wait':snap(child.pid),'timed_out':timed_out,'actual_CLI_wall_seconds_including_interpreter_bootstrap_capture':wall,'within_600_seconds':wall<600,'requires_independent_worker_native_physical_closure':True,'automatic_retry':False})
print(json.dumps({'exit_code':code,'CLI_wall_seconds':wall,'timed_out':timed_out}));raise SystemExit(code if not timed_out and wall<600 else 19)
