"""Source-frozen E58 supervisor: no model/native imports or automatic retries.

run_cli_once.py creates this process as its own session leader. This supervisor,
the worker and any native child share one session/process group; the outer CLI
can close that exact group. Self-contained strict IO helpers precede all spawn.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import signal
import stat
import subprocess
import time
from pathlib import Path

KIND='REAL_NATIVE_TYPED_NLI_COVERAGE_ORDER_DEV'
BASE='db7b9b21f4a13e5fc3444a8fe40de26583558e2e'
MAX_CAPTURE_BYTES=8*1024*1024
CPU_ENV={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'NUMEXPR_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2',
         'RAYON_NUM_THREADS':'2','UV_THREADPOOL_SIZE':'2','TOKENIZERS_PARALLELISM':'false'}


def need(value,code):
    if not value:raise ValueError(code)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def unique(rows):
    out={}
    for k,v in rows:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
    return out


def parse(raw):
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def safe(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and
                     not any(x.is_symlink() for x in (p,*p.parents)),'CANONICAL_NONSYMLINK_PATH')
    return p


def body(path):
    p=safe(path)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_size<=64*1024*1024,'BOUNDED_REGULAR_FILE');raw=f.read(64*1024*1024+1);b=os.fstat(f.fileno())
    need(len(raw)<=64*1024*1024 and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==
         (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns),'STABLE_FILE')
    return raw


def read_ref(r):
    need(type(r) is dict and set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int and
         r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'EXACT_TYPED_REF')
    b=body(r['path']);need(len(b)==r['bytes'] and sha(b)==r['sha256'],'REF_BYTES_SHA');return b


def ref(path):
    b=body(path);return {'path':str(safe(path)),'bytes':len(b),'sha256':sha(b)}


def put(path,value):
    p=safe(path);need(p.parent.is_dir(),'EXISTING_OUTPUT_PARENT')
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return ref(p)


def process_identity(pid):
    raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw.rsplit(')',1)[1].split()
    return {'pid':pid,'startticks':int(f[19]),'pgid':int(f[2]),'sid':int(f[3])}


def same_identity(identity,live=False):
    try:
        p=Path('/proc')/str(identity['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return False
    return int(f[19])==identity['startticks'] and (not live or f[0] not in ('Z','X'))


def session_members(pgid,sid):
    out=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(')',1)[1].split()
            if f[0] not in ('Z','X') and (int(f[2])==pgid or int(f[3])==sid):
                out.append({'pid':int(p.name),'startticks':int(f[19]),'pgid':int(f[2]),'sid':int(f[3])})
        except (FileNotFoundError,ProcessLookupError):continue
    return sorted(out,key=lambda x:x['pid'])


def python_executable(leaf,real):
    # A venv Python may be a symlink. Every parent and resolved target is pinned.
    p=Path(leaf['path']);need(p.is_absolute() and p.parent.resolve()==p.parent and
                            not any(x.is_symlink() for x in p.parents),'PYTHON_SAFE_PARENTS')
    target=safe(real['path']);need(p.resolve()==target,'EXACT_PYTHON_REALPATH')
    raw=read_ref(real)
    need(leaf['bytes']==len(raw) and leaf['sha256']==sha(raw) and os.access(p,os.X_OK),
         'PYTHON_LEAF_MATCHES_FROZEN_RESOLVED_BINARY')
    return str(p)


def load_fence(c):
    read_ref(c['fence_source'])
    spec=importlib.util.spec_from_file_location('e58_supervisor_fence',c['fence_source']['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_owner(c['owner'])
    return module


def verify_admission(c):
    need(c['schema_version']==1 and c['epoch']==58 and c['kind']==KIND and
         c['phase'] in ('engineering','dev'),'EXACT_SUPERVISOR_KIND_PHASE')
    need(os.sched_getaffinity(0)=={0,1},'CPU_AFFINITY_EXACT_0_1')
    plan=parse(read_ref(c['plan']));freeze=parse(read_ref(c['source_freeze']))
    pub=parse(read_ref(c['source_publication']));q=parse(read_ref(c['Q_admission']));root=parse(read_ref(c['root_admission']))
    need(plan['epoch']==58 and plan['kind']==KIND and plan['source_checkpoint']==BASE,'EXACT_SOURCE_CHECKPOINT_PLAN')
    need(pub['checks_passed'] is True and pub['verified_actual_remote_readback'] is True and
         pub['actual_parent']==BASE and pub['commit']!=BASE and re.fullmatch('[0-9a-f]{40}',pub['commit']) and
         pub['source_plan_sha256']==c['plan']['sha256'] and
         pub['source_freeze_sha256']==c['source_freeze']['sha256'],'ACTUAL_SOURCE_PUBLICATION_READBACK')
    for approval in (q,root):
        need(approval['epoch']==58 and approval['kind']==KIND and approval['phase']==c['phase'] and
             approval['checks_passed'] is True and
             approval['owner_ref']==c['owner'] and approval['plan_ref']==c['plan'] and
             approval['source_freeze_ref']==c['source_freeze'] and
             approval['source_publication_ref']==c['source_publication'] and
             approval['worker_config_ref']==c['worker_config'],'EXACT_Q_AND_ROOT_EXECUTION_ADMISSION')
    need(q['scientific_execution_allowed'] is True and root['execution_allowed'] is True,
         'INDEPENDENT_SCIENTIFIC_AND_ROOT_EXECUTION_AUTHORITY')
    need(root['Q_admission_ref']==c['Q_admission'],'ROOT_BINDS_INDEPENDENT_Q_ADMISSION')
    rows=freeze['files'];need(type(rows) is list and len({r['path'] for r in rows})==len(rows),'UNIQUE_FROZEN_SOURCE_FILES')
    required=[c['worker'],c['fence_source'],c['supervisor_source'],c['outer_source'],*c['required_sources']]
    for r in required:
        read_ref(r)
        need(r in freeze['runtime_sources'] and any(x['bytes']==r['bytes'] and x['sha256']==r['sha256'] for x in rows),
             'OPERATIONAL_SOURCE_AND_PUBLIC_BODY_IN_FREEZE')
    need(Path(c['supervisor_source']['path'])==Path(__file__).resolve(),'EXACT_SUPERVISOR_OPERATIONAL_SOURCE')
    wc=parse(read_ref(c['worker_config']))
    need(wc['epoch']==58 and wc['kind']==KIND and wc['phase']==c['phase'] and
         wc['plan']==c['plan'] and wc['source_freeze']==c['source_freeze'] and
         wc['source_publication']==c['source_publication'] and
         wc['owner']==c['owner'],'EXACT_WORKER_CONFIG_SOURCE_AND_OWNER_BINDING')
    need(q['source_admission_ref']==wc['independent_admission'],'DISPATCH_BINDS_NONCIRCULAR_SOURCE_ADMISSION')
    return wc


def best_effort_observe(observe,errors):
    try:observe();return True
    except BaseException as e:
        errors.append({'type':type(e).__name__,'code':'PROCESS_OBSERVATION_FAILED_DURING_CLEANUP'})
        return False


def close_owned_popen(process,deadline,errors):
    if process is None:return True
    for method in ('terminate','kill'):
        try:
            if process.poll() is not None:return True
            getattr(process,method)()
        except BaseException as e:errors.append({'type':type(e).__name__,'code':'OWNED_POPEN_SIGNAL_FAILED'})
        try:process.wait(timeout=max(.01,min(.5,deadline-time.monotonic())))
        except BaseException as e:errors.append({'type':type(e).__name__,'code':'OWNED_POPEN_REAP_INCOMPLETE'})
    try:return process.poll() is not None
    except BaseException:return False


def main():
    started=time.monotonic();ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--expected-config-sha256',required=True)
    ap.add_argument('--deadline-monotonic-seconds',required=True,type=float);a=ap.parse_args()
    need(math.isfinite(a.deadline_monotonic_seconds) and started<a.deadline_monotonic_seconds<=started+580,
         'INCLUSIVE_WORKER_DEADLINE_FROM_OUTER')
    config_ref=ref(a.config);need(config_ref['sha256']==a.expected_config_sha256,'EXACT_SUPERVISOR_CONFIG')
    c=parse(read_ref(config_ref));out=safe(c['output_directory']);need(out.is_dir(),'EXISTING_PRIVATE_OUTPUT_DIRECTORY')
    myself=process_identity(os.getpid());need(myself['pid']==myself['pgid']==myself['sid'],'SUPERVISOR_UNIQUE_SESSION_LEADER')
    attempt=put(out/'SUPERVISOR_ATTEMPT.actual.json',{'schema_version':1,'epoch':58,'config_ref':config_ref,
                'process_identity':myself,'automatic_retry':False,'monotonic_started':started,
                'worker_deadline_monotonic_seconds':a.deadline_monotonic_seconds})
    result={'schema_version':1,'epoch':58,'kind':KIND,'phase':c.get('phase'),
            'status':'SUPERVISOR_FAILED_CLOSED','errors':[],'attempt_ref':attempt,
            'owned_process_identities':[myself],'worker_exit_code':None,
            'children_physically_closed':False,'all_owned_processes_closed':False,
            'kernel_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'PID_namespace':os.readlink('/proc/self/ns/pid')}
    process=None;fence=None;handles=[];interrupted=[];seen={(myself['pid'],myself['startticks'])}
    def stop(signum,frame):interrupted.append(signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def observe():
        for identity in session_members(myself['pgid'],myself['sid']):
            key=(identity['pid'],identity['startticks'])
            if key not in seen:
                put(out/('PROCESS_%s_%s.actual.json'%key),{'schema_version':1,'epoch':58,
                    'process_identity':identity,'observed_monotonic_seconds':time.monotonic(),
                    'supervisor_identity':myself})
                result['owned_process_identities'].append(identity);seen.add(key)
    def close_children():
        best_effort_observe(observe,result['errors'])
        for sig,window in ((signal.SIGTERM,2.0),(signal.SIGKILL,2.0)):
            members=[x for x in session_members(myself['pgid'],myself['sid']) if x['pid']!=myself['pid']]
            for identity in members:
                if same_identity(identity,True):
                    try:os.kill(identity['pid'],sig)
                    except ProcessLookupError:pass
            until=time.monotonic()+window
            while time.monotonic()<until:
                if process is not None:process.poll()
                left=[x for x in session_members(myself['pgid'],myself['sid']) if x['pid']!=myself['pid']]
                if not left:break
                time.sleep(.02)
        if process is not None:
            need(process.poll() is not None,'DIRECT_WORKER_REAPED')
            result['worker_exit_code']=process.returncode
        result['children_physically_closed']=not any(x['pid']!=myself['pid'] for x in session_members(myself['pgid'],myself['sid']))
    try:
        fence=load_fence(c);wc=verify_admission(c);fence.verify_owner(c['owner'])
        need(time.monotonic()<a.deadline_monotonic_seconds,'DEADLINE_BEFORE_SPAWN')
        python=python_executable(c['python'],c['python_realpath'])
        read_ref(c['worker']);read_ref(c['worker_config'])
        env=os.environ.copy();env.update(CPU_ENV)
        paths=[out/'worker.stdout.raw',out/'worker.stderr.raw']
        for p in paths:
            fd=os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            handles.append(os.fdopen(fd,'wb',buffering=0))
        argv=[python,'-I','-B',c['worker']['path'],'--config',c['worker_config']['path'],
              '--expected-config-sha256',c['worker_config']['sha256'],
              '--deadline-monotonic-seconds',str(a.deadline_monotonic_seconds)]
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handles[0],stderr=handles[1],
                                 env=env,start_new_session=False,close_fds=True)
        identity=process_identity(process.pid)
        need(identity['pgid']==myself['pgid'] and identity['sid']==myself['sid'],'WORKER_IN_OWNED_SESSION_GROUP')
        put(out/'WORKER_STARTED.actual.json',{'schema_version':1,'epoch':58,'process_identity':identity,
             'supervisor_identity':myself,'argv':argv,'config_ref':c['worker_config'],
             'started_monotonic_seconds':time.monotonic()})
        result['owned_process_identities'].append(identity);seen.add((identity['pid'],identity['startticks']))
        while process.poll() is None:
            observe();need(not interrupted,'SUPERVISOR_INTERRUPTED')
            need(time.monotonic()<a.deadline_monotonic_seconds,'INCLUSIVE_WORKER_DEADLINE')
            need(all(p.stat().st_size<=MAX_CAPTURE_BYTES for p in paths),'WORKER_RAW_CAPTURE_LIMIT')
            fence.verify_owner(c['owner']);time.sleep(.05)
        result['worker_exit_code']=process.returncode
        need(process.returncode==0,'WORKER_NONZERO_EXIT')
        need(all(p.stat().st_size<=MAX_CAPTURE_BYTES for p in paths),'WORKER_RAW_CAPTURE_LIMIT')
        worker_result_ref=ref(safe(wc['output_directory'])/'RESULT.actual.json')
        worker_result=parse(read_ref(worker_result_ref))
        need(worker_result['epoch']==58 and worker_result['kind']==KIND and
             worker_result['phase']==c['phase'] and worker_result['errors']==[] and
             worker_result['plan_sha256']==c['plan']['sha256'] and
             worker_result['source_freeze_sha256']==c['source_freeze']['sha256'], 'EXACT_WORKER_RAW_RESULT')
        for child in worker_result['owned_process_identities']:
            need(type(child) is dict and set(child)=={'pid','startticks','pgid','sid'} and
                 all(type(child[k]) is int and child[k]>1 for k in child) and
                 child['pgid']==myself['pgid'] and child['sid']==myself['sid'], 'WORKER_REPORTED_PROCESS_IN_OWNED_SESSION')
            key=(child['pid'],child['startticks'])
            if key not in seen:
                result['owned_process_identities'].append(child);seen.add(key)
        result['actual_results_ref']=worker_result_ref
        result['status']='WORKER_COMPLETED_AWAITING_INDEPENDENT_ACTUAL_AUDIT'
    except BaseException as error:
        result['errors'].append({'type':type(error).__name__,
                                'code':str(error) if re.fullmatch('[A-Z0-9_]{1,150}',str(error)) else 'SUPERVISOR_EXCEPTION'})
    finally:
        try:close_children()
        except BaseException as error:result['errors'].append({'type':type(error).__name__,'code':'CHILD_CLOSURE_FAILED'})
        finally:
            if not close_owned_popen(process,a.deadline_monotonic_seconds+10,result['errors']):
                result['errors'].append({'type':'RuntimeError','code':'DIRECT_WORKER_CLOSURE_UNCERTAIN'})
        for f in handles:
            try:f.flush();os.fsync(f.fileno());f.close()
            except OSError:pass
        for name in ('worker.stdout.raw','worker.stderr.raw'):
            if (out/name).exists():result[name.replace('.','_')+'_ref']=ref(out/name)
        result['wall_seconds']=time.monotonic()-started
        result['supervisor_resource_usage']={k:getattr(resource.getrusage(resource.RUSAGE_SELF),k)
                 for k in ('ru_utime','ru_stime','ru_maxrss','ru_inblock','ru_oublock')}
        result['child_resource_usage']={k:getattr(resource.getrusage(resource.RUSAGE_CHILDREN),k)
                 for k in ('ru_utime','ru_stime','ru_maxrss','ru_inblock','ru_oublock')}
        if result['errors']:result['status']='SUPERVISOR_FAILED_CLOSED'
        put(out/'SUPERVISOR_RESULT.actual.json',result)
    return 0 if not result['errors'] and result['children_physically_closed'] else 2


if __name__=='__main__':raise SystemExit(main())
