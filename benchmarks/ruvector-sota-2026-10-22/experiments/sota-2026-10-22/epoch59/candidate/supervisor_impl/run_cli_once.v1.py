"""One E59 bounded CLI attempt, launched in its own session by Root.

Required launch containment: ``setsid taskset -c 0,1 <python> -I -B ...``.
The outer process then creates a separate supervisor session, verifies all
observed identities, reaps its child and reports closure. No retry entrypoint.
The 598-second final hard alarm leaves two seconds inside the 600-second cap.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path


def need(v,c):
    if not v:raise ValueError(c)


def sha(b):return hashlib.sha256(b).hexdigest()


def unique(rows):
    out={}
    for k,v in rows:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
    return out


def safe(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and
         not any(x.is_symlink() for x in (p,*p.parents)),'CANONICAL_NONSYMLINK_PATH');return p


def body(path):
    p=safe(path)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_size<=64*1024*1024,'BOUNDED_REGULAR_FILE');b=f.read(64*1024*1024+1);z=os.fstat(f.fileno())
    need(len(b)<=64*1024*1024 and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==
         (z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'STABLE_FILE');return b


def ref(path):
    b=body(path);return {'path':str(safe(path)),'bytes':len(b),'sha256':sha(b)}


def read_ref(r):
    need(type(r) is dict and set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int and
         r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'EXACT_TYPED_REF')
    b=body(r['path']);need(len(b)==r['bytes'] and sha(b)==r['sha256'],'REF_BYTES_SHA');return b


def parse(b):return json.loads(b,object_pairs_hook=unique,
    parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def put(path,value):
    p=safe(path);raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return ref(p)


def close_uncertain_direct_child(process,deadline):
    """Close only the owned unreaped Popen; no guessed process-group signal."""
    if process.poll() is not None:return True
    process.terminate()
    try:process.wait(timeout=max(.01,min(1.0,deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        process.kill()
        try:process.wait(timeout=max(.01,min(1.0,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:return False
    return process.poll() is not None


def hard_kill_owned(process,module,identity,identity_validated,exit_fn=os._exit):
    """Fallible proc/group observations cannot bypass final598 hard exit."""
    try:
        if process is not None and module is not None and identity is not None and identity_validated:
            try:
                if module.same_identity(identity,True):os.killpg(identity['pgid'],signal.SIGKILL)
                else:
                    for child in module.session_members(identity['pgid'],identity['sid']):
                        if child['sid']==identity['sid'] and module.same_identity(child,True):
                            try:os.kill(child['pid'],signal.SIGKILL)
                            except ProcessLookupError:pass
            except BaseException:pass
    finally:
        if process is not None:
            try:process.kill()  # owned Popen polls before signaling its direct child
            except BaseException:pass
        exit_fn(124)


def main():
    origin=time.monotonic();process=None;module=None;identity=None;identity_validated=False
    # The handler performs only identity-checked group kill and immediate exit.
    # A final hard interruption is an uncertain failed attempt, never retried.
    def hard_limit(signum,frame):
        hard_kill_owned(process,module,identity,identity_validated)
    signal.signal(signal.SIGALRM,hard_limit);signal.setitimer(signal.ITIMER_REAL,598)
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--expected-config-sha256',required=True)
    a=ap.parse_args();config_ref=ref(a.config);need(config_ref['sha256']==a.expected_config_sha256,'EXACT_OUTER_CONFIG')
    c=parse(read_ref(config_ref));out=safe(c['output_directory']);need(out.is_dir(),'EXISTING_PRIVATE_OUTPUT_DIRECTORY')
    read_ref(c['outer_source']);need(Path(c['outer_source']['path'])==Path(__file__).resolve(),
                                   'EXACT_OPERATIONAL_OUTER_SOURCE')
    read_ref(c['supervisor_source'])
    spec=importlib.util.spec_from_file_location('e59_supervisor_pinned',c['supervisor_source']['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    myself=module.process_identity(os.getpid())
    need(myself['pid']==myself['pgid']==myself['sid'],'OUTER_CLI_UNIQUE_SESSION_LEADER')
    attempt=put(out/'OUTER_ATTEMPT.actual.json',{'schema_version':1,'epoch':59,'config_ref':config_ref,
       'process_identity':myself,'origin_monotonic_seconds':origin,'worker_deadline_monotonic_seconds':origin+580,
       'TERM_deadline_monotonic_seconds':origin+590,'hard_exit_deadline_monotonic_seconds':origin+598,
       'maximum_inclusive_seconds':600,'automatic_retry':False})
    result={'schema_version':1,'epoch':59,'kind':module.KIND,'phase':c.get('phase'),
        'status':'OUTER_FAILED_CLOSED','errors':[],'attempt_ref':attempt,
        'owned_process_identities':[myself],'supervisor_exit_code':None,
        'all_owned_processes_closed':False,
        'kernel_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'PID_namespace':os.readlink('/proc/self/ns/pid')}
    handles=[];interrupted=[];seen={(myself['pid'],myself['startticks'])}
    def stop(signum,frame):interrupted.append(signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def observe():
        if not identity_validated:return
        for child in module.session_members(identity['pgid'],identity['sid']):
            key=(child['pid'],child['startticks'])
            if key not in seen:
                put(out/('OUTER_OBSERVED_PROCESS_%s_%s.actual.json'%key),
                    {'schema_version':1,'epoch':59,'process_identity':child,
                     'observed_monotonic_seconds':time.monotonic(),'outer_identity':myself})
                result['owned_process_identities'].append(child);seen.add(key)
    try:
        fence=module.load_fence(c);module.verify_admission(c)
        python=module.python_executable(c['python'],c['python_realpath'])
        read_ref(c['supervisor_source']);fence.verify_owner(c['owner'])
        need(time.monotonic()<origin+580,'PREFLIGHT_WITHIN_INCLUSIVE_BUDGET')
        paths=[out/'supervisor.stdout.raw',out/'supervisor.stderr.raw']
        for p in paths:
            handles.append(os.fdopen(os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb',buffering=0))
        argv=[python,'-I','-B',c['supervisor_source']['path'],'--config',a.config,
              '--expected-config-sha256',a.expected_config_sha256,
              '--deadline-monotonic-seconds',str(origin+580)]
        env=os.environ.copy();env.update(module.CPU_ENV)
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handles[0],stderr=handles[1],
                                 env=env,start_new_session=True,close_fds=True)
        put(out/'SUPERVISOR_POPEN_BORN.actual.json',{'schema_version':1,'epoch':59,
            'direct_child_pid':process.pid,'start_new_session_requested':True,
            'identity_observed':False,'spawn_returned_monotonic_seconds':time.monotonic()})
        identity=module.process_identity(process.pid)
        need(identity['pid']==identity['pgid']==identity['sid'] and identity['sid']!=myself['sid'],
             'EXACT_DISTINCT_SUPERVISOR_SESSION')
        identity_validated=True
        put(out/'SUPERVISOR_STARTED.actual.json',{'schema_version':1,'epoch':59,
            'process_identity':identity,'outer_identity':myself,'argv':argv,
            'started_monotonic_seconds':time.monotonic(),'supervisor_source_ref':c['supervisor_source']})
        result['owned_process_identities'].append(identity);seen.add((identity['pid'],identity['startticks']))
        while process.poll() is None:
            observe();need(not interrupted,'OUTER_INTERRUPTED')
            need(time.monotonic()<origin+590,'OUTER_TERM_DEADLINE')
            need(all(p.stat().st_size<=module.MAX_CAPTURE_BYTES for p in paths),'SUPERVISOR_CAPTURE_LIMIT')
            fence.verify_owner(c['owner']);time.sleep(.05)
        result['supervisor_exit_code']=process.returncode
        need(process.returncode==0,'SUPERVISOR_NONZERO_EXIT')
        s=parse(module.read_ref(module.ref(out/'SUPERVISOR_RESULT.actual.json')))
        need(s['errors']==[] and s['children_physically_closed'] is True,'SUPERVISOR_CHILD_CLOSURE')
        for child in s['owned_process_identities']:
            key=(child['pid'],child['startticks'])
            if key not in seen:result['owned_process_identities'].append(child);seen.add(key)
        result['status']='CLI_COMPLETE_AWAITING_INDEPENDENT_ACTUAL_AUDIT'
    except BaseException as error:
        result['errors'].append({'type':type(error).__name__,
          'code':str(error) if re.fullmatch('[A-Z0-9_]{1,150}',str(error)) else 'OUTER_EXCEPTION'})
    finally:
        try:
            if process is not None:
                module.best_effort_observe(observe,result['errors'])
                if not identity_validated:
                    result['supervisor_process_identity_uncertain']=True
                    result['uncertain_direct_child_pid']=process.pid
                    result['uncertain_direct_child_reaped']=close_uncertain_direct_child(process,origin+597)
                    result['supervisor_session_physically_closed']=False
                    result['errors'].append({'type':'RuntimeError','code':'SUPERVISOR_SESSION_IDENTITY_UNCERTAIN'})
                for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,2)):
                    if identity_validated and module.same_identity(identity,True):
                        try:os.killpg(identity['pgid'],sig)
                        except ProcessLookupError:pass
                    until=min(origin+597,time.monotonic()+seconds)
                    while process.poll() is None and time.monotonic()<until:time.sleep(.02)
                result['supervisor_exit_code']=process.poll()
                # A dead supervisor may have left native children in the unique session.
                if identity_validated:
                    for child in module.session_members(identity['pgid'],identity['sid']):
                        if module.same_identity(child,True):
                            try:os.kill(child['pid'],signal.SIGKILL)
                            except ProcessLookupError:pass
                    until=min(origin+597,time.monotonic()+1)
                    while module.session_members(identity['pgid'],identity['sid']) and time.monotonic()<until:time.sleep(.02)
                    result['supervisor_session_physically_closed']=not module.session_members(identity['pgid'],identity['sid'])
                    if not result['supervisor_session_physically_closed']:
                        result['errors'].append({'type':'RuntimeError','code':'SUPERVISOR_SESSION_CLOSURE_FAILED'})
                if process.poll() is None:
                    result['errors'].append({'type':'RuntimeError','code':'SUPERVISOR_DIRECT_CHILD_NOT_REAPED'})
        except BaseException as error:
            result['errors'].append({'type':type(error).__name__,'code':'SESSION_CLEANUP_OBSERVATION_FAILED'})
        finally:
            if process is not None:
                if identity_validated:
                    try:
                        if module.same_identity(identity,True):os.killpg(identity['pgid'],signal.SIGKILL)
                    except BaseException as error:
                        result['errors'].append({'type':type(error).__name__,'code':'VALIDATED_GROUP_FINAL_SIGNAL_FAILED'})
                if not module.close_owned_popen(process,origin+597,result['errors']):
                    result['errors'].append({'type':'RuntimeError','code':'DIRECT_SUPERVISOR_CLOSURE_UNCERTAIN'})
        for f in handles:
            try:f.flush();os.fsync(f.fileno());f.close()
            except OSError:pass
        for name in ('supervisor.stdout.raw','supervisor.stderr.raw'):
            if (out/name).exists():result[name.replace('.','_')+'_ref']=ref(out/name)
        result['wall_seconds_inclusive']=time.monotonic()-origin
        result['inclusive_limit_passed']=result['wall_seconds_inclusive']<=600
        # This process is still alive while writing; Q verifies its later physical
        # closure. Never report all-owned closure from inside an owned process.
        result['all_owned_processes_closed']=False
        if result['errors']:result['status']='OUTER_FAILED_CLOSED'
        put(out/'OUTER_RESULT.actual.json',result)
        signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if not result['errors'] and result['inclusive_limit_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
