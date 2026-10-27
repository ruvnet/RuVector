"""E57 native process bridge; stdlib-only, no models, text, qrels or scoring.

Source contract: run_native(plan,bundle,external,counters,deadline,progress,state).
The exact frozen plan on disk is authoritative. NativeSnapshotLedger is imported
only from the hash-pinned reviewed E56 source. Expected totals are validation
bounds; only captured observed snapshots populate the fourteen native counters.
No retries. Failed captures preserve observed prefixes and uncertain outcomes.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

LEDGER_SHA='379588fcbe15f9db598c899740c54c280237d1853a3fea6575b72d0898ff21be'
MODEL_SHA='6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4'
BINARY_SHA='ff559e9fac2e3d29a8b32e890fdb047bcd4bd28f389b4731ad72148b221d5a2e'
GEOMETRY={'engineering':('REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV',128),
          'retrieval_development':('REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT',5183)}
KEYS=('native_index_initializations_attempted','native_index_initializations_completed',
      'native_insertions_attempted','native_insertions_completed','native_queries_attempted',
      'native_queries_completed','native_len_calls_attempted','native_len_calls_completed',
      'native_hit_source_validations','native_binary_loads_attempted','native_binary_loads_completed',
      'native_version_calls_attempted','native_version_calls_completed','Node_network_operations_attempted')


def need(value,code):
    if not value:raise ValueError(code)


def _unique(pairs):
    value={}
    for key,item in pairs:
        need(key not in value,'NATIVE_BRIDGE_DUPLICATE_JSON_KEY');value[key]=item
    return value


def _load(raw):
    return json.loads(raw,object_pairs_hook=_unique,parse_constant=lambda _:need(False,'NONFINITE_JSON'))


def _read(path,cap):
    path=Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path,*path.parents)),
         'ABSOLUTE_NONSYMLINK_FILE')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=cap,'REGULAR_FILE_BYTE_CAP')
        blocks=[];size=0
        while size<=cap:
            block=os.read(fd,min(1048576,cap+1-size))
            if not block:break
            blocks.append(block);size+=len(block)
        after=os.fstat(fd);raw=b''.join(blocks)
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(signature(before)==signature(after) and len(raw)==before.st_size and len(raw)<=cap,
             'STABLE_FILE_DURING_CAPTURE')
        return raw
    finally:os.close(fd)


def verify(ref):
    need(type(ref)is dict and set(ref)=={'path','bytes','sha256'} and type(ref['bytes'])is int
         and ref['bytes']>=0 and type(ref['sha256'])is str and len(ref['sha256'])==64,
         'CANONICAL_ASSET_REFERENCE')
    raw=_read(ref['path'],ref['bytes'])
    need(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'EXACT_ASSET_SIZE_SHA256')
    return Path(ref['path'])


def _reference(path,cap=16777216):
    raw=_read(path,cap);return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def put(path,value):
    path=Path(path)
    need(path.is_absolute() and path.parent.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)),
         'SAFE_NEW_OUTPUT')
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        written=os.write(fd,raw)
        need(type(written)is int and written>0 and written==len(raw),'SHORT_NATIVE_BRIDGE_OUTPUT_WRITE')
        os.fsync(fd)
    finally:os.close(fd)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def proc(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'startticks':int(fields[19]),'pgid':int(fields[2]),'sid':int(fields[3]),'state':fields[0]}
    except(FileNotFoundError,ProcessLookupError):return None


def expected_native(plan):
    g=GEOMETRY.get(plan.get('phase'))
    need(g and plan.get('schema_version')==1 and plan.get('epoch')==57 and plan.get('kind')==g[0]
         and plan.get('corpus_document_count')==g[1] and plan.get('query_count')==16
         and plan.get('top_k')==10,'FROZEN_NATIVE_GEOMETRY')
    ids=plan.get('selected_query_ids')
    need(type(ids)is list and len(ids)==16 and all(type(x)is str and x for x in ids)
         and len(set(ids))==16,'FROZEN_QUERY_IDENTITIES')
    need(plan.get('bge_model',{}).get('sha256')==MODEL_SHA,'INT8_MODEL_PROVENANCE_PIN')
    return dict(zip(KEYS,(1,1,g[1],g[1],16,16,2,2,160,1,1,1,1,0)))


def load_ledger(ref):
    need(ref['sha256']==LEDGER_SHA,'REVIEWED_NATIVE_LEDGER_PIN');path=verify(ref)
    spec=importlib.util.spec_from_file_location('ruvector_e57_reviewed_native_accounting',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(tuple(module.KEYS)==KEYS and module.MAX_BYTES==4194304 and module.MAX_RECORDS==10000,
         'REVIEWED_NATIVE_LEDGER_CONTRACT')
    return module.NativeSnapshotLedger


def _plan(plan):
    path=Path(plan['path']);raw=_read(path,2097152);frozen=_load(raw);sha=hashlib.sha256(raw).hexdigest()
    need(type(frozen)is dict and all(frozen.get(k)==v for k,v in plan.items() if k not in ('path','_sha256')),
         'FROZEN_PLAN_CONTEXT_AGREEMENT')
    need(plan.get('_sha256',sha)==sha,'FROZEN_PLAN_SHA256')
    expected_native(frozen)
    for key in ('node','native_binary','native_helper_source','native_accounting_source'):verify(frozen[key])
    need(frozen['native_binary']['sha256']==BINARY_SHA,'FROZEN_NATIVE_BINARY_PIN')
    for key,upper in (('native_helper_max_seconds',60),('native_process_timeout_seconds',85)):
        need(type(frozen.get(key))in (int,float) and math.isfinite(frozen[key]) and 0<frozen[key]<=upper,
             'NATIVE_TIME_CAP')
    need(frozen['native_process_timeout_seconds']>=frozen['native_helper_max_seconds']+5,
         'NATIVE_CLEANUP_TIME_RESERVE')
    return frozen,path,sha


def _validate_binding(value,bindings,plan,plan_sha,bundle_ref,identity,helper_ref,progress_ref):
    need(value.get('schema_version')==1 and value.get('epoch')==57 and value.get('kind')==plan['kind']
         and value.get('phase')==plan['phase'] and value.get('plan_sha256')==plan_sha
         and value.get('bundle_sha256')==bundle_ref['sha256'] and value.get('helper_pid')==identity['pid']
         and value.get('helper_source_sha256')==helper_ref['sha256']
         and value.get('native_binary_sha256')==BINARY_SHA and value.get('binary_version')=='2.3.0',
         'NATIVE_TERMINAL_SOURCE_BINDINGS')
    need(bindings.get('schema_version')==1 and bindings.get('epoch')==57 and bindings.get('kind')==plan['kind']
         and bindings.get('phase')==plan['phase'] and bindings.get('helper_pid')==identity['pid']
         and bindings.get('helper_startticks')==identity['startticks'] and bindings.get('plan_sha256')==plan_sha
         and bindings.get('bundle_sha256')==bundle_ref['sha256']
         and bindings.get('helper_source_sha256')==helper_ref['sha256']
         and bindings.get('progress_path')==progress_ref['path']
         and bindings.get('maximum_progress_bytes')==4194304 and bindings.get('maximum_progress_rows')==10000
         and bindings.get('helper_max_seconds')==plan['native_helper_max_seconds']
         and type(bindings.get('process_clock_origin_milliseconds'))in (int,float)
         and math.isfinite(bindings['process_clock_origin_milliseconds']),
         'NATIVE_PROCESS_BINDING_RECEIPT')
    need(value.get('progress')==dict(progress_ref,rows=plan['corpus_document_count']+37),
         'NATIVE_PROGRESS_TERMINAL_HASH_BINDING')
    rows=value.get('queries')
    need(type(rows)is list and [r.get('query_id') for r in rows]==plan['selected_query_ids'],
         'NATIVE_TERMINAL_QUERY_ORDER')
    need(value.get('corpus_document_count')==plan['corpus_document_count'] and value.get('query_count')==16
         and value.get('top_k')==10,'NATIVE_TERMINAL_GEOMETRY')
    for row in rows:
        need(type(row.get('ids'))is list and len(row['ids'])==10 and len(set(row['ids']))==10
             and row.get('rank_scores')==list(range(10,0,-1)),'NATIVE_TERMINAL_TOP10_RANKS')


def run_native(plan,bundle,external,counters,deadline,progress,state):
    """One child, one ledger capture, no retry; helper remains in supervisor group."""
    frozen,plan_path,plan_sha=_plan(plan);external=Path(external)
    need(external.is_absolute() and external.is_dir() and not any(p.is_symlink()for p in (external,*external.parents)),
         'PRIVATE_NATIVE_OUTPUT_DIRECTORY')
    need(bundle.get('plan_sha256')==plan_sha and bundle.get('epoch')==57 and bundle.get('kind')==frozen['kind']
         and bundle.get('phase')==frozen['phase'],'BUNDLE_FROZEN_PLAN_IDENTITY')
    expected=expected_native(frozen);ledger=load_ledger(frozen['native_accounting_source'])(counters,expected)
    state['accounting']=ledger.receipt();state['native_progressobserved']=None
    bundle_ref=put(external/'native-bundle.json',bundle);output=external/'native-result.json'
    command=[frozen['node']['path'],frozen['native_helper_source']['path'],str(plan_path),bundle_ref['path'],str(output)]
    need(counters.get('native_corpus_batches_attempted',0)==0,'ONE_NATIVE_CHILD_PER_PHASE')
    # This is an observed parent launch attempt, not a native operation count.
    counters['native_corpus_batches_attempted']=1
    state.update(schema_version=1,epoch=57,kind=frozen['kind'],phase=frozen['phase'],command=command,
                 bundle=bundle_ref,plan_sha256=plan_sha,child_closed=None,exit_code=None,
                 bridge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    child=None;identity=None;error=None;started=time.monotonic();capture_error=None
    stdout_path=external/'native.stdout.txt';stderr_path=external/'native.stderr.txt'
    try:
        need(os.environ.get('RAYON_NUM_THREADS')=='2' and os.environ.get('UV_THREADPOOL_SIZE')=='2',
             'NATIVE_CPU_THREAD_ENV')
        remaining=deadline();need(type(remaining)in (int,float) and remaining>5,'NATIVE_REMAINING_CLEANUP_RESERVE')
        timeout=min(frozen['native_process_timeout_seconds']-5,remaining-5)
        progress('NATIVE_HELPER_START_BEFORE')
        with stdout_path.open('xb')as stdout,stderr_path.open('xb')as stderr:
            os.chmod(stdout_path,0o600);os.chmod(stderr_path,0o600)
            try:
                child=subprocess.Popen(command,cwd=external,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                       close_fds=True,start_new_session=False)
                identity=proc(child.pid);state['identity']=identity
                need(identity and identity['pgid']==os.getpgrp() and identity['sid']==os.getsid(0),'NATIVE_CHILD_SESSION')
                child.wait(timeout=timeout)
            finally:
                if child is not None and child.poll()is None:
                    now=proc(child.pid)
                    # Popen's unreaped child handle owns this PID even before /proc identity capture.
                    need(now and (identity is None or now['startticks']==identity['startticks']),'OWNED_NATIVE_CLEANUP')
                    child.send_signal(signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
                stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
    except Exception as exc:error=exc
    finally:
        state.update(child_closed=child is not None and child.poll()is not None,
                     exit_code=None if child is None else child.returncode,wall_seconds=time.monotonic()-started)
        for name,path in (('stdout',stdout_path),('stderr',stderr_path)):
            if path.exists():
                try:state[name]=_reference(path,1048576)
                except Exception as exc:state[name+'_capture_error']=type(exc).__name__+':'+str(exc);error=error or exc
        try:ledger.capture(external/'native-progress.jsonl')
        except Exception as exc:capture_error=exc;state['capture_error']=type(exc).__name__+':'+str(exc)
        state['accounting']=ledger.receipt();state['native_progressobserved']=ledger.receipt()['latest_validated_snapshot']
        # Persist closure before examining a terminal result; partial observations remain durable.
        put(external/'native-process-closure.json',state)
    if error is not None:raise error
    if capture_error is not None:raise capture_error
    need(child is not None and child.returncode==0 and identity,'NATIVE_HELPER_COMPLETED')
    value=_load(_read(output,2097152));bindings_ref=value.get('bindings')
    need(type(bindings_ref)is dict and bindings_ref.get('path')==str(external/'native-bindings.json'),'NATIVE_BINDINGS_PATH')
    verify(bindings_ref)
    bindings=_load(_read(bindings_ref['path'],16384));progress_ref=ledger.receipt()['capture_ref']
    _validate_binding(value,bindings,frozen,plan_sha,bundle_ref,identity,frozen['native_helper_source'],progress_ref)
    need(ledger.last_milliseconds<=frozen['native_helper_max_seconds']*1000,'NATIVE_PROGRESS_TIME_BOUND')
    ledger.finalize(value);state['accounting']=ledger.receipt();state['native_progressobserved']=ledger.receipt()['latest_validated_snapshot']
    put(external/'native-accounting-final.json',state['accounting'])
    need(state['accounting']['qualified'],'NATIVE_SNAPSHOT_ACCOUNTING_QUALIFIED')
    counters['native_corpus_batches_completed']=counters.get('native_corpus_batches_completed',0)+1
    counters['native_corpus_batches']=counters.get('native_corpus_batches',0)+1
    progress('NATIVE_HELPER_COMPLETE');return value


native=run_native
