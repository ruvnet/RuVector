"""One-shot E57 outer supervisor: source/hash/lock admission and physical cleanup.

The absolute 600-second clock starts on entry before preflight hashing/imports.
No worker retry. Owned process session is monitored and closed, with 570-second
worker deadline, TERM at 580 and KILL by 585, leaving capture/cleanup margin.
"""
import time
START=time.monotonic()
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
ENGINEERING_KIND='REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV'
PILOT_KIND='REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT'
KIND=ENGINEERING_KIND
LATEST_START_UNIX=1793122500
LATEST_FINISH_UNIX=1793123100
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
LIMIT=START+600
DENY={'486888de9602bbb5d817a55f707b4fcfd11c65e0','3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'}
ENV={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2',
     'BLIS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','MKL_DYNAMIC':'FALSE','OMP_THREAD_LIMIT':'2',
     'RAYON_NUM_THREADS':'2','UV_THREADPOOL_SIZE':'2','TOKENIZERS_PARALLELISM':'false',
     'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}


def require(x,code):
    if not x:raise ValueError(code)


def deadline():
    require(time.monotonic()<LIMIT,'OUTER_ABSOLUTE_600_SECOND_DEADLINE')


def regular(p):
    p=Path(p);require(p.is_absolute() and p.is_file() and not any(q.is_symlink() for q in [p,*p.parents]),'ABSOLUTE_REGULAR_NON_SYMLINK_FILE');return p


def ref(path):
    p=regular(path);h=hashlib.sha256();n=0
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        b=os.fstat(fd)
        while True:
            deadline();chunk=os.read(fd,1048576)
            if not chunk:break
            h.update(chunk);n+=len(chunk)
        a=os.fstat(fd)
        require((b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
                and n==b.st_size,'UNCHANGED_STREAM_HASH_FILE')
    finally:os.close(fd)
    return {'path':str(p),'bytes':n,'sha256':h.hexdigest()}


def exact(r):
    require(ref(r['path'])==r,'EXACT_SOURCE_OR_ASSET_HASH');return Path(r['path'])


def parse(r):return json.loads(exact(r).read_bytes())


def put(path,value):
    p=Path(path);require(p.parent.is_dir() and not p.is_symlink(),'ONCE_RESULT_PARENT')
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def snap(pid):
    try:
        p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();status=(p/'status').read_text()
        return {'pid':pid,'startticks':int(v[19]),'ppid':int(v[1]),'pgid':int(v[2]),'sid':int(v[3]),'state':v[0],
                'wait_status':int(v[49]),'FDSize':next(int(x.split(':')[1]) for x in status.splitlines() if x.startswith('FDSize:')),
                'statm':[int(x) for x in (p/'statm').read_text().split()]}
    except (FileNotFoundError,ProcessLookupError):return None


def fence(r):
    owner=parse(r);s=snap(owner['pid']);require(owner['epoch']==57 and owner['generation']==23 and owner['pid']==186899
        and owner['startticks']==44193256 and s and s['startticks']==owner['startticks'] and s['state'] not in ('Z','X')
        and time.time()<owner['expires_unix'] and not (W/'release.request').exists() and os.sched_getaffinity(owner['pid'])=={0,1},'LIVE_SCOPED_GUARDIAN23')
    p=regular(owner['lock']);st=p.stat();fd=os.stat('/proc/'+str(owner['pid'])+'/fd/3')
    require((st.st_dev,st.st_ino)==(fd.st_dev,fd.st_ino)==(owner['lock_device'],owner['lock_inode'])==(33,13626),'ORIGINAL_LOCK_INODE')
    info=Path('/proc/'+str(owner['pid'])+'/fdinfo/3').read_text();require('FLOCK' in info and str(owner['pid']) in info and ':'+str(owner['lock_inode']) in info,'OWNER_KERNEL_FLOCK')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['kernel_boot_id']
            and os.readlink('/proc/self/ns/pid')==owner['PID_namespace'],'OWNER_BOOT_NAMESPACE')
    lockfd=os.open(owner['lock'],os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(lockfd,fcntl.LOCK_UN);raise ValueError('SCOPED_LOCK_WAS_FREE')
    finally:os.close(lockfd)
    return owner


def main():
    global KIND
    if sys.flags.optimize!=0:raise ValueError('NONOPTIMIZED_PYTHON_REQUIRED')
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['engineering','retrieval_development'],required=True);p.add_argument('--plan',required=True);p.add_argument('--expected-plan-sha256',required=True)
    p.add_argument('--admission',required=True);p.add_argument('--expected-admission-sha256',required=True)
    p.add_argument('--approval',required=True);p.add_argument('--expected-approval-sha256',required=True);a=p.parse_args()
    KIND=ENGINEERING_KIND if a.phase=='engineering' else PILOT_KIND
    prefix='ENGINEERING' if a.phase=='engineering' else 'RETRIEVAL'
    require(time.time()<=LATEST_START_UNIX,'LATEST_SCIENTIFIC_START_UTC')
    # This O_EXCL intent is written before any input validation: every actual CLI call is burned.
    attempt=put(W/(prefix+'_OUTER_ATTEMPT.actual.v1.json'),{'schema_version':1,'epoch':57,'kind':KIND,
        'absolute_monotonic_start':START,'absolute_monotonic_deadline':LIMIT,'automatic_retry':False,
        'supervisor_identity':snap(os.getpid()),'supplied_plan_sha256':a.expected_plan_sha256,
        'supplied_admission_sha256':a.expected_admission_sha256,'supplied_approval_sha256':a.expected_approval_sha256})
    outcome={'schema_version':1,'epoch':57,'kind':KIND,'phase':a.phase,'outer_attempt_ref':attempt,
             'actual_worker_starts':0,'automatic_retry':False,'errors':[],'owned_process_identities':[]}
    child=None;observed={};groups=set();known=set();reason=None;term_at=None;killed=False;rc=None
    stop_requested=[False]
    def stop_for_external_signal(signum,frame):
        # Deferred flag avoids asynchronous exceptions during Popen/identity capture.
        stop_requested[0]=True
    signal.signal(signal.SIGTERM,stop_for_external_signal)
    signal.signal(signal.SIGINT,stop_for_external_signal)
    try:
        require(sorted(os.sched_getaffinity(0))==[0,1],'CPU_AFFINITY_0_1')
        for name in ('plan','admission','approval'):
            require(Path(getattr(a,name)).is_relative_to(W),'ROOT_ADMITTED_CONTROL_PATH')
        planref=ref(a.plan);adref=ref(a.admission);apref=ref(a.approval)
        require(planref['sha256']==a.expected_plan_sha256 and adref['sha256']==a.expected_admission_sha256
                and apref['sha256']==a.expected_approval_sha256,'FROZEN_CONTROL_HASHES')
        plan=json.loads(Path(a.plan).read_bytes());ad=json.loads(Path(a.admission).read_bytes());approval=json.loads(Path(a.approval).read_bytes())
        require(plan['epoch']==ad['epoch']==approval['epoch']==57 and plan['kind']==ad['kind']==approval['kind']==KIND,'CONTROL_KIND')
        require(plan['work_root']==str(W) and plan['maximum_native_batches']==1 and plan['maximum_outer_seconds']==600
                and plan['typed_model_initializations_authorized']==plan['NLI_forwards_authorized']==0
                and plan['launch_environment']==ENV and plan['phase']==a.phase and plan['BGE_batch_size']==32 and plan['BGE_token_cap']==256,'FROZEN_PHASE_RESOURCE_SCOPE')
        require(ad['allowed'] is True and approval['checks_passed'] is True and approval['dispatch_allowed'] is True
                and ad['plan_sha256']==approval['plan_sha256']==planref['sha256']
                and approval['admission_sha256']==adref['sha256'],'INDEPENDENT_EXACT_DISPATCH_ADMISSION')
        require(ad['source_publication_independently_verified'] is True and ad['source_full_tree_verified'] is True
                and ad['source_commit'] not in DENY and len(ad['source_commit'])==40,'ACTUAL_SOURCE_PUBLICATION_REQUIRED')
        source=parse(ad['source_publication_proof_ref']);require(source['source_commit']==ad['source_commit']
                and source['source_publication_independently_verified'] is True
                and source['source_full_tree_verified'] is True,'INDEPENDENT_ACTUAL_SOURCE_PROOF')
        sf=parse(ad['source_freeze_ref']);require(sf['epoch']==57 and sf['kind']==ENGINEERING_KIND
                and ad['source_freeze_sha256']==ad['source_freeze_ref']['sha256'],'SOURCE_FREEZE_IDENTITY')
        project=Path(plan['project_root']);require(project.is_relative_to(W) and project.is_dir() and not project.is_symlink(),'ISOLATED_PROJECT')
        expected=[]
        for row in sf['files']:
            rel=Path(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'SOURCE_RELATIVE_PATH')
            expected.append({'path':str(project/rel),'bytes':row['bytes'],'sha256':row['sha256']})
        require(ad['source_files']==expected,'ALL_FROZEN_SOURCE_FILES_ADMITTED')
        for row in expected:exact(row)
        source_names={Path(row['path']).name for row in expected}
        require({'engineering_worker.py','pilot_worker.py','runtime_common.py','supervisor.py','native_helper.cjs','native_bridge.py'}<=source_names,
                'EXACT_RETRIEVAL_SOURCE_COMPONENTS')
        require(any(row['path']==str(Path(a.plan)) and row['sha256']==planref['sha256'] for row in expected),'PHASE_PLAN_IS_SOURCE_FROZEN')
        require(all(sum(row['path']==str(project/rel) for row in expected)==1 for rel in ('experiments/sota-2026-10-22/epoch57/plan.json','experiments/sota-2026-10-22/epoch57/retrieval/plan.json')),'BOTH_PHASE_PLANS_FROZEN_BEFORE_FIRST_SCI')
        if a.phase=='retrieval_development':
            first=parse(ad['prerequisite_engineering_acceptance_ref'])
            require(first['epoch']==57 and first['independent_review_complete'] is True and first['engineering_qualified'] is True and first['throughput_gate_passed'] is True and first['actual_execution_independently_derived'] is True and first['all_owned_processes_closed'] is True and first['native_counter_accounting']['qualified'] is True,'PRIOR_ENGINEERING_THROUGHPUT_NATIVE_AND_CLOSURE_QUALIFIED')
            rr=first['actual_results_ref'];require(type(rr) is dict and set(rr)=={'path','bytes','sha256'} and not Path(rr['path']).is_absolute() and '..' not in Path(rr['path']).parts,'ACCEPTED_ENGINEERING_ACTUAL_RESULT_REF')
            er=parse({'path':str(project/rr['path']),'bytes':rr['bytes'],'sha256':rr['sha256']})
            engineering_plan=next(row for row in expected if row['path']==str(project/'experiments/sota-2026-10-22/epoch57/plan.json'))
            require(er['epoch']==57 and er['kind']==ENGINEERING_KIND and er['phase']=='engineering' and er['plan_sha256']==engineering_plan['sha256'] and er['source_commit']==ad['source_commit'] and er['errors']==[] and er['status']=='QUANTIZED_ENGINEERING_FORECASTS_SEALED_NOT_SCORED' and er['engineering_gate']['local_gate_passed'] is True and er['native_state']['accounting']['qualified'] is True and er['semantic_gold_decodes']==0,'ACTUAL_SAME_FROZEN_ENGINEERING_SOURCE_GATE')
        asset_names=['bge_model','bge_tokenizer','native_binary','node','native_helper_source','native_bridge_source']
        asset_names+=['bge_fp32_model','engineering_features'] if a.phase=='engineering' else ['corpus_features','query_features']
        for name in asset_names:exact(plan[name])
        documents=128 if a.phase=='engineering' else 5183
        require(plan['corpus_document_count']==documents and plan['query_count']==16 and plan['top_k']==10,'FROZEN_PHASE_GEOMETRY')
        require(len(plan['selected_query_ids'])==len(set(plan['selected_query_ids']))==16,'FROZEN_QUERY_IDENTITIES')
        require(plan['BGE_initialization_cap']==(2 if a.phase=='engineering' else 1) and plan['BGE_forward_cap']==(9 if a.phase=='engineering' else 163) and plan['BGE_texts_cap']==(272 if a.phase=='engineering' else 5199),'FROZEN_PHASE_EMBEDDING_COUNTS')
        require(ad['owner_ref']==plan['owner_ref'],'OWNER_BINDING');owner=fence(ad['owner_ref'])
        require(time.time()+600<owner['expires_unix'] and time.monotonic()<START+30 and time.time()<=LATEST_START_UNIX,'PREHASH_AND_OWNER_DEADLINE_MARGIN')
        out=W/'private-run'/a.phase;require(not out.exists(),'ONE_RETRIEVAL_RUN_DIRECTORY')
        out.mkdir(parents=True,mode=0o700);worker=exact(plan['worker_source'])
        env=os.environ.copy();env.update(ENV)
        command=[plan['python_executable'],'-I','-B',str(worker),'--plan',a.plan,'--expected-plan-sha256',planref['sha256'],
                 '--admission',a.admission,'--expected-admission-sha256',adref['sha256'],'--out-dir',str(out),'--deadline',str(START+570)]
        outcome.update(plan_ref=planref,admission_ref=adref,independent_approval_ref=apref,source_commit=ad['source_commit'],
                       source_freeze_ref=ad['source_freeze_ref'],owner_ref=ad['owner_ref'],nonsecret_environment_overrides=ENV)
        require(not stop_requested[0],'EXTERNAL_SUPERVISOR_SIGNAL')
        with (out/'worker.stdout.txt').open('xb') as stdout,(out/'worker.stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True,close_fds=True)
        outcome['actual_worker_starts']=1;first=snap(child.pid);require(first and first['pgid']==first['sid']==child.pid,'OWNED_NEW_WORKER_SESSION')
        observed[(first['pid'],first['startticks'])]=first;groups={child.pid};known={child.pid}
        put(W/(prefix+'_WORKER_STARTED.actual.v1.json'),{'schema_version':1,'epoch':57,'kind':KIND,'identity':first,
             'outer_start_monotonic':START,'worker_soft_deadline':START+570,'command':command,'outer_attempt_ref':attempt})

        def observe():
            allp=[snap(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()];allp=[x for x in allp if x]
            changed=True
            while changed:
                changed=False
                for s in allp:
                    if s['pid'] in known or s['ppid'] in known or s['pgid'] in groups or s['sid'] in groups:
                        observed[(s['pid'],s['startticks'])]=s
                        if s['pid'] not in known:known.add(s['pid']);groups.update((s['pgid'],s['sid']));changed=True
            return [s for s in allp if (s['pid'],s['startticks']) in observed and s['state'] not in ('Z','X')]
        def signal_live(sig,live):
            for group in sorted({s['pgid'] for s in live}):
                require(group in groups,'ONLY_OWNED_GROUP_SIGNAL')
                try:os.killpg(group,sig)
                except ProcessLookupError:pass
        while True:
            live=observe();rc=child.poll();elapsed=time.monotonic()-START
            if stop_requested[0]:reason=reason or 'EXTERNAL_SUPERVISOR_SIGNAL'
            if rc is not None and not live:break
            try:fence(ad['owner_ref'])
            except Exception:reason=reason or 'OWNER_LOST'
            if rc is not None and live:reason=reason or 'STRAGGLER_AFTER_WORKER_EXIT'
            if elapsed>=580:reason=reason or 'ABSOLUTE_580_SECOND_BACKSTOP'
            if time.time()>=LATEST_FINISH_UNIX:reason=reason or 'SCIENTIFIC_FINISH_UTC_BACKSTOP'
            if reason and term_at is None:signal_live(signal.SIGTERM,live);term_at=time.monotonic()
            if term_at is not None and not killed and (time.monotonic()-term_at>=5 or elapsed>=585):
                signal_live(signal.SIGKILL,live);killed=True
            if elapsed>=595:reason=reason or 'PHYSICAL_CLEANUP_DEADLINE';break
            time.sleep(.1)
        rc=child.poll();live=observe()
        require(rc is not None and not live,'ALL_OWNED_WORKER_NATIVE_PROCESSES_CLOSED')
        if rc==0 and reason is None:
            rawref=ref(out/'raw.json');raw=json.loads((out/'raw.json').read_bytes())
            require(raw['status']==('QUANTIZED_ENGINEERING_FORECASTS_SEALED_NOT_SCORED' if a.phase=='engineering' else 'RETRIEVAL_DEVELOPMENT_FORECASTS_SEALED_NOT_SCORED') and raw['errors']==[]
                    and raw['plan_sha256']==planref['sha256'] and raw['phase']==a.phase and raw['counters']['typed_model_initializations']==raw['counters']['NLI_forwards_completed']==raw['semantic_gold_decodes']==0
                    and raw['metrics'] is None and raw['forecast_seal']['sha256']==ref(out/'forecasts.json')['sha256'] and raw['native_state']['accounting']['qualified'] is True,'COMPLETE_RETRIEVAL_RAW_RECEIPT')
            outcome['raw_ref']=rawref
    except BaseException as e:
        code=str(e);outcome['errors'].append({'type':type(e).__name__,'code':code if code.replace('_','').isalnum() and len(code)<=120 else 'SUPERVISOR_FAILURE'})
        reason=reason or 'SUPERVISOR_EXCEPTION'
    finally:
        # Preserve the bounded owned cleanup if a caller repeats TERM/INT.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            s=snap(child.pid)
            if s and (s['pid'],s['startticks']) in observed:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=max(.1,min(2,LIMIT-time.monotonic()-3)))
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=max(.1,min(2,LIMIT-time.monotonic()-1)))
                    except subprocess.TimeoutExpired:pass
        finals=[];live=[]
        for identity in observed.values():
            current=snap(identity['pid']);finals.append({'identity':identity,'current':current})
            if current and current['startticks']==identity['startticks'] and current['state'] not in ('Z','X'):live.append(current)
        # Scan the whole known groups again to catch a child that outlived its parent.
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                s=snap(int(p.name))
                if s and (s['pgid'] in groups or s['sid'] in groups) and s['state'] not in ('Z','X'):
                    if s not in live:live.append(s)
        if live:
            for group in sorted({s['pgid'] for s in live}):
                if group in groups:
                    try:os.killpg(group,signal.SIGKILL)
                    except ProcessLookupError:pass
            for _ in range(20):
                if time.monotonic()>=LIMIT-1:break
                live=[s for old in live if (s:=snap(old['pid'])) and s['startticks']==old['startticks'] and s['state'] not in ('Z','X')]
                if not live:break
                time.sleep(.05)
        rc=child.poll() if child is not None else None
        partial_dir=W/'private-run'/a.phase
        for name,key in [('raw.json','raw_ref'),('worker-progress.jsonl','worker_progress_ref')]:
            path=partial_dir/name
            if path.is_file():
                try:outcome[key]=ref(path)
                except Exception:outcome[key+'_capture_failed']=True
        wall=time.monotonic()-START
        outcome.update(supervisor_identity=snap(os.getpid()),worker_exit_code=rc,forced_reason=reason,killed=killed,
             outer_wall_seconds_including_preflight_hash_init_native_capture_cleanup=wall,within_600_seconds=wall<600,
             owned_process_identities=list(observed.values()),final_process_snapshots=finals,live_owned_processes=live,
             all_owned_experiment_processes_physically_closed=not live and (child is None or rc is not None),
             typed_model_initializations=0,NLI_forwards=0,Gold_EVAL_calls=0,semantic_quality_metrics=None)
        zero_terminal=all(x['current'] is None or (x['current']['startticks']==x['identity']['startticks'] and x['current']['state']=='Z' and x['current']['wait_status']==0 and x['current']['FDSize']==0 and not any(x['current']['statm'])) for x in finals)
        outcome['all_recorded_terminal_process_resources_zero']=zero_terminal
        outcome['checks_passed']=rc==0 and reason is None and not outcome['errors'] and not live and zero_terminal and wall<600
        result=put(W/(prefix+'_OUTER_EXECUTION.actual.v1.json'),outcome);print(json.dumps({'execution':result,'checks_passed':outcome['checks_passed'],'worker_exit_code':rc,'outer_wall_seconds':wall,'live_owned':len(live)}),flush=True)
    return 0 if outcome['checks_passed'] else 17

if __name__=='__main__':raise SystemExit(main())
