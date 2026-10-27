"""One bounded, admitted source-asset GET; never imports a model runtime."""
import sys
sys.dont_write_bytecode = True
import argparse, fcntl, hashlib, json, os, signal, stat, time, urllib.request
from pathlib import Path

START = time.monotonic()
def alarm(_signum, _frame): raise TimeoutError('ASSET_WHOLE_WORKER_100_SECONDS')
signal.signal(signal.SIGALRM, alarm)
signal.alarm(100)

def need(ok, why):
    if not ok: raise ValueError(why)

def body(path):
    p = Path(path)
    need(p.is_absolute() and not any(x.is_symlink() for x in (p, *p.parents)), 'ABSOLUTE_NO_SYMLINK')
    a = p.stat(); need(stat.S_ISREG(a.st_mode), 'REGULAR_SOURCE')
    b = p.read_bytes(); z = p.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns), 'SOURCE_STABLE')
    return b

def ref(path):
    b = body(path); return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def checked(r):
    need(set(r)=={'path','bytes','sha256'}, 'TYPED_REF')
    b=body(r['path']); need(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'], 'SOURCE_REF_HASH')
    return b

def put(path, value):
    p=Path(path); need(p.is_absolute() and not any(x.is_symlink() for x in (p,*p.parents)), 'OUTPUT_NO_SYMLINK')
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)
    return ref(p)

def headroom():
    maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    need(maximum.isdecimal(), 'FINITE_MEMORY_CGROUP')
    current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    return {'maximum':int(maximum),'current':current,'available':int(maximum)-current}

def owner_fence(r, source):
    import importlib.util
    checked(source)
    spec=importlib.util.spec_from_file_location('e58_asset_guardian26',source['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify_owner(r)

def main():
    need(not sys.flags.optimize and os.sched_getaffinity(0)=={0,1}, 'CPU2_NONOPTIMIZED')
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); ap.add_argument('--expected-plan-sha256',required=True)
    ap.add_argument('--admission',required=True); ap.add_argument('--expected-admission-sha256',required=True)
    ap.add_argument('--asset',choices=['nli_model','nli_tokenizer'],required=True); args=ap.parse_args()
    pb=body(args.plan); need(hashlib.sha256(pb).hexdigest()==args.expected_plan_sha256, 'EXACT_ASSET_PLAN')
    plan=json.loads(pb); need(plan['epoch']==58 and plan['mode']=='SOURCE_ONLY_ASSET_GET_NO_SCI', 'ASSET_PLAN_KIND')
    need(ref(Path(__file__).resolve())==plan['downloader_source'], 'REVIEWED_DOWNLOADER_SOURCE')
    ab=body(args.admission); need(hashlib.sha256(ab).hexdigest()==args.expected_admission_sha256, 'EXACT_ASSET_ADMISSION')
    admission=json.loads(ab)
    need(admission['epoch']==58 and admission['checks_passed'] is True and admission['download_allowed'] is True and admission['scientific_authorization_granted'] is False, 'SOURCE_GET_ONLY_APPROVAL')
    need(admission['plan_ref']==ref(args.plan) and admission['downloader_source']==plan['downloader_source'] and admission['owner']==plan['owner'] and admission['fence_source']==plan['fence_source'], 'SOURCE_GET_APPROVAL_BINDING')
    owner_fence(plan['owner'],plan['fence_source']); before=headroom(); need(before['available']>=plan['pre_GET_headroom_bytes'], 'SOURCE_GET_HEADROOM')
    row=plan['assets'][args.asset]; api=json.loads(checked(plan['model_API_source']))
    need(api['id']=='cross-encoder/nli-deberta-v3-small' and api['sha']==plan['model_revision'], 'OFFICIAL_PINNED_REPOSITORY')
    source=next(x for x in api['siblings'] if x['rfilename']==row['repository_path'])
    need(source['size']==row['bytes'] and source['blobId']==row['Hub_blob_or_pointer_GitSHA'], 'PINNED_ASSET_METADATA')
    if args.asset=='nli_model': need(source['lfs']['sha256']==row['sha256'], 'LFS_WEIGHT_HASH')
    else: need('lfs' not in source and row['body_GitSHA']==source['blobId'], 'REGULAR_TOKENIZER_GIT_HASH')
    expected_url='https://huggingface.co/cross-encoder/nli-deberta-v3-small/resolve/'+plan['model_revision']+'/'+row['repository_path']
    need(row['url']==expected_url, 'ONE_PINNED_PUBLIC_URL')
    folder=Path(plan['output_directory']); folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    need(not any(x.is_symlink() for x in (folder,*folder.parents)), 'ASSET_DIRECTORY_NO_SYMLINK')
    target=folder/row['filename']; partial=folder/(row['filename']+'.partial')
    attempt=folder/(args.asset+'.GET_ATTEMPT.json'); result=folder/(args.asset+'.GET_RESULT.actual.json')
    need(not any(os.path.lexists(x) for x in (target,partial,attempt,result)), 'SOURCE_GET_ONCE_NO_RETRY')
    put(attempt,{'epoch':58,'asset':args.asset,'plan_ref':ref(args.plan),'owner':plan['owner'],'source_only':True,'automatic_retry':False,'started_monotonic':START})
    record={'schema_version':1,'epoch':58,'asset':args.asset,'status':'FAILED_NO_RETRY','requested_pinned_public_url':expected_url,'HTTP_GET_attempts':0,'HTTP_GET_completed':0,'bytes_read':0,'model_tokenizer_native_initializations':0,'paid_API_or_GPU_calls':0,'before_memory':before,'errors':[]}
    out=None
    try:
        out=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        h=hashlib.sha256(); gh=hashlib.sha1(b'blob '+str(row['bytes']).encode()+b'\0')
        request=urllib.request.Request(expected_url,headers={'User-Agent':'RuVector-E58-public-source-asset/1'})
        record['HTTP_GET_attempts']=1
        with urllib.request.urlopen(request,timeout=25) as response:
            need(response.status==200,'HTTP200_REQUIRED')
            while True:
                chunk=response.read(min(1048576,row['bytes']-record['bytes_read']+1))
                if not chunk: break
                record['bytes_read']+=len(chunk); need(record['bytes_read']<=row['bytes'],'ASSET_SIZE_CAP')
                h.update(chunk); gh.update(chunk); offset=0
                while offset<len(chunk):
                    n=os.write(out,chunk[offset:]); need(n>0,'ASSET_WRITE_PROGRESS'); offset+=n
        need(record['bytes_read']==row['bytes'],'EXACT_ASSET_SIZE')
        actual_sha=h.hexdigest(); actual_git=gh.hexdigest()
        if row.get('sha256') is not None: need(actual_sha==row['sha256'],'EXACT_ASSET_SHA256')
        if row.get('body_GitSHA') is not None: need(actual_git==row['body_GitSHA'],'EXACT_ASSET_BODY_GITSHA')
        os.fsync(out); os.close(out); out=None
        owner_fence(plan['owner'],plan['fence_source']); after=headroom(); need(after['available']>=plan['post_GET_headroom_bytes'],'POST_GET_HEADROOM')
        os.link(partial,target,follow_symlinks=False); partial.unlink()
        directory=os.open(folder,os.O_RDONLY|os.O_DIRECTORY); os.fsync(directory); os.close(directory)
        record.update(status='COMPLETE_VERIFIED_SOURCE_ASSET_ONLY',HTTP_GET_completed=1,asset_ref={'path':str(target),'bytes':row['bytes'],'sha256':actual_sha},body_GitSHA=actual_git,after_memory=after)
    except Exception as error:
        record['errors'].append({'type':type(error).__name__,'message_sha256':hashlib.sha256(str(error).encode()).hexdigest()})
        if out is not None: os.close(out); out=None
        if partial.exists(): partial.unlink()
    finally:
        record['whole_worker_elapsed_seconds']=time.monotonic()-START
        record['source_GET_inner_SIGALRM_seconds']=100
        record['required_outer_timeout_seconds']=108
        record['redirect_urls_headers_and_credentials_not_persisted']=True
        result_ref=put(result,record)
    print(json.dumps({'result_ref':result_ref,'status':record['status']},sort_keys=True))
    return 0 if record['status']=='COMPLETE_VERIFIED_SOURCE_ASSET_ONLY' else 2

if __name__=='__main__': raise SystemExit(main())
