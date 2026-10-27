"""E59 exact source restoration; no model, native, label or Git entrypoint.

The unchanged bootstrap/checkpoint/transport extractors remain hash-pinned.
The known legacy22 metadata bound is explicitly raised only in memory to4MiB.
Kit retirement unlinks only newly copied, independently hash-equal R twins.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time

D = Path('/dev/shm/ruvector-e59-data')
W = Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983')
K, R, B = W/'kit', W/'project', W/'base-cache'
HEAD = '9b677a91698352bd563f189fee861d6ff2a1ad3a'
DENY = {'486888de9602bbb5d817a55f707b4fcfd11c65e0','3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'}
BOOT = 'eceb2ca259feaf642bc81172aca57a2a209cc34cd17f9b281861c51f21b95806'
RESTORE = 'affd9ecf660f380c3632f78cd14fb7a36e00797ceb097cf48c0c60136fac0be8'
E22 = '7691256e21cb201e9b64c64b2f880fd6d0274b562cfb4acfa03c5ce673a4ba8f'
def need(ok, message):
    if not ok: raise ValueError(message)
def sha(p, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git: h.update(b'blob '+str(p.stat().st_size).encode()+b'\0')
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()
def regular(p):
    need(p.is_file() and not any(x.is_symlink() for x in [p,*p.parents]), 'regular source '+str(p))
def ref(p):
    regular(p); return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(row):
    need(type(row) is dict and set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int,'typed3 reference')
    p=Path(row['path']); need(ref(p)==row,'source identity '+str(p)); return p
def relative(name):
    p=Path(name); need(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'relative source path');return p
def put(p, value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(p)
def copy(src,dst):
    regular(src);need(not dst.exists() and not dst.is_symlink(),'new destination '+str(dst));dst.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with src.open('rb') as a,os.fdopen(fd,'wb') as b:shutil.copyfileobj(a,b,1048576);b.flush();os.fsync(b.fileno())
    need((src.stat().st_dev,src.stat().st_ino)!=(dst.stat().st_dev,dst.stat().st_ino),'distinct physical copy')
def activation(args):
    need(not sys.flags.optimize and os.sched_getaffinity(0)=={0,1},'unoptimized CPU0,1 source-only invocation')
    p=Path(args.config);need(sha(p)==args.config_sha,'configuration SHA');c=json.loads(p.read_bytes())
    a=Path(args.approval);need(sha(a)==args.approval_sha,'approval SHA');q=json.loads(a.read_bytes())
    need(c['epoch']==59 and c['source_checkpoint']==HEAD and c['kind']=='SOURCE_ONLY_HASH_VERIFIED_RECONSTRUCTION','exact source-only kind')
    need(q['checks_passed'] is True and q['source_reconstruction_allowed'] is True and q['configuration_ref']==ref(p) and q['restoration_code_ref']==c['restoration_code'],'independent exact restoration admission')
    need(checked(c['restoration_code'])==Path(__file__).resolve(),'exact operational restoration source')
    fp=checked(c['fence_source']);spec=importlib.util.spec_from_file_location('e59_source_owner_fence',fp);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    need(c['owner']['path']==str(W/'guardian27/owner.json'),'original generation27 owner');f.verify_owner(c['owner'])
    return c,lambda:f.verify_owner(c['owner'])
def legacy22(c,fence):
    fence();p=K/'experiments/sota-2026-10-22/epoch22/publication/restore_archived_artifact_v1.py';need(p.stat().st_size==11810 and sha(p)==E22,'unchanged original legacy22 extractor')
    spec=importlib.util.spec_from_file_location('e59_unchanged_legacy22',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(Path(m.__file__).absolute()==p and m.METADATA_LIMIT==2097152,'unaltered extractor and original runtime bound');m.METADATA_LIMIT=4194304
    result=m.restore(str(K),'experiments/sota-2026-10-22/epoch22/publication/actualarchive.dev.raw.v1.json','8ab3d8ec7856b423e6ff295f8e5ef74fc81dcaa3d3382fa2757e21563e0a99c6',verify_only=True)
    fence();print(json.dumps({'original_extractor_unchanged':True,'metadata_bound_before':2097152,'metadata_bound_effective':4194304,'verify_only':True,'result':result}));return
def run(args,c,fence):
    need(not K.exists() and not R.exists() and not B.exists(),'fresh isolated targets')
    M=json.loads(checked(c['actual_map']).read_bytes());C=json.loads(checked(c['current_CP']).read_bytes());donors=json.loads(checked(c['donor_map']).read_bytes())
    need(M['commit']==HEAD and M['root']=='6da59a0283d3696e682f8d6b96b3175c6830a00b' and len(M['files'])==9232 and len(C['project_files'])==9225,'actual checkpoint size/root')
    need(donors['actual_commit']==HEAD and len(donors['files'])==9302 and len({r['path']for r in donors['files']})==9302,'all current Git and70CP-only sources')
    need(int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())>=2*sum(x['bytes']for x in donors['files'])+1073741824,'source-copy peak and oneGiB reserve')
    index={x['path']:x for x in M['files']};known={x['path']:x for x in donors['files']};need(len(set(x['path']for x in C['project_files'])-set(index))==70,'CP-only transport count')
    for i,x in enumerate(donors['files']):
        if i%128==0:fence()
        name=relative(x['path']);src=Path(x['physical_source']);need(src.stat().st_size==x['bytes'] and sha(src)==x['sha256'],'fresh donor hash '+str(name))
        if x['GitSHA'] is not None:need(x['GitSHA'] not in DENY and sha(src,True)==index[str(name)]['sha']==x['GitSHA'],'fresh Git source '+str(name))
        copy(src,K/name);need(sha(K/name)==x['sha256'],'new Kit body hash')
    need(sha(K/'bootstrap.py')==BOOT and sha(K/'restore_checkpoint.py')==RESTORE,'reviewed original bootstrap and restore')
    need(sha(K/'BASE_PUBLIC_MANIFEST.json')=='0db67a018ad5fd9deba1cfb0bd8897537a330dd2353f169e6e387e86e09fce7a','original base manifest')
    base={x['path']:x for x in json.loads((K/'BASE_PUBLIC_MANIFEST.json').read_bytes())['files']};names=json.loads((K/'BASE_DOWNLOADS.json').read_bytes())['paths'];need(len(names)==48,'all48 original base downloads')
    for name in names:
        x=base[name];src=Path(c['base_cache_donor'])/relative(name);need(src.stat().st_size==x['bytes'] and sha(src)==x['sha256'],'exact base-cache source');copy(src,B/relative(name))
    py=[sys.executable,'-I','-B'];e42=K/'recoveries/epoch42-publication-20261026';e44=K/'recoveries/epoch44-publication-20261026-g9';e49=R/'recoveries/epoch49-stop-publication-20261027-g15'
    common=['--config',args.config,'--config-sha',args.config_sha,'--approval',args.approval,'--approval-sha',args.approval_sha]
    steps=[('E42-archive-materialized.json',py+[str(e42/'restore_epoch42_archive.py'),'--kit-root',str(K),'--dest',str(R),'--restore-scope',str(W),'--archive','recoveries/epoch42-publication-20261026/immutable-e42-artifacts.zip','--projection','recoveries/epoch42-publication-20261026/ARCHIVE_PROJECTED_FILES.v1.json']),
      ('legacy22-verified-cap4m.json',py+[str(Path(__file__).resolve()),*common,'--legacy22-verify-only']),
      ('legacy26-verified.json',py+[str(K/'experiments/sota-2026-10-22/epoch26/publication/restore_epoch26_archives_v1.py'),'--kit-root',str(K),'--manifest','experiments/sota-2026-10-22/epoch26/publication/archive-transport.v1.json','--expected-manifest-sha256','9b79fc0119e1e5d15470201b7180f4b39e55b687a207dac1ff9f333d91a8ef24','--verify-only']),
      ('bootstrap-source-verified.json',py+[str(K/'bootstrap.py'),'--dest',str(R),'--base-local-directory',str(B)]),
      ('checkpoint-restored.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B)]),
      ('checkpoint-verify-only.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B),'--verify-only']),
      ('E42-archive-verify-only.json',py+[str(e42/'restore_epoch42_archive.py'),'--kit-root',str(K),'--dest',str(R),'--restore-scope',str(W),'--archive','recoveries/epoch42-publication-20261026/immutable-e42-artifacts.zip','--projection','recoveries/epoch42-publication-20261026/ARCHIVE_PROJECTED_FILES.v1.json','--verify-only'])]
    cmd44=py+[str(e44/'restore_epoch44_overlay.py'),'--restore-scope',str(W),'--kit',str(K),'--dest',str(R),'--envelope',str(e44/'PUBLIC_ARTIFACTS.envelope.v1.json'),'--expected-envelope-sha','fc0f741eb7bd840a941e9ef3505b6924ab9be7842cd561722d4236bd98cf1d1d','--overlay',str(e44/'ARCHIVE_OVERLAY.v1.json'),'--expected-overlay-sha','815aaa11a602de738b1b62b4f9cd057639b8201b59007b3633588aaa674ba01d']
    cmd49=py+[str(e49/'restore_e49_overlay.py'),'--envelope',str(e49/'PUBLIC_RECOVERY.envelope.v1.json'),'--manifest',str(e49/'ARCHIVE_OVERLAY.v1.json'),'--project',str(R)]
    steps += [('E44-overlay-restored.json',cmd44),('E44-overlay-verified.json',cmd44+['--verify-only']),('E49-overlay-restored.json',cmd49),('E49-overlay-verified.json',cmd49+['--verify-only'])]
    receipts=[]
    for name,argv in steps:
        fence();script=Path(argv[3]);need(not any(x in argv for x in ['--install-npm','--install-python','--acquire-models','--verify-models']),'no acquisition/installation or inference flags')
        if script==Path(__file__).resolve():need(ref(script)==c['restoration_code'],'exact new metadata-bound adapter')
        else:
            key=str(script.relative_to(K if script.is_relative_to(K) else R));need(sha(script,True)==index[key]['sha'],'exact unchanged transport extractor')
        t=time.monotonic();result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        receipts.append(put(W/name,{'argv':argv,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-t,'scientific_entrypoints_dispatched':[]}));need(result.returncode==0,'source-only step failed '+name);print(name+' PASS',flush=True)
    for root in [K,R]:
        for i,x in enumerate(C['project_files']):
            if i%128==0:fence()
            p=root/relative(x['path']);need(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'all CP bodies '+str(p))
    for x in M['files']:need((K/x['path']).stat().st_size==x['size'] and sha(K/x['path'],True)==x['sha'],'all fresh Git Kit bodies')
    for x in donors['files']:
        p=Path(x['physical_source']);need(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'original donor preserved')
    fence();before={p.name:sum(x.is_file()for x in p.rglob('*'))for p in [K,R,B]}
    # Source-only cache cleanup: only verified new Kit/R twins, no donor mutation.
    retired=[]
    for i,p in enumerate(sorted(x for x in K.rglob('*') if x.is_file())):
        if i%128==0:fence()
        q=R/p.relative_to(K)
        if q.is_file() and not q.is_symlink() and p.stat().st_size==q.stat().st_size and sha(p)==sha(q):
            need((p.stat().st_dev,p.stat().st_ino)!=(q.stat().st_dev,q.stat().st_ino),'distinct verified cache twins');retired.append({'path':str(p.relative_to(K)),'bytes':p.stat().st_size,'sha256':sha(p)});p.unlink()
    rr=put(W/'KIT_TWIN_RETIREMENT.actual.v1.json',{'epoch':59,'only_new_Kit_twins_unlinked':True,'original_donor_mutations':0,'retired_files':len(retired),'retired_bytes':sum(x['bytes']for x in retired),'retired_entries':retired})
    for i,x in enumerate(C['project_files']):
        if i%128==0:fence()
        p=R/x['path'];need(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'R preserved after Kit retirement')
    fence();summary={'schema_version':1,'epoch':59,'kind':c['kind'],'checks_passed':True,'actual_commit':HEAD,'all_CP_R_body_hashes_verified':len(C['project_files']),'all_fresh_Git_Kit_hashes_verified_before_retirement':len(M['files']),'all70_CP_only_sources_restored':True,'base_files':48,'unchanged_bootstrap_and_restore_executed':True,'original_helpers_unmodified':True,'only_legacy22_runtime_bound_override':[2097152,4194304],'source_only_steps':receipts,'counts_before_Kit_retirement':before,'counts_after_Kit_retirement':{p.name:sum(x.is_file()for x in p.rglob('*'))for p in [K,R,B]},'Kit_twin_retirement':rr,'original_donors_preserved':True,'scientific_entrypoints_dispatched':[],'model_acquisition_installation_paid_or_Git_mutations':0,'remaining_headroom_bytes':int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())}
    print(json.dumps(put(W/'SOURCE_RESTORATION_COMPLETE.v1.json',summary)),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--approval',required=True);p.add_argument('--approval-sha',required=True);p.add_argument('--legacy22-verify-only',action='store_true');a=p.parse_args();c,f=activation(a)
    if a.legacy22_verify_only:legacy22(c,f)
    else:run(a,c,f)
if __name__=='__main__':main()
