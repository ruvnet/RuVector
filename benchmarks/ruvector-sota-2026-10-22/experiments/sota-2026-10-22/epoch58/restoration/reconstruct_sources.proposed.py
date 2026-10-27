"""E58 source-only reconstruction proposal. No model/native/data scoring entrypoint."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
D=Path('/dev/shm/ruvector-e58-data')
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1')
OLD=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
K=W/'kit';R=W/'project';B=W/'base-cache'
DENY={'486888de9602bbb5d817a55f707b4fcfd11c65e0','3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'}
def sha(p,git=False):
 h=hashlib.sha1()if git else hashlib.sha256()
 if git:h.update(b'blob '+str(p.stat().st_size).encode()+b'\0')
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fence():
 p=W/'owner.json';assert p.stat().st_size==710 and sha(p)=='3eb3df852ccca79425f8c129d4e9efaabb1a8cbbce33706da9b8dc08873d3fa3'
 o=json.loads(p.read_text());assert o['generation']==25 and o['pid']==191872 and o['startticks']==45020273 and time.time()<o['expires_unix']
 assert not(W/'release.request').exists()and not(W/'released.json').exists()
 assert o['kernel_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and o['PID_namespace']==os.readlink('/proc/self/ns/pid')
 s=Path('/proc/191872/stat').read_text().rsplit(')',1)[1].split();assert int(s[19])==45020273 and s[0]not in('Z','X')and os.sched_getaffinity(191872)=={0,1}
 a=Path('/proc/191872/fd/3').stat();l=Path('/tmp/ruvector-sota-2026-10-22/exclusive.work.lock');z=l.stat();assert(a.st_dev,a.st_ino)==(z.st_dev,z.st_ino)==(33,13626)
 assert any('FLOCK'in s and 'WRITE'in s and s.split()[4]=='191872'and s.split()[5]=='00:21:13626'for s in Path('/proc/locks').read_text().splitlines())
 fd=os.open(l,os.O_RDWR|os.O_NOFOLLOW)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:fcntl.flock(fd,fcntl.LOCK_UN);raise ValueError('original lock free')
 finally:os.close(fd)
def copy(src,dst):
 assert src.is_file()and not any(p.is_symlink()for p in[src,*src.parents]);assert not dst.exists()and not dst.is_symlink()
 dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb')as a,dst.open('xb')as b:shutil.copyfileobj(a,b,1048576);b.flush();os.fsync(b.fileno())
 assert(src.stat().st_dev,src.stat().st_ino)!=(dst.stat().st_dev,dst.stat().st_ino)
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 assert not sys.flags.optimize and os.sched_getaffinity(0)=={0,1};fence()
 assert sha(D/'actual-git-map.json')=='554e42b12b10c0142f19f526bd5846fe14619ac2de5fce14297116e3722d0636'
 M=json.loads((D/'actual-git-map.json').read_text());C=json.loads((D/'current/CHECKPOINT_MANIFEST.json').read_text())
 assert M['commit']=='db7b9b21f4a13e5fc3444a8fe40de26583558e2e'and M['scope_sha']=='c65ae43bd4ed4f19d175cfafce04b10117201b0a'and len(M['files'])==9055
 assert sha(D/'current/CHECKPOINT_MANIFEST.json')=='ae9b6df276bbd2b3446fd21b4a3a6b2a8e3e2b96083716a3f1478318c4144d8f'and len(C['project_files'])==9048
 assert not K.exists()and not R.exists()and not B.exists()
 sources=[D/'current',OLD/'project',OLD/'kit',OLD/'base-cache']
 for i,row in enumerate(M['files']):
  if i%128==0:fence()
  name=row['path'];assert row['sha']not in DENY and not Path(name).is_absolute()and '..'not in Path(name).parts
  src=next((root/name for root in sources if(root/name).is_file()and(root/name).stat().st_size==row['size']and sha(root/name,True)==row['sha']),None)
  assert src is not None,name;copy(src,K/name);assert sha(K/name,True)==row['sha']
 gitnames={r['path']for r in M['files']};extras=[r for r in C['project_files']if r['path']not in gitnames];assert len(extras)==70
 for row in extras:
  name=row['path'];assert not Path(name).is_absolute()and '..'not in Path(name).parts
  src=next((root/name for root in sources if(root/name).is_file()and(root/name).stat().st_size==row['bytes']and sha(root/name)==row['sha256']),None)
  assert src is not None,name;copy(src,K/name)
 assert sha(K/'bootstrap.py')=='eceb2ca259feaf642bc81172aca57a2a209cc34cd17f9b281861c51f21b95806'and sha(K/'restore_checkpoint.py')=='affd9ecf660f380c3632f78cd14fb7a36e00797ceb097cf48c0c60136fac0be8'
 assert sha(K/'BASE_PUBLIC_MANIFEST.json')=='0db67a018ad5fd9deba1cfb0bd8897537a330dd2353f169e6e387e86e09fce7a'
 base={r['path']:r for r in json.loads((K/'BASE_PUBLIC_MANIFEST.json').read_text())['files']};downloads=json.loads((K/'BASE_DOWNLOADS.json').read_text())['paths'];assert len(downloads)==48
 for name in downloads:
  row=base[name];src=OLD/'base-cache'/name;assert src.stat().st_size==row['bytes']and sha(src)==row['sha256'];copy(src,B/name)
 py=[sys.executable,'-I','-B']
 e42=K/'recoveries/epoch42-publication-20261026';e44=K/'recoveries/epoch44-publication-20261026-g9';e49=R/'recoveries/epoch49-stop-publication-20261027-g15'
 steps=[
 ('E42-archive-materialized.json',py+[str(e42/'restore_epoch42_archive.py'),'--kit-root',str(K),'--dest',str(R),'--restore-scope',str(W),'--archive','recoveries/epoch42-publication-20261026/immutable-e42-artifacts.zip','--projection','recoveries/epoch42-publication-20261026/ARCHIVE_PROJECTED_FILES.v1.json']),
 ('legacy22-verified.json',py+[str(K/'experiments/sota-2026-10-22/epoch22/publication/restore_archived_artifact_v1.py'),'--kit-root',str(K),'--archive','experiments/sota-2026-10-22/epoch22/publication/actualarchive.dev.raw.v1.json','--expected-archive-sha256','8ab3d8ec7856b423e6ff295f8e5ef74fc81dcaa3d3382fa2757e21563e0a99c6','--verify-only']),
 ('legacy26-verified.json',py+[str(K/'experiments/sota-2026-10-22/epoch26/publication/restore_epoch26_archives_v1.py'),'--kit-root',str(K),'--manifest','experiments/sota-2026-10-22/epoch26/publication/archive-transport.v1.json','--expected-manifest-sha256','9b79fc0119e1e5d15470201b7180f4b39e55b687a207dac1ff9f333d91a8ef24','--verify-only']),
 ('bootstrap-source-verified.json',py+[str(K/'bootstrap.py'),'--dest',str(R),'--base-local-directory',str(B)]),
 ('checkpoint-restored.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B)]),
 ('checkpoint-verify-only.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B),'--verify-only']),
 ('E42-archive-verify-only.json',py+[str(e42/'restore_epoch42_archive.py'),'--kit-root',str(K),'--dest',str(R),'--restore-scope',str(W),'--archive','recoveries/epoch42-publication-20261026/immutable-e42-artifacts.zip','--projection','recoveries/epoch42-publication-20261026/ARCHIVE_PROJECTED_FILES.v1.json','--verify-only']),
 ]
 cmd44=py+[str(e44/'restore_epoch44_overlay.py'),'--restore-scope',str(W),'--kit',str(K),'--dest',str(R),'--envelope',str(e44/'PUBLIC_ARTIFACTS.envelope.v1.json'),'--expected-envelope-sha','fc0f741eb7bd840a941e9ef3505b6924ab9be7842cd561722d4236bd98cf1d1d','--overlay',str(e44/'ARCHIVE_OVERLAY.v1.json'),'--expected-overlay-sha','815aaa11a602de738b1b62b4f9cd057639b8201b59007b3633588aaa674ba01d']
 cmd49=py+[str(e49/'restore_e49_overlay.py'),'--envelope',str(e49/'PUBLIC_RECOVERY.envelope.v1.json'),'--manifest',str(e49/'ARCHIVE_OVERLAY.v1.json'),'--project',str(R)]
 steps += [('E44-overlay-restored.json',cmd44),('E44-overlay-verified.json',cmd44+['--verify-only']),('E49-overlay-restored.json',cmd49),('E49-overlay-verified.json',cmd49+['--verify-only'])]
 gitindex={x['path']:x for x in M['files']}
 for name,argv in steps:
  fence();script=Path(argv[3]);relative=str(script.relative_to(K if script.is_relative_to(K)else R));row=gitindex[relative];assert sha(script,True)==row['sha']
  assert not any(x in argv for x in['--install-npm','--install-python','--acquire-models','--verify-models'])
  t=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True,timeout=60);write(W/name,{'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-t,'model_native_Gold_Git_calls':0});assert r.returncode==0,(name,r.stderr);print(name,'PASS',flush=True)
 for root in[K,R]:
  for i,row in enumerate(C['project_files']):
   if i%128==0:fence()
   p=root/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256'],str(p)
 for row in M['files']:assert(K/row['path']).stat().st_size==row['size']and sha(K/row['path'],True)==row['sha']
 fence();out={'schema_version':1,'epoch':58,'checks_passed':True,'actual_commit':M['commit'],'CP_rows_verified_per_kit_and_project':9048,'Git_leaves_verified_in_kit':9055,'CP_only_source_files':70,'base_files':48,'reviewed_bootstrap_executed_without_install_or_acquisition':True,'reviewed_restore_checkpoint_executed_and_verify_only_passed':True,'physical_distinct_copies_no_hardlinks':True,'source_only_steps':[{ 'path':n,'bytes':(W/n).stat().st_size,'sha256':sha(W/n)}for n,_ in steps],'counts':{p.name:sum(f.is_file()for f in p.rglob('*'))for p in[K,R,B]},'model_native_Gold_Git_calls':0}
 write(W/'SOURCE_RESTORATION_COMPLETE.v1.json',out);print(json.dumps(out))
if __name__=='__main__':main()
