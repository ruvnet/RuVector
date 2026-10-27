import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
W=Path(__file__).resolve().parent
OLD=Path('/dev/shm/ruvector-sota-2026-10-22/epoch56-ffdd0699f325')
K=W/'kit';R=W/'project';B=W/'base-cache'
def digest(p,git=False):
 h=hashlib.sha1() if git else hashlib.sha256()
 if git:h.update(b'blob '+str(p.stat().st_size).encode()+b'\0')
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def cp(src,dst):
 assert src.is_file() and not src.is_symlink()
 dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as a,dst.open('xb') as b:
  shutil.copyfileobj(a,b,1048576);b.flush();os.fsync(b.fileno())
 assert src.stat().st_ino!=dst.stat().st_ino
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
M=json.loads((W/'actual-git-map.json').read_text());C=json.loads((W/'remote-current/CHECKPOINT_MANIFEST.json').read_text())
assert len(M['files'])==8969 and len(C['project_files'])==8962
assert M['commit']=='5e426edd8ada8c0fdcec38ec8807054affc89e3c'
assert digest(W/'remote-current/CHECKPOINT_MANIFEST.json')=='1520b99a2233e54b2da642486df176eb93444c1172c42ed9d36a07c5fb30ca16'
sources=[W/'remote-current',OLD/'project',OLD/'kit',OLD/'base-cache']
for row in M['files']:
 p=row['path'];assert row['sha'] not in {'486888de9602bbb5d817a55f707b4fcfd11c65e0','3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'};assert not Path(p).is_absolute() and '..' not in Path(p).parts
 found=next((root/p for root in sources if (root/p).is_file() and (root/p).stat().st_size==row['size'] and digest(root/p,True)==row['sha']),None)
 assert found is not None,p
 cp(found,K/p);assert digest(K/p,True)==row['sha']
gitnames={r['path'] for r in M['files']};extra=[r for r in C['project_files'] if r['path'] not in gitnames]
assert len(extra)==70
for row in extra:
 found=next((root/row['path'] for root in sources if (root/row['path']).is_file() and (root/row['path']).stat().st_size==row['bytes'] and digest(root/row['path'])==row['sha256']),None)
 assert found is not None,row['path'];cp(found,K/row['path'])
base=json.loads((K/'BASE_PUBLIC_MANIFEST.json').read_text());assert digest(K/'BASE_PUBLIC_MANIFEST.json')=='0db67a018ad5fd9deba1cfb0bd8897537a330dd2353f169e6e387e86e09fce7a'
bi={x['path']:x for x in base['files']};paths=json.loads((K/'BASE_DOWNLOADS.json').read_text())['paths'];assert len(paths)==48
for name in paths:
 row=bi[name];src=OLD/'base-cache'/name;assert src.stat().st_size==row['bytes'] and digest(src)==row['sha256'];cp(src,B/name)
write(W/'CACHE_RECONSTRUCTION.actual.v1.json',{'schema_version':1,'epoch':57,'actual_commit':M['commit'],'Git_leaves':8969,'CP_only_files':70,'checkpoint_rows':8962,'base_files':48,'additional_CP_rows':extra,'all_fresh_Git_blob_hashes_verified':True,'physical_copy_no_hardlinks':True,'native_model_Gold_EVAL_Git_calls':0})
steps=['E42-archive-materialized.json','legacy22-verified.json','legacy26-verified.json','bootstrap-source-verified.json','checkpoint-restored.json','checkpoint-verify-only.json','E42-archive-verify-only.json','E44-overlay-restored.json','E44-overlay-verified.json','E49-overlay-restored.json','E49-overlay-verified.json']
allowed_scripts={'restore_epoch42_archive.py','restore_archived_artifact_v1.py','restore_epoch26_archives_v1.py','bootstrap.py','restore_checkpoint.py','restore_epoch44_overlay.py','restore_e49_overlay.py'}
STEP_RECEIPT_PINS={'E42-archive-materialized.json': {'bytes': 1858, 'sha256': '8caadf7882ae72a637e38dab7c8e3ca79790f29becf356cc8a193677b5dcae56'}, 'legacy22-verified.json': {'bytes': 1660, 'sha256': 'f64ae0374c94ba6ec79d66084420853bbf0730d04da5da91fd0725d4b4c22f91'}, 'legacy26-verified.json': {'bytes': 1819, 'sha256': '8df1426946ba00f638c1b80c6163f75f9a50055ef82de6ce3329dac539f012e1'}, 'bootstrap-source-verified.json': {'bytes': 8404, 'sha256': '8fb8753682f9dc43042bd1e9d92acb57a31291a624c36dca8c44bfc946a2b3e2'}, 'checkpoint-restored.json': {'bytes': 776, 'sha256': '8f81c39b6b803e580b4638358d81a0ba0114bb3ba74e8164e3332b48ccbdf9e7'}, 'checkpoint-verify-only.json': {'bytes': 797, 'sha256': '2cbae848d048024495e773699421cfccd511a9d21b88003fe0006d9b99083027'}, 'E42-archive-verify-only.json': {'bytes': 1878, 'sha256': 'b7b879a87253407c14d7e0041971f1d63012cde339544406847fdac0fb0e4743'}, 'E44-overlay-restored.json': {'bytes': 1425, 'sha256': '03faa8bc08e8f331034bf9a3953171c896914acaa651dda0c02ec69c50ca38ab'}, 'E44-overlay-verified.json': {'bytes': 1445, 'sha256': '947a5ff031688bcd46b5d6415be87fcb702be462753c55cb8c117b70094f4566'}, 'E49-overlay-restored.json': {'bytes': 1043, 'sha256': '6cd623b4739181672dd235312610eab4a1df1d7cecbf0cfb5e775ba127a4bd52'}, 'E49-overlay-verified.json': {'bytes': 1062, 'sha256': '109d23e57a27573751576920d474181830568eb3b63261116a7202c36ecb2977'}}
for name in steps:
 assert (OLD/name).stat().st_size==STEP_RECEIPT_PINS[name]['bytes'] and digest(OLD/name)==STEP_RECEIPT_PINS[name]['sha256']
 old=json.loads((OLD/name).read_text());argv=[x.replace(str(OLD),str(W)) for x in old['argv']];assert any(Path(v).name in allowed_scripts and (str(K) in v or str(R) in v) for v in argv);assert not any(v in argv for v in ('--install-npm','--install-python','--acquire-models'));t=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 d={'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-t,'native_model_Gold_EVAL_Git_calls':0};write(W/name,d)
 assert r.returncode==0,(name,r.stderr);print(name,'PASS',flush=True)
for root in (K,R):
 for row in C['project_files']:
  p=root/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p)
for row in M['files']:assert (K/row['path']).stat().st_size==row['size'] and digest(K/row['path'],True)==row['sha']
write(W/'SOURCE_RESTORATION_COMPLETE.v1.json',{'schema_version':1,'epoch':57,'actual_commit':M['commit'],'checkpoint_rows_verified_per_root':8962,'Git_leaves_verified_in_kit':8969,'base_files':48,'counts':{str(p.name):sum(f.is_file() for f in p.rglob('*')) for p in (K,R,B)},'source_only_steps':[{ 'path':n,'bytes':(W/n).stat().st_size,'sha256':digest(W/n)} for n in steps],'physical_copy_no_hardlinks':True,'source_checkpoint_sha256':digest(W/'remote-current/CONTINUATION.json'),'checkpoint_manifest_sha256':digest(W/'remote-current/CHECKPOINT_MANIFEST.json'),'immutable_bootstrap_sha256':digest(K/'bootstrap.py'),'immutable_restore_sha256':digest(K/'restore_checkpoint.py'),'native_model_Gold_EVAL_Git_calls':0,'all_passed':True})
print((W/'SOURCE_RESTORATION_COMPLETE.v1.json').read_text())
