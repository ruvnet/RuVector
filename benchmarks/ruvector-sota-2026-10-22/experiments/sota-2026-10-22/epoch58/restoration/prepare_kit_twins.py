"""Read-only source-mirror inventory for an independently approved kit retirement."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1')
K,R=W/'kit',W/'project'
D=W/'kit-twin-retirement'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sig(p):
 s=p.lstat();return dict(dev=s.st_dev,ino=s.st_ino,mode=s.st_mode,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink)
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def put(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 assert not sys.flags.optimize and os.sched_getaffinity(0)=={0,1}
 p=Path('/dev/shm/ruvector-e58-data/reconstruct_sources.proposed.py');assert sha(p)=='e6632280704fe3d40efc6b3dfee3137370dcddb32a0837cbe7d41f1904e5bb7a'
 s=importlib.util.spec_from_file_location('reviewed_guardian25_fence',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.fence()
 q=Path('/dev/shm/ruvector-e58-audit/ACTUAL_SOURCE_RESTORATION_INDEPENDENT.v1.json');assert q.stat().st_size==12612 and sha(q)=='5c2788f5fc22b8dab4235c16fdd9b6aac01111c25b9de775185b7afba5fba140'
 assert not D.exists();D.mkdir()
 actual=sorted(K.rglob('*'));assert not any(p.is_symlink()for p in actual)
 files=[p for p in actual if p.is_file()];assert len(files)==9180
 twins=[];keep=[];dirs=[]
 for p in actual:
  assert not any(x.is_symlink()for x in[p,*p.parents])
  if p.is_dir():
   z=p.stat();dirs.append(dict(path=str(p),dev=z.st_dev,ino=z.st_ino,mode=z.st_mode))
 for i,p in enumerate(files):
  if i%128==0:m.fence()
  rel=p.relative_to(K);r=R/rel;ps=sig(p);ph=sha(p)
  if r.is_file()and not any(x.is_symlink()for x in[r,*r.parents])and r.stat().st_size==p.stat().st_size and sha(r)==ph:
   rs=sig(r);assert(ps['dev'],ps['ino'])!=(rs['dev'],rs['ino'])
   twins.append(dict(relative_path=str(rel),kit=str(p),project_mirror=str(r),bytes=ps['size'],sha256=ph,kit_signature=ps,project_signature=rs))
  else:keep.append(dict(path=str(p),signature=ps,sha256=ph))
  assert sig(p)==ps
 cp=Path('/dev/shm/ruvector-e58-data/current/CHECKPOINT_MANIFEST.json');assert sha(cp)=='ae9b6df276bbd2b3446fd21b4a3a6b2a8e3e2b96083716a3f1478318c4144d8f'
 rows=json.loads(cp.read_text())['project_files'];assert len(rows)==9048
 for i,row in enumerate(rows):
  if i%128==0:m.fence()
  p=R/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
 full=D/'KIT_TWIN_RETIREMENT_MANIFEST.v1.json';put(full,dict(epoch=58,twins=twins,preserved_kit_files=keep,unchanged_kit_directory_topology=dirs))
 plan=dict(schema_version=1,epoch=58,kit_root=str(K),preserved_project_root=str(R),delete_exact_individual_kit_files_only=True,directory_removal_authorized=False,kit_twin_files_to_retire=len(twins),logical_bytes_reclaimable=sum(x['bytes']for x in twins),preserved_kit_files=len(keep),all9048_CP_rows_remain_in_R=True,full_manifest=ref(full),source_restoration_independent_proof=ref(q),source_inventory=ref(Path(__file__).resolve()),model_native_Gold_Git_calls=0)
 pp=D/'KIT_TWIN_RETIREMENT_PLAN.v1.json';put(pp,plan);m.fence();print(json.dumps(dict(plan=ref(pp),twins=len(twins),bytes=plan['logical_bytes_reclaimable'],keep=len(keep))))
if __name__=='__main__':main()
