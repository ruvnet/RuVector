"""Independent complete physical target and distinct-inode verification; no source payload decode."""
from pathlib import Path
import hashlib,json,os
D=Path('/dev/shm/ruvector-e58-data');W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');K=W/'kit';R=W/'project';B=W/'base-cache';OLD=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
def sha(p,git=False):
 h=hashlib.sha1()if git else hashlib.sha256()
 if git:h.update(b'blob '+str(p.stat().st_size).encode()+b'\0')
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):return{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
M=json.loads((D/'actual-git-map.json').read_text());C=json.loads((D/'current/CHECKPOINT_MANIFEST.json').read_text());a=json.loads((W/'SOURCE_RESTORATION_COMPLETE.v1.json').read_text())
assert M['commit']==a['actual_commit']=='db7b9b21f4a13e5fc3444a8fe40de26583558e2e'and a['checks_passed']is True
assert len(M['files'])==9055 and len(C['project_files'])==9048
donors=[];counts={'current':0,'project':0,'kit':0,'base-cache':0};CPbytes=0
for row in C['project_files']:
 p=R/row['path'];q=K/row['path'];assert p.is_file()and q.is_file()and not p.is_symlink()and not q.is_symlink()
 assert p.stat().st_size==q.stat().st_size==row['bytes']and sha(p)==sha(q)==row['sha256'];CPbytes+=row['bytes']
 assert(p.stat().st_dev,p.stat().st_ino)!=(q.stat().st_dev,q.stat().st_ino)
 for old in[OLD/'project'/row['path'],OLD/'kit'/row['path'],OLD/'base-cache'/row['path']]:
  if old.exists():assert(p.stat().st_dev,p.stat().st_ino)!=(old.stat().st_dev,old.stat().st_ino)and(q.stat().st_dev,q.stat().st_ino)!=(old.stat().st_dev,old.stat().st_ino)
for row in M['files']:
 p=K/row['path'];assert p.stat().st_size==row['size']and sha(p,True)==row['sha']
 roots=[D/'current',R,K,B]if row['path']in['CONTINUATION.json','CHECKPOINT_MANIFEST.json','ACTIVE_RUN.json']else[R,K,D/'current',B]
 p=next((root/row['path']for root in roots if(root/row['path']).is_file()and(root/row['path']).stat().st_size==row['size']and sha(root/row['path'],True)==row['sha']),None);assert p is not None,row['path'];counts[p.relative_to(D).parts[0]if p.is_relative_to(D)else p.relative_to(W).parts[0]]+=1
 donors.append({'path':row['path'],'physical_source':str(p),'bytes':row['size'],'GitSHA':row['sha'],'sha256':sha(p)})
gitpaths={r['path']for r in M['files']}
for row in C['project_files']:
 if row['path']not in gitpaths:donors.append({'path':row['path'],'physical_source':str(R/row['path']),'bytes':row['bytes'],'sha256':row['sha256'],'CP_only':True})
assert len(donors)==9125
base={r['path']:r for r in json.loads((K/'BASE_PUBLIC_MANIFEST.json').read_text())['files']};paths=json.loads((K/'BASE_DOWNLOADS.json').read_text())['paths'];assert len(paths)==48
for name in paths:assert(B/name).stat().st_size==base[name]['bytes']and sha(B/name)==base[name]['sha256']
for s in a['source_only_steps']:
 p=W/s['path'];assert p.stat().st_size==s['bytes']and sha(p)==s['sha256']and json.loads(p.read_text())['exit_code']==0
write(D/'CURRENT_VERIFIED_DONOR_MAP.v1.json',{'schema_version':1,'epoch':58,'actual_commit':M['commit'],'Git_leaves':9055,'CP_only_rows':70,'physical_layout_Git':counts,'files':donors})
report={'schema_version':1,'epoch':58,'checks_passed':True,'commit':M['commit'],'full_source_restoration_actual':ref(W/'SOURCE_RESTORATION_COMPLETE.v1.json'),'Git_files_physically_rehashed':9055,'CP_rows_physically_rehashed_per_root':9048,'CP_bytes_per_root':CPbytes,'CP_roots_distinct_inodes_from_each_other_and_old_donors':True,'base_files_physically_rehashed':48,'physical_layout_Git':counts,'donor_map':ref(D/'CURRENT_VERIFIED_DONOR_MAP.v1.json'),'source_only_steps':a['source_only_steps'],'immutable_bootstrap':ref(K/'bootstrap.py'),'immutable_restore':ref(K/'restore_checkpoint.py'),'first_failed_CP_bound_receipt_preserved':ref(W/'legacy22-verified.json'),'model_native_Gold_DATA_payload_decode_Git_mutations':0}
write(D/'SOURCE_RESTORATION_PHYSICAL_TARGETS_INDEPENDENT.v1.json',report);print(json.dumps({'report':ref(D/'SOURCE_RESTORATION_PHYSICAL_TARGETS_INDEPENDENT.v1.json'),'donor_map':report['donor_map'],'counts':counts}))
