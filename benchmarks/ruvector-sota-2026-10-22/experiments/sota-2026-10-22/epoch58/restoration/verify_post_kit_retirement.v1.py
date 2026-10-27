"""Read-only full current R/K46 source verification after reviewed Kit twin retirement."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat
D=Path('/dev/shm/ruvector-e58-data');W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');R=W/'project';K=W/'kit';B=W/'base-cache';OLD=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
OWNER={'path':str(W/'guardian26/owner.json'),'bytes':831,'sha256':'78d50fe9cf3cd9e4d70a84644a152e921fb2cc5cd3b0ee143888c2c85e684bac'}
FENCE={'path':str(W/'guardian26/owner_fence_v2.py'),'bytes':3376,'sha256':'5b56fa209a1d947d1bbabe31e5e4ed3d52e8dc9d2258e76fa8577c36b459d693'}
cache={}
def digests(p):
 p=Path(p);assert p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in[p,*p.parents]);a=p.stat();assert stat.S_ISREG(a.st_mode);sig=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
 if (str(p),sig)in cache:return cache[str(p),sig]
 h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(a.st_size).encode()+b'\0')
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);g.update(b)
 z=p.stat();assert sig==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns);value=(a.st_size,h.hexdigest(),g.hexdigest());cache[str(p),sig]=value;return value
def exact(p,size,sha):assert digests(p)[:2]==(size,sha);return p
def ref(p):size,sha,_=digests(p);return{'path':str(p),'bytes':size,'sha256':sha}
def write(p,v):
 b=(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
def guard():
 exact(Path(FENCE['path']),FENCE['bytes'],FENCE['sha256']);s=importlib.util.spec_from_file_location('post_kit_owner',FENCE['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.verify_owner(OWNER)
guard()
exact(D/'actual-git-map.json',3112129,'554e42b12b10c0142f19f526bd5846fe14619ac2de5fce14297116e3722d0636');exact(D/'current/CHECKPOINT_MANIFEST.json',2117492,'ae9b6df276bbd2b3446fd21b4a3a6b2a8e3e2b96083716a3f1478318c4144d8f');exact(W/'kit-twin-retirement/KIT_TWIN_RETIREMENT_MANIFEST.v1.json',9309238,'f934d6ba2f533553b83970f63e4c16d9801434442ef305e6224c51499d35d415')
M=json.loads((D/'actual-git-map.json').read_bytes());C=json.loads((D/'current/CHECKPOINT_MANIFEST.json').read_bytes());KM=json.loads((W/'kit-twin-retirement/KIT_TWIN_RETIREMENT_MANIFEST.v1.json').read_bytes());A=json.loads((W/'KIT_TWIN_RETIREMENT.actual.v1.json').read_bytes());assert M['commit']=='db7b9b21f4a13e5fc3444a8fe40de26583558e2e'and len(M['files'])==9055 and len(C['project_files'])==9048 and A['checks_passed']is True and A['exact_kit_files_retired']==9134 and A['preserved_kit_files']==46
for row in KM['twins']:
 assert not os.path.lexists(row['kit']);exact(Path(row['project_mirror']),row['bytes'],row['sha256'])
for row in KM['preserved_kit_files']:exact(Path(row['path']),row['signature']['size'],row['sha256'])
actualK={str(p)for p in K.rglob('*')if p.is_file()};assert actualK=={x['path']for x in KM['preserved_kit_files']}
old_inode_checks=0
for row in C['project_files']:
 p=R/row['path'];exact(p,row['bytes'],row['sha256']);a=p.stat()
 for old in [OLD/'project'/row['path'],OLD/'kit'/row['path'],OLD/'base-cache'/row['path']]:
  if old.exists():z=old.stat();assert(a.st_dev,a.st_ino)!=(z.st_dev,z.st_ino);old_inode_checks+=1
rows=[];counts={'current':0,'project':0,'kit':0,'base-cache':0}
for row in M['files']:
 roots=[D/'current',R,K,B]if row['path']in['CONTINUATION.json','CHECKPOINT_MANIFEST.json','ACTIVE_RUN.json']else[R,K,D/'current',B];source=None
 for root in roots:
  p=root/row['path']
  if p.is_file():size,sha,git=digests(p)
  else:continue
  if(size,git)==(row['size'],row['sha']):source=p;break
 assert source is not None,row['path'];label='current'if source.is_relative_to(D/'current')else source.relative_to(W).parts[0];counts[label]+=1
 rows.append({'path':row['path'],'physical_source':str(source),'bytes':row['size'],'GitSHA':row['sha'],'sha256':digests(source)[1]})
byname={r['path']:r for r in rows};gitpaths=set(byname)
for row in C['project_files']:
 if row['path']not in gitpaths:rows.append({'path':row['path'],'physical_source':str(R/row['path']),'bytes':row['bytes'],'sha256':row['sha256'],'CP_only':True})
assert len(rows)==9125
base=json.loads(Path(byname['BASE_PUBLIC_MANIFEST.json']['physical_source']).read_bytes());paths=json.loads(Path(byname['BASE_DOWNLOADS.json']['physical_source']).read_bytes())['paths'];base={x['path']:x for x in base['files']};assert len(paths)==48
for name in paths:exact(B/name,base[name]['bytes'],base[name]['sha256'])
guard()
mapref=write(D/'CURRENT_VERIFIED_DONOR_MAP_POST_KIT.v2.json',{'schema_version':2,'epoch':58,'source_checkpoint_commit':M['commit'],'Git_leaves':9055,'CP_only_rows':70,'layout':'R first, exact current controls, retained K46 fallback; no full-Kit assumption.','physical_layout_Git':counts,'files':rows})
report={'schema_version':2,'epoch':58,'checks_passed':True,'source_checkpoint_commit':M['commit'],'source_map_ref':ref(D/'actual-git-map.json'),'checkpoint_manifest_ref':ref(D/'current/CHECKPOINT_MANIFEST.json'),'retirement_receipt_ref':ref(W/'KIT_TWIN_RETIREMENT.actual.v1.json'),'historical_full_Kit_restoration_proof':ref(D/'SOURCE_RESTORATION_PHYSICAL_TARGETS_INDEPENDENT.v1.json'),'CP_rows_rehashed_in_R':9048,'CP_bytes_rehashed_in_R':sum(x['bytes']for x in C['project_files']),'Git_leaves_rehashed_current_layout':9055,'retired_Kit_twins_absent_and_R_mirrors_rehashed':9134,'retained_Kit_files_rehashed':46,'base_cache_files_rehashed':48,'R_old_donor_distinct_inode_checks':old_inode_checks,'physical_layout_Git':counts,'donor_map_ref':mapref,'data_payload_semantic_decodes_model_native_Gold_network_Git_or_cleanup_mutations':0}
reportref=write(D/'SOURCE_LAYOUT_AFTER_KIT_RETIREMENT_INDEPENDENT.v2.json',report);print(json.dumps({'report':reportref,'donor_map':mapref,'layout':counts},sort_keys=True))
