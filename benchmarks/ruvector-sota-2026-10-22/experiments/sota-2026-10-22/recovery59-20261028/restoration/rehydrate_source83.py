"""Root-only newKit source83 rehydration after verified streamed retirement; no R writes."""
import argparse,hashlib,importlib.util,json,os,stat,time
from pathlib import Path
W=Path('/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028');K=W/'kit';R=W/'project';G=W/'guardian28';P=Path('/dev/shm/ruvector-oct28-publication')
OWNER={'path':str(G/'owner.json'),'bytes':848,'sha256':'39acf0db29a8dff63e76a550927644fc4170d662152961fc0f244147ea9c069e'}
def need(v,c):
 if not v:raise ValueError(c)
def sha(b):return hashlib.sha256(b).hexdigest()
def body(p):
 p=Path(p);need(p.is_absolute()and p.resolve()==p and not any(q.is_symlink()for q in[p,*p.parents]),'canonical no-symlink path');a=p.stat();need(stat.S_ISREG(a.st_mode),'regular source');b=p.read_bytes();z=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'stable source');return b

def ref(p):b=body(p);return{'path':str(p),'bytes':len(b),'sha256':sha(b)}
def checked(r):
 need(type(r)is dict and set(r)=={'path','bytes','sha256'}and type(r['path'])is str and type(r['bytes'])is int and type(r['sha256'])is str,'exact typed3 reference');b=body(r['path']);need(len(b)==r['bytes']and sha(b)==r['sha256'],'exact ref body');return b
def put(p,d):
 b=(json.dumps(d,indent=2,sort_keys=True)+'\n').encode();p=Path(p);need(p.is_absolute()and p.resolve()==p and not any(q.is_symlink()for q in[p,*p.parents]),'safe output');p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);return ref(p)
def run(a):
 need(os.sched_getaffinity(0)=={0,1},'CPU01');cb=body(a.config);need(sha(cb)==a.config_sha,'config pinned');c=json.loads(cb);ab=body(a.approval);need(sha(ab)==a.approval_sha,'approval pinned');ap=json.loads(ab);need(ap['checks_passed']is True and ap['Kit_source83_rehydration_allowed']is True and ap['configuration_sha256']==a.config_sha,'independent exact rehydration approval')
 need(c['kind']=='SOURCE_ONLY_NEW_KIT83_REHYDRATION'and c['epoch']==59 and c['Kit']==str(K)and c['R']==str(R)and c['owner']==OWNER,'exact rehydration scope')
 fb=checked(c['owner_fence_source']);need(sha(fb)=='f4502591a06b4518d77eecd4f7f0eecee2d1cf5383cf24f8b17e6c5663df3791','fixed fence source');sp=importlib.util.spec_from_file_location('rehydrate_g28_fence',c['owner_fence_source']['path']);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.verify_owner(OWNER)
 preservation=json.loads(checked(c['source_preservation_review']));need(preservation['checks_passed']is True,'actual source preservation independently reviewed')
 overlay=json.loads(checked(c['archive_overlay']));sm=json.loads(checked(c['fresh_Git_map']));need(sm['commit']=='993dce6842bf602dd87864d00c1621a56eebc613','fresh SOURCE map');gm={r['path']:r for r in sm['files']};rows=[r for r in overlay['files']if r['source_Git_present']is True];need(len(rows)==83,'source83 only')
 need(K.is_dir()and R.is_dir()and K.resolve()==K and R.resolve()==R and K!=R,'existing separate freshKitR');attempt=W/'KIT_SOURCE83_REHYDRATION_ATTEMPT.actual.v1.json';result=W/'KIT_SOURCE83_REHYDRATION.actual.v1.json';need(not attempt.exists()and not result.exists(),'no prior attempt or replay')
 prepared=[]
 for r in rows:
  p=r['path'];need(type(p)is str and p and not p.startswith('/')and '\\'not in p and all(x not in('','.','..')for x in p.split('/')),'relative source path');s=R/p;t=K/p;need(not t.exists()and not t.is_symlink()and t.resolve()==t,'83Kit targets absent before attempt');b=body(s);h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();need(len(b)==r['bytes']and sha(b)==r['sha256']and p in gm and gm[p]['sha']==h and gm[p]['size']==len(b),'source83 exact frozen hash and fresh Git identity');st=s.stat();prepared.append((r,s,t,b,st.st_dev,st.st_ino))
 need(len({str(t)for r,s,t,b,dev,ino in prepared})==83,'83uniqueKit targets');put(attempt,{'epoch':59,'kind':c['kind'],'configuration_sha256':a.config_sha,'actual_prechecked_files':83,'actual_prechecked_bytes':sum(len(b)for r,s,t,b,dev,ino in prepared),'source_R_writes':0,'retries_authorized':0,'scientific_Git_calls':0})
 written=[]
 for r,s,t,b,dev,ino in prepared:
  m.verify_owner(OWNER);need((s.stat().st_dev,s.stat().st_ino)==(dev,ino)and body(s)==b,'source R preserved before write');t.parent.mkdir(parents=True,exist_ok=True);need(t.parent.resolve()==t.parent and not any(q.is_symlink()for q in[t,*t.parents]),'target parent no symlink');fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  d=os.open(t.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d);z=t.stat();need(body(t)==b and(z.st_dev,z.st_ino)!=(dev,ino),'new distinct Kit inode exact body');written.append({'path':r['path'],'bytes':len(b),'sha256':sha(b),'R_device':dev,'R_inode':ino,'Kit_device':z.st_dev,'Kit_inode':z.st_ino})
 for r,s,t,b,dev,ino in prepared:need((s.stat().st_dev,s.stat().st_ino)==(dev,ino)and body(s)==b and body(t)==b,'all83exact and originalR inodes preserved after')
 m.verify_owner(OWNER);return put(result,{'schema_version':1,'epoch':59,'kind':c['kind'],'checks_passed':True,'actual_files_rehydrated':83,'actual_bytes_rehydrated':sum(r['bytes']for r in written),'all_original_R_bytes_and_inodes_preserved':True,'all_new_Kit_files_OEXCL_and_distinct_R_inodes':True,'owner':OWNER,'configuration_ref':ref(a.config),'approval_ref':ref(a.approval),'scientific_Git_calls':0,'rows':written,'observed_unix':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha',required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha',required=True);a=p.parse_args();print(json.dumps(run(a),sort_keys=True))
