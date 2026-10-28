"""Restore reviewed E59 public161 overlay after standard restoration. No science."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path,PurePosixPath
import argparse,base64,hashlib,io,json,os,stat,tarfile
ROOTS=tuple('experiments/sota-2026-10-22/'+x+'/'for x in('epoch59','audit/epoch59','runtime/epoch59'))
OLD=Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983')
ARCHIVE_SHA='1c43224ad190f7514d750b2a265f12376447cbd21ac24603426579f98996c4e9'
MANIFEST_SHA='4ff12f979068f0874f7bdb80671ea04f680d560f2aa4021481f70e10f6faeffe'
def need(v,c):
 if not v:raise ValueError(c)
def sha(b):return hashlib.sha256(b).hexdigest()
def rel(s):
 need(type(s)is str and s and not PurePosixPath(s).is_absolute()and '\\'not in s and all(x not in('','.','..')for x in s.split('/')),'canonical relative path');return s
def canonical(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and p.resolve()==p and not any(x.is_symlink()for x in[p,*p.parents]),'canonical no-symlink path');return p
def body(p):
 p=canonical(p);a=p.stat();need(stat.S_ISREG(a.st_mode),'regular file');b=p.read_bytes();z=p.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)and len(b)==z.st_size,'stable file');return b
def encoded(d):return(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def write(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);canonical(p.parent);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def unique(pairs):
 d={}
 for k,v in pairs:need(k not in d,'duplicate JSON key');d[k]=v
 return d
def parse(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def validate(eb,ob):
 e=parse(eb);o=parse(ob);need(set(e)=={'schema_version','format','archive_bytes','archive_sha256','content_base64'}and type(e['schema_version'])is int and e['schema_version']==1 and e['format']=='tar.gz','exact envelope schema')
 need(type(e['archive_bytes'])is int and e['archive_bytes']==948170 and e['archive_sha256']==ARCHIVE_SHA and type(e['content_base64'])is str and len(e['content_base64'])==1264228,'fixed canonical archive encoding')
 raw=base64.b64decode(e['content_base64'],validate=True);need(len(raw)==948170 and sha(raw)==ARCHIVE_SHA,'exact archive payload')
 need(o['schema_version']==1 and o['epoch']==59 and o['kind']=='E59_ARCHIVE_OVERLAY_ONLY_NO_SCIENCE'and o['source_commit']=='993dce6842bf602dd87864d00c1621a56eebc613'and o['CP_project_files_includes_overlay78']is False and o['no_scientific_execution_authorized']is True,'exact overlay semantics')
 rows=o['files'];need(type(rows)is list and len(rows)==161,'exact161 scientific rows');expected={rel(r['path']):r for r in rows};need(len(expected)==161,'unique161')
 need(sum(r['source_Git_present']is True for r in rows)==83 and sum(r['source_Git_present']is False for r in rows)==78 and all(type(r['source_Git_present'])is bool and type(r['bytes'])is int and 0<=r['bytes']<=2000000 and any(r['path'].startswith(root)for root in ROOTS)for r in rows),'83source/78overlay only in3sealedroots')
 files={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
  members=t.getmembers();need(len(members)==167 and sum(m.size for m in members)==5954606,'exact167 bounded archive members')
  need(all(m.isreg()and not m.issym()and not m.islnk()and 0<=m.size<=2000000 for m in members),'regular bounded members')
  names=[rel(m.name)for m in members];need(len(set(names))==167,'unique relative member names')
  mb=t.extractfile('RECOVERY_MANIFEST.json').read(70497);need(len(mb)==70496 and sha(mb)==MANIFEST_SHA,'exact reviewed recovery manifest');rm=parse(mb);need(rm['public_only_archive_approved']is True and rm['checks_passed']is True and len(rm['files'])==166,'reviewed public-only payload')
  allrows={rel(r['archive_path']):r for r in rm['files']};need(len(allrows)==166 and set(allrows)|{'RECOVERY_MANIFEST.json'}==set(names),'all exact payload members')
  for m in members:
   if m.name=='RECOVERY_MANIFEST.json':continue
   r=allrows[m.name];need(type(r['bytes'])is int and m.size==r['bytes'],'member declared size');b=t.extractfile(m).read(r['bytes']+1);need(len(b)==r['bytes']and sha(b)==r['sha256'],'every archive member exact hash')
   if m.name.startswith('project/'):
    p=rel(m.name[8:]);need(p in expected and p not in files and expected[p]['bytes']==len(b)and expected[p]['sha256']==sha(b),'only exact E59 scientific projection');files[p]=b
 need(set(files)==set(expected),'every161E59memberpresent');return files,expected

def run(a):
 scope=canonical(a.restore_scope);kit=canonical(a.kit);dest=canonical(a.dest);need(scope.is_dir()and kit.is_dir()and dest.is_dir()and kit!=dest and not kit.is_relative_to(dest)and not dest.is_relative_to(kit),'existing separate KIT and project')
 for p in(scope,kit,dest):need(not(p==OLD or p.is_relative_to(OLD)or OLD.is_relative_to(p)),'never mutate old E59 workspace')
 need(kit.is_relative_to(scope)and dest.is_relative_to(scope),'fresh explicit restore scope');eb=body(a.envelope);ob=body(a.overlay);need(sha(eb)==a.expected_envelope_sha and sha(ob)==a.expected_overlay_sha,'pinned outer envelope/overlay bytes');files,rows=validate(eb,ob)
 # Validate all members and all destination identities before any mutation.
 for root in(kit,dest):
  for p,b in files.items():
   t=canonical(root/p)
   if a.verify_only or rows[p]['source_Git_present']:need(t.exists()and body(t)==b,'83source or161verify-only byte identity')
   else:need(not t.exists(),'new78mustbeabsent; no silent overwrite or replay')
 if not a.verify_only:
  write(scope/'E59_ARCHIVE_OVERLAY_ATTEMPT.actual.json',encoded({'epoch':59,'kind':'PUBLIC_METADATA_OVERLAY_ONLY_NO_SCIENCE','canonical_members':161,'new_files_per_root':78,'CP_project_files_modified':False,'actual_archive_sha256':ARCHIVE_SHA}))
  for root in(kit,dest):
   for p,b in files.items():
    if not rows[p]['source_Git_present']:write(root/p,b)
 result={'checks_passed':True,'epoch':59,'verify_only':a.verify_only,'all167_archive_members_validated_before_write':True,'all161_scientific_members_validated':True,'source83_byte_identical':True,'new78_overlay_per_root':78,'CP_project_files_modified':False,'scientific_model_tokenizer_Gold_API_Git_calls':0,'source_split_and_confirmation_labels_unchanged':True}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--restore-scope',type=Path,required=True);p.add_argument('--kit',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--envelope',type=Path,required=True);p.add_argument('--expected-envelope-sha',required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--expected-overlay-sha',required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();print(json.dumps(run(a),sort_keys=True))
