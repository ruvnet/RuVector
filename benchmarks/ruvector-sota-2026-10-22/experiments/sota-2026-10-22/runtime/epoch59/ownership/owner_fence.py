"""Exact shared E59 owner fence; this function alone grants no scientific authority."""
from pathlib import Path
import fcntl,hashlib,json,os,stat,time
EXPECTED_OWNER={'path':'/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983/guardian27/owner.json','bytes':831,'sha256':'28f34b4b0af71fbd39d94a05766ba7bf61798ae540419d1838ea3ca21d5cce9f'}
def need(ok,message):
 if not ok:raise ValueError(message)
def unique(pairs):
 out={}
 for k,v in pairs:
  need(k not in out,'duplicate owner JSON key');out[k]=v
 return out
def verify_owner(expected_owner):
 need(type(expected_owner)is dict and set(expected_owner)=={'path','bytes','sha256'} and type(expected_owner['path'])is str and type(expected_owner['bytes'])is int and type(expected_owner['sha256'])is str and expected_owner==EXPECTED_OWNER,'exact typed owner27 reference required')
 need(os.sched_getaffinity(0)=={0,1},'caller CPU0,1 required')
 p=Path(EXPECTED_OWNER['path']);need(not any(x.is_symlink()for x in[p,*p.parents]),'owner path symlink')
 before=p.stat();need(stat.S_ISREG(before.st_mode),'regular owner file');raw=p.read_bytes();after=p.stat()
 sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(sig(before)==sig(after)and len(raw)==831 and hashlib.sha256(raw).hexdigest()==EXPECTED_OWNER['sha256'],'stable owner27 bytes/hash')
 o=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite owner JSON')))
 need(o['epoch']==59 and o['generation']==27 and o['pid']==198313 and o['startticks']==46302984,'exact guardian27 identity')
 need(o['run_id']=='epoch59-9b677a916983-generation27'and o['scope']=='ruvnet/RuVector:benchmarks/ruvector-sota-2026-10-22','exact run/scope')
 need(o['purpose']=='BOUNDED_RESEARCH_E59_TYPED_NLI_AND_NATIVE_EVIDENCE'and o['scientific_invocation_authorized_by_lock_alone']is False,'bounded work fence only')
 need(o['launch_mode']=='DETACHED_BOUNDED_SESSION'and o['session_id']==o['process_group']==198313,'detached owner context')
 need(time.time()<o['expires_unix']and not os.path.lexists(p.parent/'release.request')and not os.path.lexists(p.parent/'released.json'),'owner expired/releasing')
 need(o['kernel_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and o['PID_namespace']==os.readlink('/proc/self/ns/pid'),'boot/namespace differs')
 fields=Path('/proc/198313/stat').read_text().rsplit(')',1)[1].split()
 need(int(fields[19])==46302984 and fields[0]not in('Z','X')and int(fields[1])==1 and int(fields[2])==int(fields[3])==198313,'live detached original guardian27 birth')
 need(os.sched_getaffinity(198313)=={0,1},'guardian CPU0,1')
 lock=Path('/tmp/ruvector-sota-2026-10-22/exclusive.work.lock')
 need(o['lock']==str(lock)and o['lock_device']==33 and o['lock_inode']==13626 and o['lock_fd']==3,'original lock declaration')
 a=Path('/proc/198313/fd/3').stat();b=lock.stat();need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(33,13626),'original inode/FD3')
 need(any('FLOCK'in s and 'WRITE'in s and s.split()[4]=='198313'and s.split()[5]=='00:21:13626'for s in Path('/proc/locks').read_text().splitlines()),'guardian27 kernel FLOCK absent')
 fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:
   fcntl.flock(fd,fcntl.LOCK_UN);raise ValueError('scope lock was free')
 finally:os.close(fd)
 return o
