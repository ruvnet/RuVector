"""Exact shared E58 owner fence; this function alone grants no scientific authority."""
from pathlib import Path
import fcntl,hashlib,json,os,stat,time
EXPECTED_OWNER={'path':'/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1/guardian26/owner.json','bytes':831,'sha256':'78d50fe9cf3cd9e4d70a84644a152e921fb2cc5cd3b0ee143888c2c85e684bac'}
def need(ok,message):
 if not ok:raise ValueError(message)
def unique(pairs):
 out={}
 for k,v in pairs:
  need(k not in out,'duplicate owner JSON key');out[k]=v
 return out
def verify_owner(expected_owner):
 need(type(expected_owner)is dict and set(expected_owner)=={'path','bytes','sha256'} and type(expected_owner['path'])is str and type(expected_owner['bytes'])is int and type(expected_owner['sha256'])is str and expected_owner==EXPECTED_OWNER,'exact typed owner26 reference required')
 need(os.sched_getaffinity(0)=={0,1},'caller CPU0,1 required')
 p=Path(EXPECTED_OWNER['path']);need(not any(x.is_symlink()for x in[p,*p.parents]),'owner path symlink')
 before=p.stat();need(stat.S_ISREG(before.st_mode),'regular owner file');raw=p.read_bytes();after=p.stat()
 sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(sig(before)==sig(after)and len(raw)==831 and hashlib.sha256(raw).hexdigest()==EXPECTED_OWNER['sha256'],'stable owner26 bytes/hash')
 o=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite owner JSON')))
 need(o['epoch']==58 and o['generation']==26 and o['pid']==192464 and o['startticks']==45182825,'exact guardian26 identity')
 need(o['run_id']=='epoch58-db7b9b21f4a1-generation26'and o['scope']=='ruvnet/RuVector:benchmarks/ruvector-sota-2026-10-22','exact run/scope')
 need(o['purpose']=='BOUNDED_RESEARCH_E58_TYPED_NLI_AND_NATIVE_EVIDENCE'and o['scientific_invocation_authorized_by_lock_alone']is False,'bounded work fence only')
 need(o['launch_mode']=='DETACHED_BOUNDED_SESSION'and o['session_id']==o['process_group']==192464,'detached owner context')
 need(time.time()<o['expires_unix']and not os.path.lexists(p.parent/'release.request')and not os.path.lexists(p.parent/'released.json'),'owner expired/releasing')
 need(o['kernel_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and o['PID_namespace']==os.readlink('/proc/self/ns/pid'),'boot/namespace differs')
 fields=Path('/proc/192464/stat').read_text().rsplit(')',1)[1].split()
 need(int(fields[19])==45182825 and fields[0]not in('Z','X')and int(fields[1])==1 and int(fields[2])==int(fields[3])==192464,'live detached original guardian26 birth')
 need(os.sched_getaffinity(192464)=={0,1},'guardian CPU0,1')
 lock=Path('/tmp/ruvector-sota-2026-10-22/exclusive.work.lock')
 need(o['lock']==str(lock)and o['lock_device']==33 and o['lock_inode']==13626 and o['lock_fd']==3,'original lock declaration')
 a=Path('/proc/192464/fd/3').stat();b=lock.stat();need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(33,13626),'original inode/FD3')
 need(any('FLOCK'in s and 'WRITE'in s and s.split()[4]=='192464'and s.split()[5]=='00:21:13626'for s in Path('/proc/locks').read_text().splitlines()),'guardian26 kernel FLOCK absent')
 fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:
   fcntl.flock(fd,fcntl.LOCK_UN);raise ValueError('scope lock was free')
 finally:os.close(fd)
 return o
