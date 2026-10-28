"""Exact generation28 recovery owner fence. It grants no scientific authority."""
from pathlib import Path
import fcntl,hashlib,json,os,re,stat,time
OWNER_PATH='/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028/guardian28/owner.json'
RUN='recovery59-993dce6842bf-20261028-generation28'
BOOT='4632fb06-b0ef-4df5-b72b-0e875bbad810'
NAMESPACE='pid:[4026532200]'
def need(ok,message):
 if not ok:raise ValueError(message)
def unique(pairs):
 out={}
 for k,v in pairs:
  need(k not in out,'duplicate owner JSON key');out[k]=v
 return out
def verify_owner(expected_owner):
 need(type(expected_owner)is dict and set(expected_owner)=={'path','bytes','sha256'} and type(expected_owner['path'])is str and type(expected_owner['bytes'])is int and type(expected_owner['sha256'])is str,'exact typed owner28 reference required')
 need(expected_owner['path']==OWNER_PATH and 100<=expected_owner['bytes']<=4096 and re.fullmatch('[0-9a-f]{64}',expected_owner['sha256'])is not None,'fixed owner28 path and bounded hash reference')
 need(os.sched_getaffinity(0)=={0,1},'caller CPU0,1 required')
 p=Path(OWNER_PATH);need(not any(x.is_symlink()for x in[p,*p.parents]),'owner path symlink')
 before=p.stat();need(stat.S_ISREG(before.st_mode),'regular owner file');raw=p.read_bytes();after=p.stat()
 sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(sig(before)==sig(after)and len(raw)==expected_owner['bytes']and hashlib.sha256(raw).hexdigest()==expected_owner['sha256'],'stable exact approved owner28 bytes/hash')
 o=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite owner JSON')))
 need(type(o['epoch'])is int and type(o['generation'])is int and o['epoch']==59 and o['generation']==28 and type(o['pid'])is int and type(o['startticks'])is int and o['pid']>1 and o['startticks']>0,'generation28 typed identity')
 pid=o['pid'];ticks=o['startticks']
 need(o['run_id']==RUN and o['scope']=='ruvnet/RuVector:benchmarks/ruvector-sota-2026-10-22','exact run/scope')
 need(o['purpose']=='E59_PUBLICATION_RECOVERY_ONLY_ZERO_SCIENTIFIC_AUTHORITY'and o['scientific_invocation_authorized_by_lock_alone']is False,'publication recovery fence only')
 need(o['launch_mode']=='DETACHED_BOUNDED_SESSION'and o['session_id']==o['process_group']==pid,'detached owner context')
 need(type(o['started_unix'])is float and type(o['expires_unix'])is float and o['expires_unix']-o['started_unix']==7200 and time.time()<o['expires_unix']and not os.path.lexists(p.parent/'release.request')and not os.path.lexists(p.parent/'released.json'),'owner expired/releasing or wrong bounded lifetime')
 need(o['kernel_boot_id']==BOOT==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and o['PID_namespace']==NAMESPACE==os.readlink('/proc/self/ns/pid'),'boot/namespace differs')
 fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
 need(int(fields[19])==ticks and fields[0]not in('Z','X')and int(fields[1])==1 and int(fields[2])==int(fields[3])==pid,'live detached original guardian28 birth')
 need(os.sched_getaffinity(pid)=={0,1},'guardian CPU0,1')
 lock=Path('/tmp/ruvector-sota-2026-10-22/exclusive.work.lock')
 need(o['lock']==str(lock)and o['lock_device']==33 and o['lock_inode']==13626 and o['lock_fd']==3,'original lock declaration')
 a=Path('/proc/'+str(pid)+'/fd/3').stat();b=lock.stat();need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(33,13626),'original inode/FD3')
 need(any('FLOCK'in s and 'WRITE'in s and s.split()[4]==str(pid)and s.split()[5]=='00:21:13626'for s in Path('/proc/locks').read_text().splitlines()),'guardian28 kernel FLOCK absent')
 fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:
   fcntl.flock(fd,fcntl.LOCK_UN);raise ValueError('scope lock was free')
 finally:os.close(fd)
 return o
