"""Distinct reviewed streaming new-Kit cache retirement; no source restoration retry.

Preserves the failed64MiB wrapper and all eleven source-step receipts. Only new
Kit files with exact distinct-inode freshR twins are unlinked. No global Path
collection/sort, original donor mutation, network or scientific entrypoint.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,stat,sys,time
D=Path('/dev/shm/ruvector-oct28-data')
W=Path('/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028')
K=W/'kit';R=W/'project';LIMIT=64*1024*1024

def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(p):
 p=Path(p);need(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in[p,*p.parents]),'canonical nonsymlink path');return p
def digest_file(p):
 p=canonical(p);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256()
 with os.fdopen(fd,'rb')as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode),'regular file')
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
  z=os.fstat(f.fileno())
 need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'stable file identity during streaming hash')
 return a,h.hexdigest()
def ref(p):
 a,h=digest_file(p);return {'path':str(p),'bytes':a.st_size,'sha256':h}
def checked(row):
 need(type(row)is dict and set(row)=={'path','bytes','sha256'}and type(row['bytes'])is int,'typed exact reference');p=Path(row['path']);need(ref(p)==row,'exact reference body');return p
def load(p):return json.loads(canonical(p).read_bytes())
def put(p,x):
 b=(json.dumps(x,sort_keys=True,indent=2)+'\n').encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
def headroom():return int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--config-sha',required=True);a.add_argument('--approval',required=True);a.add_argument('--approval-sha',required=True);a.add_argument('--root-admission',required=True);a.add_argument('--root-admission-sha',required=True);args=a.parse_args()
 need(not sys.flags.optimize and os.sched_getaffinity(0)=={0,1},'unoptimized CPU0,1 only');resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT));initial=headroom()
 cpath=Path(args.config);qpath=Path(args.approval);rpath=Path(args.root_admission);need(ref(cpath)['sha256']==args.config_sha and ref(qpath)['sha256']==args.approval_sha and ref(rpath)['sha256']==args.root_admission_sha,'exact configuration and admissions')
 c=load(cpath);q=load(qpath);root=load(rpath)
 need(c['kind']=='DISTINCT_STREAMED_VERIFIED_NEW_KIT_TWIN_RETIREMENT'and c['epoch']==59 and c['recovery_generation']==28 and c['Kit']==str(K)and c['R']==str(R)and c['AS_limit_bytes']==LIMIT and c['max_Kit_files']==9440,'frozen new streamed algorithm scope')
 need(q['checks_passed']is True and q['new_streamed_Kit_retirement_allowed']is True and q['configuration_ref']==ref(cpath)and q['code_ref']==c['code_ref'],'independent new streamed algorithm approval')
 need(root['new_streamed_Kit_retirement_allowed']is True and root['configuration_ref']==ref(cpath)and root['code_ref']==c['code_ref']and root['independent_approval_ref']==ref(qpath),'separate Root actual one-attempt admission')
 need(checked(c['code_ref'])==Path(__file__).resolve(),'exact executing cleanup source');failure=load(checked(c['prior_wrapper_failure']));need(failure['actual_full_CLI_exit_code']==1 and failure['Kit_unlinks_from_this_failed_retirement_attempt']==0,'retain actual prior wrapper failure')
 proof=load(checked(c['independent_complete_source_proof']));need(proof['checks_passed']is True and proof['source_component_reconstruction_complete_independently_verified']is True and proof['original_wrapper_exit_code']==1 and proof['no_missing_newKit_file_and_no_retirements_began']is True and proof['current_verified_distinct_inode_new_Kit_R_twins']==c['expected_retired_files']==9394 and proof['current_verified_twin_bytes']==c['expected_retired_bytes']==573734626,'independent complete fresh source proof despite optional cleanup failure')
 fp=checked(c['fence_source']);s=importlib.util.spec_from_file_location('streamed_new_Kit_owner_fence',fp);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 def fence():m.verify_owner(c['owner'])
 fence();need(initial>=LIMIT+512*1024*1024+16*1024*1024,'same64MiB AS plus512MiB reserve and16MiB journal allowance');need(canonical(K).is_dir()and canonical(R).is_dir()and K.parent==R.parent==W and K!=R,'distinct fresh Kit/R roots')
 attempt=W/'STREAMED_NEW_KIT_TWIN_RETIREMENT_ATTEMPT.actual.v1.json';journal=W/'STREAMED_NEW_KIT_TWIN_RETIREMENT.journal.actual.v1.jsonl';output=W/'STREAMED_NEW_KIT_TWIN_RETIREMENT.actual.v1.json';need(not output.exists(),'no repeated streamed cleanup')
 put(attempt,{'epoch':59,'recovery_generation':28,'kind':c['kind'],'configuration_ref':ref(cpath),'independent_approval_ref':ref(qpath),'Root_admission_ref':ref(rpath),'actual_AS_limit':list(resource.getrlimit(resource.RLIMIT_AS)),'headroom_before':initial,'old_helper_retry':False,'source_or_scientific_dispatch':False})
 fd=os.open(journal,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);seen=retired=retired_bytes=0;retained=[];started=time.monotonic()
 with os.fdopen(fd,'wb')as log:
  def row(x):log.write((json.dumps(x,sort_keys=True)+'\n').encode());log.flush();os.fsync(log.fileno())
  for parent,dirs,files in os.walk(K,topdown=True,followlinks=False):
   need(all(not(Path(parent)/name).is_symlink()for name in dirs),'no symlink directories')
   for name in files:
    if seen%64==0:fence()
    p=canonical(Path(parent)/name);need(p.is_relative_to(K),'new Kit only');rel=p.relative_to(K).as_posix();seen+=1;need(seen<=9440,'bounded expected Kit count');ka,kh=digest_file(p);t=R/rel
    if t.exists():
     ra,rh=digest_file(t);need((ka.st_size,kh)==(ra.st_size,rh)and(ka.st_dev,ka.st_ino)!=(ra.st_dev,ra.st_ino),'byte-identical independent freshR twin');now=p.stat();rnow=t.stat();need((now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)==(ka.st_dev,ka.st_ino,ka.st_size,ka.st_mtime_ns,ka.st_ctime_ns)and(rnow.st_dev,rnow.st_ino,rnow.st_size,rnow.st_mtime_ns,rnow.st_ctime_ns)==(ra.st_dev,ra.st_ino,ra.st_size,ra.st_mtime_ns,ra.st_ctime_ns),'unchanged exact twins before unlink')
     row({'phase':'verified_unlink_intent','path':rel,'bytes':ka.st_size,'sha256':kh,'Kit_inode':[ka.st_dev,ka.st_ino],'R_inode':[ra.st_dev,ra.st_ino]});p.unlink();row({'phase':'unlink_completed','path':rel,'bytes':ka.st_size,'sha256':kh});retired+=1;retired_bytes+=ka.st_size
    else:
     need(not t.is_symlink(),'no dangling R symlink');retained.append({'path':rel,'bytes':ka.st_size,'sha256':kh});need(len(retained)<=128,'bounded retained controls only');row({'phase':'non_twin_retained','path':rel,'bytes':ka.st_size,'sha256':kh})
 fence();need(seen==9440 and retired+len(retained)==9440 and retired==c['expected_retired_files'] and retired_bytes==c['expected_retired_bytes'] and len(retained)==46,'every exact independently reviewed new Kit file disposition');remaining=0
 for parent,dirs,files in os.walk(K,followlinks=False):
  for name in files:remaining+=1
 need(remaining==len(retained),'exact retained Kit file count')
 summary={'schema_version':1,'epoch':59,'recovery_generation':28,'kind':c['kind'],'checks_passed':True,'actual_new_streamed_cleanup_CLI_invocations':1,'actual_source_step_reruns':0,'old_failed_wrapper_exit_code':1,'old_wrapper_failure_ref':c['prior_wrapper_failure'],'independent_complete_source_proof':c['independent_complete_source_proof'],'actual_AS_limit':list(resource.getrlimit(resource.RLIMIT_AS)),'actual_maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'original_Kit_files_processed':seen,'retired_files':retired,'retired_bytes':retired_bytes,'retained_files':len(retained),'retained_entries':retained,'journal_ref':ref(journal),'attempt_ref':ref(attempt),'all_unlinked_files_had_exact_distinct_inode_R_twins':True,'original_donor_or_R_mutations':0,'source_scientific_model_native_Gold_network_or_Git_calls':0,'headroom_before':initial,'headroom_after':headroom(),'elapsed_seconds':time.monotonic()-started}
 print(json.dumps(put(output,summary)))
if __name__=='__main__':main()
