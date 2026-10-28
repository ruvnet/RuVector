"""Root-only one canonical E59 capsule recovery staging. No Git/scientific calls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,copy,importlib.util,json
import guard as g

def canonical(row):
 g.need(type(row)is dict and set(row)=={'path','bytes','sha256'},'exact public3ref');g.relative(row['path']);g.checked(row,True);return row
def typed(row,project=False):
 g.need(type(row)is dict and set(row)=={'path','bytes','sha256'}and type(row['path'])is str and type(row['bytes'])is int and row['bytes']>=0 and type(row['sha256'])is str and g.re.fullmatch('[0-9a-f]{64}',row['sha256']),'typed3reference')
 if project:g.relative(row['path'])
 else:g.need(Path(row['path']).is_absolute(),'absolute reference')
def baseline(row):
 v=g.parse(g.checked(row));return g.unwrap(v)if'structuredContent'in v else v
def shape(x):
 keys={'epoch','phase','kind','source_checkpoint','parent','owner','source_map','root_baseline','bench_baseline','current_CP','current_CONT','independent_source_readback','scientific_source_freeze','science_manifest','scientific_plan','recovery_freeze','recovery_plan','capsule_envelope','archive_overlay','overlay_adapter','recovery_policy','transport_sources','artifacts','ACTIVE','CONT_patch','CP_final_record','CP_recovery_record','commit_message'}
 g.need(type(x)is dict and set(x)==keys,'complete exact recovery input schema');g.need(type(x['epoch'])is int and x['epoch']==59 and x['phase']=='recovery'and x['kind']==g.KIND and x['source_checkpoint']==g.BASE and x['parent']==g.SOURCE_PARENT,'recovery identity')
 for k in('owner','source_map','root_baseline','bench_baseline','current_CP','current_CONT','independent_source_readback'):typed(x[k])
 for k in('scientific_source_freeze','science_manifest','scientific_plan','recovery_freeze','recovery_plan','capsule_envelope','archive_overlay','overlay_adapter','recovery_policy'):typed(x[k],True)
 g.need(type(x['transport_sources'])is dict and set(x['transport_sources'])=={'guard','stagebuilder','controller'},'three exact transport sources')
 for row in x['transport_sources'].values():typed(row,True)
 g.need(type(x['artifacts'])is list and len(x['artifacts'])>0,'new normal recovery artifacts')
 for row in x['artifacts']:
  g.need(type(row)is dict and set(row)=={'path','bytes','sha256','local_path'}and type(row['local_path'])is str and Path(row['local_path']).is_absolute(),'artifact4row');typed({k:row[k]for k in('path','bytes','sha256')},True)
 for k in('ACTIVE','CONT_patch','CP_final_record','CP_recovery_record'):g.need(type(x[k])is dict,'control objects')
 g.need(type(x['commit_message'])is str and x['commit_message'],'commit message')
def current_recovery_rows():
 out={}
 for p in sorted((g.R/g.RECOVERY_ROOT).rglob('*')):
  g.need(not p.is_symlink(),'recovery symlink')
  if p.is_file():r=g.ref(p);r['path']=p.relative_to(g.R).as_posix();out[r['path']]=r
 return out
def layout(leaves,root,bench):
 scope=g.scope_sha(leaves);b=copy.deepcopy(bench['tree']);r=copy.deepcopy(root['tree']);g.need(sum(x['path']=='ruvector-sota-2026-10-22'for x in b)==1 and sum(x['path']=='benchmarks'for x in r)==1,'unique root/bench scoped siblings')
 next(x for x in b if x['path']=='ruvector-sota-2026-10-22')['sha']=scope;bs=g.tree_sha(b);next(x for x in r if x['path']=='benchmarks')['sha']=bs;return g.tree_sha(r),bs,scope

def preserve_controls(oldcont,oldcp,x,newrows,overlay78):
 patch=x['CONT_patch'];allowed={'epoch59_final','epoch59_publication_recovery28','rejection_ledger','latest_learning','next_research_priority','cumulative_budget_ledger','benchmark_refresh','publisher_status','exposed_development_sets'}
 g.need(set(patch)==allowed,'exact current final facts and recovery patch keys')
 cont=copy.deepcopy(oldcont);cont.update(copy.deepcopy(patch));active=copy.deepcopy(x['ACTIVE'])
 g.need(active['epoch']==59 and active['phase']=='recovery'and active['source_checkpoint']==g.BASE and active['owner_sha256']==g.OWNER['sha256']and active['kind']==g.KIND and active['work_active']is False and active['experiments_active']is False and active['OS_lock_observation_scope']=='PREPUBLICATION_SNAPSHOT_NOT_CURRENT_REMOTE_LEASE'and active['publication_is_not_scientific_authorization']is True,'honest closed ACTIVE snapshot')
 cont['active_run']=active
 for k,v in oldcont.items():
  if k not in patch and k!='active_run':g.need(g.same(cont[k],v),'old CONT field/type modified')
 for k in('goal_met','official_rank','champion','unused_confirmation_sets'):g.need(g.same(cont[k],oldcont[k]),'champion/goal/eligibility/confirmation preserved')
 before=oldcont['cumulative_budget_ledger'];after=cont['cumulative_budget_ledger'];g.need(type(before)is dict and type(after)is dict and set(after)-set(before)=={'epoch59_actual_execution'},'one truthful E59 actual budget row')
 for k,v in before.items():g.need(k in after and g.same(after[k],v),'prior budget/history fact changed')
 b=oldcont['exposed_development_sets'];a=cont['exposed_development_sets'];g.need(type(b)is type(a),'DEV history type')
 if type(b)is list:g.need(g.same(a[:len(b)],b),'old DEV prefix')
 elif type(b)is dict:
  for k,v in b.items():g.need(k in a and g.same(a[k],v),'old DEV field')
 else:g.need(g.same(a,b),'old DEV history')
 fin=cont['epoch59_final'];rec=cont['epoch59_publication_recovery28'];g.need(fin['status']=='REJECTED_FROZEN_DEV_MECHANISM_QUALITY_GATE'and fin['candidate_promoted']is False and fin['official_goal_verified']is False and fin['used_confirmation_identities']==[]and fin['all_scientific_slots_closed']is True,'fixed rejected science completion')
 g.need(rec['archive_overlay']['CP_project_files_includes_overlay78']is False and rec['archive_overlay']['project_additions_AFTER_standard_restore']==78 and rec['old_failed_blob_historical_outcome']=='UNKNOWN'and rec['old_failed_blob_current_read']=='GITHUB_NOT_FOUND_404_AT_SINGLE_OBSERVATION'and rec['old_28_blob_recreation_forbidden']is True,'truthful distinct transport and old uncertainty')
 cp=copy.deepcopy(oldcp);g.need('epoch59_final'not in cp and'epoch59_publication_recovery28'not in cp,'new CP final and recovery records only');cp['epoch59_final']=copy.deepcopy(x['CP_final_record']);cp['epoch59_publication_recovery28']=copy.deepcopy(x['CP_recovery_record'])
 cp['project_files']=sorted([*oldcp['project_files'],*newrows],key=lambda r:r['path']);g.need(len({r['path']for r in cp['project_files']})==len(cp['project_files'])and not(set(r['path']for r in cp['project_files'])&overlay78),'CP unique ordinary files exclude archiveoverlay78')
 for k,v in oldcp.items():
  if k!='project_files':g.need(g.same(cp[k],v),'old CP field/type modified')
 cont['checkpoint_manifest_sha256']=g.sha(g.encoded(cp));return cont,cp,active

def build(a,check_only=False):
 g.need(type(check_only)is bool,'explicit checkonly');g.owner_fence();raw=g.body(a.input);g.need(g.sha(raw)==a.input_sha,'phase input pinned');x=g.parse(raw);shape(x);approval=g.parse(g.body(a.approval))
 g.need(g.sha(g.body(a.approval))==a.approval_sha and approval.get('checks_passed')is True and approval.get('phase_preparation_allowed')is True and approval.get('phase_input_sha256')==a.input_sha,'independent exact recovery preparation approval')
 g.need(g.same(x['owner'],g.OWNER),'exact live approved owner28');out=Path(a.out);g.need(out==g.W/'stage-recovery'and not out.exists(),'fresh exact stage outside R')
 g.scientific_closure(x['science_manifest']);sf=g.parse(g.checked(x['scientific_source_freeze'],True));plan=g.parse(g.checked(x['scientific_plan'],True));g.need(sf['epoch']==plan['epoch']==59 and sf['source_checkpoint']==plan['source_checkpoint']==g.BASE,'unchanged scientific source identity')
 for r in sf['files']:canonical(r)
 rf=g.parse(g.checked(x['recovery_freeze'],True));rp=g.parse(g.checked(x['recovery_plan'],True));g.need(rf['epoch']==rp['epoch']==59 and rf['source_parent']==rp['source_parent']==g.SOURCE_PARENT and rf['kind']==rp['kind']==g.KIND and rf['scientific_calls_authorized']==rp['scientific_calls_authorized']==0,'new recovery source only')
 frozen={r['path']:canonical(r)for r in rf['files']};now=current_recovery_rows();g.need(len(frozen)==len(rf['files'])and set(frozen)|{x['recovery_freeze']['path']}==set(now),'all recovery files frozen except freeze self')
 for p,r in frozen.items():g.need(g.same(r,now[p]),'recovery source hash changed')
 g.need(frozen.get(g.FENCE_PUBLIC_PATH)=={'path':g.FENCE_PUBLIC_PATH,'bytes':g.FENCE_SOURCE['bytes'],'sha256':g.FENCE_SOURCE['sha256']},'owner28 fence source frozen')
 for name,row in x['transport_sources'].items():canonical(row);g.need(row==frozen.get(row['path']),'new transport source frozen '+name)
 g.need(g.sha(g.body(Path(g.__file__).resolve()))==x['transport_sources']['guard']['sha256']and g.sha(g.body(Path(__file__).resolve()))==x['transport_sources']['stagebuilder']['sha256'],'operational helpers match new source freeze')
 for k in('recovery_plan','capsule_envelope','archive_overlay','overlay_adapter','recovery_policy'):g.need(x[k]==frozen.get(x[k]['path']),'frozen recovery input '+k)
 overlay=g.canonical_capsule(x['capsule_envelope'],x['archive_overlay']);overlay78={r['path']for r in overlay['files']if not r['source_Git_present']}
 g.need(len(overlay78)==78,'exact78 archive-only files');policy=g.parse(g.checked(x['recovery_policy'],True));g.need(policy['checks_passed']is True and policy['failed_blob_create_authorized']is False and policy['failed_blob_reuse_authorized']is False and policy['read_outcome']=='GITHUB_NOT_FOUND_404_AT_SINGLE_OBSERVATION','old404notcreateorreuseauthority')
 # Pure metadata archive validation uses the exact independently reviewed adapter.
 spec=importlib.util.spec_from_file_location('pinned_e59_overlay',g.R/x['overlay_adapter']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);files,orows=mod.validate(g.checked(x['capsule_envelope'],True),g.checked(x['archive_overlay'],True))
 for p,b in files.items():g.need(g.body(g.R/p)==b,'all161 archived project bodies equal actual isolated R')
 sm=g.parse(g.checked(x['source_map']));root=baseline(x['root_baseline']);bench=baseline(x['bench_baseline']);oldgit={r['path']:r for r in sm['files']}
 g.need(sm['commit']==g.SOURCE_PARENT and root['sha']==sm['root']and root['sha']==g.tree_sha(root['tree'])and bench['sha']==g.tree_sha(bench['tree'])and not root.get('truncated')and not bench.get('truncated'),'fresh actual SOURCE root/bench')
 g.need(len(oldgit)==9315 and len(oldgit)==len(sm['files'])and all(r['mode']=='100644'and r['type']=='blob'and type(r['size'])is int and r['sha']not in g.DENY for r in oldgit.values())and g.scope_sha(sm['files'])==sm['scope_sha'],'all9315 typed complete source leaves and Merkle')
 g.need(next(r['sha']for r in root['tree']if r['path']=='benchmarks')==bench['sha']and next(r['sha']for r in bench['tree']if r['path']=='ruvector-sota-2026-10-22')==sm['scope_sha'],'root/bench/scope chain')
 ind=g.parse(g.checked(x['independent_source_readback']));g.need(ind['checks_passed']is True and ind['verified_actual_remote_readback']is True and ind['commit']==g.SOURCE_PARENT and ind['root']==sm['root']and ind['tree_sha']==sm['root'],'fresh independently read actual SOURCE')
 cpraw=g.checked(x['current_CP']);contraw=g.checked(x['current_CONT']);cp=g.parse(cpraw);cont=g.parse(contraw)
 for n,b in [('CHECKPOINT_MANIFEST.json',cpraw),('CONTINUATION.json',contraw)]:g.need(g.git(b)==oldgit[n]['sha']and len(b)==oldgit[n]['size'],'actual source control Git body')
 oldrows={r['path']:r for r in cp['project_files']};g.need(len(oldrows)==9308 and len(oldrows)==len(cp['project_files']),'all9308 ordinary CP rows')
 for p,r in oldrows.items():g.checked(r,True)
 cp_only=set(oldrows)-set(oldgit);g.need(len(cp_only)==70,'existing70 CP-only remain')
 e44=cont['epoch44_publication_recovery9']['archive_overlay'];g.need(e44['CP_project_files_includes_overlay55']is False and e44['project_additions_AFTER_standard_restore']==55 and e44['canonical_members']==137,'E44overlay55 remains separate')
 g.need(not(overlay78&set(oldgit))and not(overlay78&set(oldrows)),'overlay78 not falsely looseGit or CP')
 for p,r in orows.items():
  if r['source_Git_present']:b=files[p];g.need(p in oldgit and g.git(b)==oldgit[p]['sha']and len(b)==oldgit[p]['size'],'83actualSOURCEduplicates exact')
 payloads={};newrows=[]
 for r in x['artifacts']:
  p=g.relative(r['path']);g.need(g.new_path(p)and p not in oldgit and p not in oldrows and p not in payloads and Path(r['local_path'])==g.R/p,'new canonical recovery-only artifact')
  b=g.checked({'path':r['local_path'],'bytes':r['bytes'],'sha256':r['sha256']});g.public_text(p,b);payloads[p]=b;newrows.append({k:r[k]for k in('path','bytes','sha256')})
 g.need(set(payloads)==set(now),'exact complete new normal recovery artifact inventory')
 newcont,newcp,active=preserve_controls(cont,cp,x,newrows,overlay78);payloads.update({'CONTINUATION.json':g.encoded(newcont),'CHECKPOINT_MANIFEST.json':g.encoded(newcp),'ACTIVE_RUN.json':g.encoded(active)})
 g.need(set(r['path']for r in newcp['project_files'])-(set(oldgit)|set(payloads))==cp_only,'same70CP-only and no overlayCP additions')
 updates=[{'path':p,'mode':'100644','type':'blob','sha':g.git(b),'size':len(b)}for p,b in sorted(payloads.items())];g.need(all(r['sha']not in g.DENY and r['sha']not in g.PRIOR_ATTEMPTED_BLOB_SHAS for r in updates),'no original28 or forbiddenGit object recreation/live leaf')
 emitted={}
 def emit(p,b):
  p=Path(p);g.need(p.is_absolute()and p.is_relative_to(out)and all(not q.is_symlink()for q in(p,*p.parents)),'safe stage output');n=p.relative_to(out).as_posix();g.relative(n);g.need(n not in emitted and not p.exists(),'fresh nonrepeated stage output');emitted[n]={'file':n,'bytes':len(b),'sha256':g.sha(b)}
  if not check_only:g.write(p,b)
 if not check_only:out.mkdir(mode=0o700)
 requests=[];prefix='recovery59-g28'
 def request(n,kind,v,expected=None):
  g.validate_request(kind,v);b=g.encoded(v);g.need(len(b)<(8384512 if kind==g.TOOLS['blob'] else 175904),'tool argument transport cap');emit(out/(n+'.json'),b);r={'id':prefix+'-'+n,'kind':kind,'file':n+'.json','bytes':len(b),'sha256':g.sha(b)}
  if expected is not None:r['expected_sha']=expected
  requests.append(r);return r
 # Deduplicate body GitSHAs globally, including the identical USED summary aliases.
 groups={}
 for r in updates:groups.setdefault(r['sha'],[]).append(r)
 blobs=[]
 for i,(gs,rs)in enumerate(sorted(groups.items())):
  b=payloads[rs[0]['path']];g.need(all(payloads[r['path']]==b for r in rs),'GitSHA body collision rejected');q=request('blob-'+str(i),g.TOOLS['blob'],{'repository_full_name':g.REPO,'encoding':'utf-8','content':b.decode()},gs);blobs.append({**q,'project_paths':[r['path']for r in rs],'payload_bytes':len(b),'GitSHA':gs})
 g.need(sum(r['GitSHA']=='7be438edb75e9a09cd27b6e4e4eb81ca835bb142'for r in blobs)==1,'ONE canonical capsule request only')
 working=copy.deepcopy(oldgit);tree=sm['root'];trees=[]
 for i in range(0,len(updates),32):
  chunk=updates[i:i+32]
  for r in chunk:working[r['path']]=r
  fr,fb,fs=layout(list(working.values()),root,bench);v={'repository_full_name':g.REPO,'base_tree_sha':tree,'tree_elements':[{k:(g.SCOPE+r[k]if k=='path'else r[k])for k in('path','mode','type','sha')}for r in chunk]};trees.append(request('tree-'+str(i//32),g.TOOLS['tree'],v,fr));tree=fr
 fr,fb,fs=layout(list(working.values()),root,bench);request('commit',g.TOOLS['commit'],{'repository_full_name':g.REPO,'tree_sha':fr,'parent_sha':g.SOURCE_PARENT,'message':x['commit_message']});request('ref',g.TOOLS['ref'],{'repository_full_name':g.REPO,'branch_name':g.BRANCH,'sha':'0'*40,'force':False,'expected_sha':g.SOURCE_PARENT})
 leafraw=g.encoded({'scope':g.SCOPE,'files':sorted(working.values(),key=lambda r:r['path'])});emit(out/'expected-leaves.json',leafraw)
 m={'schema_version':1,'epoch':59,'phase':'recovery','kind':g.KIND,'source_checkpoint':g.BASE,'source_parent':g.SOURCE_PARENT,'owner':g.OWNER,'scientific_source_freeze':x['scientific_source_freeze'],'science_manifest':x['science_manifest'],'scientific_plan':x['scientific_plan'],'recovery_freeze':x['recovery_freeze'],'recovery_plan':x['recovery_plan'],'capsule_envelope':x['capsule_envelope'],'archive_overlay':x['archive_overlay'],'overlay_adapter':x['overlay_adapter'],'recovery_policy':x['recovery_policy'],'transport_sources':x['transport_sources'],'phase_input_sha256':a.input_sha,'expected_source_root':sm['root'],'expected_source_bench':bench['sha'],'expected_source_scope':sm['scope_sha'],'expected_final_root':fr,'expected_final_bench':fb,'expected_final_scope':fs,'new_Git_leaves':len(working),'CP_rows':len(newcp['project_files']),'CP_only_paths':sorted(cp_only),'E44_overlay55_remains_separate':True,'E59_overlay78_excluded_from_Git_and_CP':True,'mutation_ID_prefix':prefix,'mutation_ID_order':[r['id']for r in requests],'requests':requests,'blobs':blobs,'trees':trees,'prior_attempted_blob_SHAs':sorted(g.PRIOR_ATTEMPTED_BLOB_SHAS),'canonical_capsule_object_attempt_limit':1,'expected_leaves':{'file':'expected-leaves.json','bytes':len(leafraw),'sha256':g.sha(leafraw)},'forbidden_Git_objects':sorted(g.DENY),'original_blob5_execution':'UNKNOWN_PERMANENTLY','original_failed_blob_lookup':'GITHUB_NOT_FOUND_404_AT_SINGLE_OBSERVATION','original_failed_blob_history':'UNKNOWN','publication_only':True,'scientific_authorization_granted':False}
 emit(out/'metadata.json',g.encoded(m))
 if check_only:g.need(not out.exists(),'no checkonly stage writes');return{'metadata':m,'planned_outputs':sorted(emitted.values(),key=lambda r:r['file'])}
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--input-sha',required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();r=build(a,a.check_only)
 if a.check_only:
  m=r['metadata'];row=next(x for x in r['planned_outputs']if x['file']=='metadata.json');print(json.dumps({'check_only':True,'all_real_preparation_checks_passed':True,'actual_Git_or_SCI_calls':0,'actual_stage_or_request_or_R_writes':0,'stage_metadata_planned':row,'new_Git_leaves':m['new_Git_leaves'],'CP_rows':m['CP_rows'],'planned_calls':len(m['requests']),'expected_final_root':m['expected_final_root']}))
 else:print(json.dumps({'actual_Git_or_SCI_calls':0,'stage_metadata':g.ref(Path(a.out)/'metadata.json'),'new_Git_leaves':r['new_Git_leaves'],'CP_rows':r['CP_rows'],'planned_calls':len(r['requests']),'expected_final_root':r['expected_final_root']}))
