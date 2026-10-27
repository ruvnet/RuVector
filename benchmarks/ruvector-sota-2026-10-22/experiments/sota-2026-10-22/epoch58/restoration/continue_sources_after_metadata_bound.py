"""Continue only the known-unfinished source-only steps; never replay cache copies or E42 writes."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
ADAPTER_SHA = '8e85377429c3455a8c774e60984716724d2050c9255aaa0e3cf546cbbaa39114'
D=Path('/dev/shm/ruvector-e58-data');W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');K=W/'kit';R=W/'project';B=W/'base-cache'
def main():
 assert not sys.flags.optimize and os.sched_getaffinity(0)=={0,1}
 p=D/'reconstruct_sources.proposed.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e6632280704fe3d40efc6b3dfee3137370dcddb32a0837cbe7d41f1904e5bb7a'
 spec=importlib.util.spec_from_file_location('reviewed_e58_source_only_primitives',p);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);g.fence()
 pins=[('E42-archive-materialized.json',1863,'5632c24f79832ca8012c2006132c84de0930b6e1d33fa0a21fe49743ee69f9a6',0),('legacy22-verified.json',1836,'615f784f8b23bf915e532540d63a302ddb01e08c903b1d41ccb08daef28402ac',1)]
 for name,size,sha,code in pins:
  p=W/name;assert p.stat().st_size==size and g.sha(p)==sha and json.loads(p.read_text())['exit_code']==code
 assert 'File exceeds bounded read'in json.loads((W/'legacy22-verified.json').read_text())['stderr']
 assert g.sha(D/'actual-git-map.json')=='554e42b12b10c0142f19f526bd5846fe14619ac2de5fce14297116e3722d0636'
 M=json.loads((D/'actual-git-map.json').read_text());assert M['commit']=='db7b9b21f4a13e5fc3444a8fe40de26583558e2e'and len(M['files'])==9055
 assert g.sha(D/'current/CHECKPOINT_MANIFEST.json')=='ae9b6df276bbd2b3446fd21b4a3a6b2a8e3e2b96083716a3f1478318c4144d8f';C=json.loads((D/'current/CHECKPOINT_MANIFEST.json').read_text());assert len(C['project_files'])==9048
 for i,row in enumerate(M['files']):
  if i%128==0:g.fence()
  p=K/row['path'];assert p.stat().st_size==row['size']and g.sha(p,True)==row['sha']
 for row in C['project_files']:
  p=K/row['path'];assert p.stat().st_size==row['bytes']and g.sha(p)==row['sha256']
 base={r['path']:r for r in json.loads((K/'BASE_PUBLIC_MANIFEST.json').read_text())['files']};downloads=json.loads((K/'BASE_DOWNLOADS.json').read_text())['paths'];assert len(downloads)==48
 for name in downloads:assert(B/name).stat().st_size==base[name]['bytes']and g.sha(B/name)==base[name]['sha256']
 py=[sys.executable,'-I','-B'];e42=K/'recoveries/epoch42-publication-20261026';e44=K/'recoveries/epoch44-publication-20261026-g9';e49=R/'recoveries/epoch49-stop-publication-20261027-g15'
 steps=[('legacy22-verified-cap4m.json',py+[str(D/'legacy22_cp_bound_adapter.py')]),
 ('legacy26-verified.json',py+[str(K/'experiments/sota-2026-10-22/epoch26/publication/restore_epoch26_archives_v1.py'),'--kit-root',str(K),'--manifest','experiments/sota-2026-10-22/epoch26/publication/archive-transport.v1.json','--expected-manifest-sha256','9b79fc0119e1e5d15470201b7180f4b39e55b687a207dac1ff9f333d91a8ef24','--verify-only']),
 ('bootstrap-source-verified.json',py+[str(K/'bootstrap.py'),'--dest',str(R),'--base-local-directory',str(B)]),
 ('checkpoint-restored.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B)]),
 ('checkpoint-verify-only.json',py+[str(K/'restore_checkpoint.py'),'--dest',str(R),'--base-local-directory',str(B),'--verify-only']),
 ('E42-archive-verify-only.json',py+[str(e42/'restore_epoch42_archive.py'),'--kit-root',str(K),'--dest',str(R),'--restore-scope',str(W),'--archive','recoveries/epoch42-publication-20261026/immutable-e42-artifacts.zip','--projection','recoveries/epoch42-publication-20261026/ARCHIVE_PROJECTED_FILES.v1.json','--verify-only'])]
 cmd44=py+[str(e44/'restore_epoch44_overlay.py'),'--restore-scope',str(W),'--kit',str(K),'--dest',str(R),'--envelope',str(e44/'PUBLIC_ARTIFACTS.envelope.v1.json'),'--expected-envelope-sha','fc0f741eb7bd840a941e9ef3505b6924ab9be7842cd561722d4236bd98cf1d1d','--overlay',str(e44/'ARCHIVE_OVERLAY.v1.json'),'--expected-overlay-sha','815aaa11a602de738b1b62b4f9cd057639b8201b59007b3633588aaa674ba01d']
 cmd49=py+[str(e49/'restore_e49_overlay.py'),'--envelope',str(e49/'PUBLIC_RECOVERY.envelope.v1.json'),'--manifest',str(e49/'ARCHIVE_OVERLAY.v1.json'),'--project',str(R)]
 steps += [('E44-overlay-restored.json',cmd44),('E44-overlay-verified.json',cmd44+['--verify-only']),('E49-overlay-restored.json',cmd49),('E49-overlay-verified.json',cmd49+['--verify-only'])]
 assert all(not(W/name).exists()for name,_ in steps)
 index={r['path']:r for r in M['files']}
 for name,argv in steps:
  g.fence();script=Path(argv[3])
  if script==D/'legacy22_cp_bound_adapter.py':assert g.sha(script)==ADAPTER_SHA
  else:
   relative=str(script.relative_to(K if script.is_relative_to(K)else R));assert g.sha(script,True)==index[relative]['sha']
  assert not any(x in argv for x in['--install-npm','--install-python','--acquire-models','--verify-models'])
  t=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True,timeout=60);g.write(W/name,{'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-t,'model_native_Gold_Git_calls':0});assert r.returncode==0,(name,r.stderr);print(name,'PASS',flush=True)
 for root in[K,R]:
  for i,row in enumerate(C['project_files']):
   if i%128==0:g.fence()
   p=root/row['path'];assert p.stat().st_size==row['bytes']and g.sha(p)==row['sha256']
 for row in M['files']:assert(K/row['path']).stat().st_size==row['size']and g.sha(K/row['path'],True)==row['sha']
 g.fence();out={'schema_version':1,'epoch':58,'checks_passed':True,'actual_commit':M['commit'],'CP_rows_verified_per_kit_and_project':9048,'Git_leaves_verified_in_kit':9055,'CP_only_source_files':70,'base_files':48,'completed_E42_write_step_not_replayed':True,'failed_original_legacy22_receipt_preserved':True,'only_runtime_metadata_read_bound_override':{'old':2097152,'new':4194304},'original_Git_helpers_unmodified':True,'source_only_steps':[{'path':n,'bytes':(W/n).stat().st_size,'sha256':g.sha(W/n)}for n in['E42-archive-materialized.json']+[n for n,_ in steps]],'physical_distinct_copies_no_hardlinks':True,'counts':{p.name:sum(f.is_file()for f in p.rglob('*'))for p in[K,R,B]},'model_native_Gold_Git_calls':0}
 g.write(W/'SOURCE_RESTORATION_COMPLETE.v1.json',out);print(json.dumps(out))
if __name__=='__main__':main()
