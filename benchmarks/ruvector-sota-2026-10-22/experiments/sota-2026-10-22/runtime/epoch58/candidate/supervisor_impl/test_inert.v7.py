"""Authored-only inert tests. Never run supervisor/worker/scorer CLI entrypoints."""
import copy
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def imported(name):
    spec=importlib.util.spec_from_file_location('e58_inert_'+name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
S=imported('scorer.v3');V=imported('supervisor.v2');O=imported('run_cli_once.v5')


def authored_forecasts():
    rows=[]
    for cid in S.IDS:
        for arm in S.ARMS:
            p=[.7,.2,.1]
            rows.append({'case_id':cid,'arm':arm,'probabilities':p,
                'log_probabilities':[math.log(x) for x in p],
                'elapsed_seconds':.125,'actual_model_forward_calls':1,
                'actual_pair_inferences':2,'context_window_ids':[0,1],
                'all_source_windows_covered':True,'early_exit':False,
                'numeric_flags':[],'prediction':'Contradiction'})
    return {'schema_version':1,'epoch':58,'kind':S.KIND,'phase':'dev',
            'plan_sha256':'0'*64,'source_freeze_sha256':'1'*64,
            'labels':list(S.LABELS),'arms':list(S.ARMS),
            'selected_query_ids':list(S.IDS),'reserved_ids':list(S.IDS),'rows':rows}


class InertScorerTests(unittest.TestCase):
    def authored_joined(self):
        return {'schema_version':1,'epoch':58,'kind':S.KIND,'phase':'dev','ids':list(S.IDS),'label_order':list(S.LABELS),
                'semantic_labels_decoded':17,'sealed_forecasts_ref':{'path':'/authored/seal','bytes':1,'sha256':'0'*64},
                'rows':[{'case_id':cid,'label':'Entailment'} for cid in S.IDS]}
    def test_custody_v5_join_value_schema(self):
        j=self.authored_joined();self.assertEqual(S.validate_joined_gold(j,j['sealed_forecasts_ref']),j['rows'])
    def test_join_wrong_ids_rejected(self):
        j=self.authored_joined();j['ids']=j['ids'][:-1]
        with self.assertRaisesRegex(ValueError,'LATE_GOLD_JOIN'):S.validate_joined_gold(j,j['sealed_forecasts_ref'])
    def test_join_event_count_requires_integer(self):
        for x in (17.0,True):
            j=self.authored_joined();j['semantic_labels_decoded']=x
            with self.assertRaisesRegex(ValueError,'LATE_GOLD_JOIN'):S.validate_joined_gold(j,j['sealed_forecasts_ref'])
    def test_join_label_order_rejected(self):
        j=self.authored_joined();j['label_order']=['Entailment','Contradiction','NotMentioned']
        with self.assertRaisesRegex(ValueError,'LATE_GOLD_JOIN'):S.validate_joined_gold(j,j['sealed_forecasts_ref'])
    def test_forward_participations_not_global_counts(self):
        f=authored_forecasts();g=[{'case_id':cid,'label':'Entailment'} for cid in S.IDS];r=S.evaluate(f,g)
        for arm in S.ARMS:
            self.assertEqual(r['arms'][arm]['case_forward_participations'],17)
            self.assertNotIn('actual_model_forward_calls',r['arms'][arm])
        self.assertTrue(r['model_forward_accounting']['participation_sums_are_not_physical_forward_counts'])
        self.assertEqual(r['model_forward_accounting']['global_physical_forward_source'],'raw_worker.counters.NLI_forward_calls_completed')
    def test_authored_proper_scores(self):
        f=authored_forecasts();g=[{'case_id':cid,'label':'Entailment'} for cid in S.IDS]
        r=S.evaluate(f,g)
        self.assertEqual(r['actual_scored_forecasts'],51)
        self.assertEqual(r['actual_scored_case_ids'],list(S.IDS))
        self.assertFalse(r['mechanism_quality_gate']['quality_checks_passed'])
        self.assertAlmostEqual(r['arms']['native_order']['Brier'],1.14)
        self.assertAlmostEqual(r['arms']['native_order']['NLL'],-math.log(.2))
        self.assertEqual(r['uncertainty']['document_clusters'],1)
        self.assertIsNone(r['uncertainty']['independent_document_CI'])
        self.assertFalse(r['candidate_promotion_allowed'])
    def test_log_probability_mismatch_rejected(self):
        f=authored_forecasts();f['rows'][0]['log_probabilities'][0]=-.9
        with self.assertRaisesRegex(ValueError,'SOFTMAX'):S.validate_forecasts(f)
    def test_missing_row_rejected(self):
        f=authored_forecasts();f['rows'].pop()
        with self.assertRaisesRegex(ValueError,'ALL51'):S.validate_forecasts(f)
    def test_wrong_E_C_order_rejected(self):
        f=authored_forecasts();f['labels']=['Entailment','Contradiction','NotMentioned']
        with self.assertRaisesRegex(ValueError,'SCHEMA'):S.validate_forecasts(f)
    def test_NEI_early_exit_rejected(self):
        f=authored_forecasts();r=f['rows'][0];r.update(probabilities=[.1,.2,.7],
            log_probabilities=[math.log(.1),math.log(.2),math.log(.7)],prediction='NotMentioned',
            early_exit=True,all_source_windows_covered=False,context_window_ids=[0],actual_pair_inferences=1)
        with self.assertRaisesRegex(ValueError,'NO_NEI_EARLY_EXIT'):S.validate_forecasts(f)
    def test_numeric_flags_only_disable_exit(self):
        f=authored_forecasts();f['rows'][0]['numeric_flags']=['claim_has_numerals']
        S.validate_forecasts(f)
        f['rows'][0]['early_exit']=True
        with self.assertRaisesRegex(ValueError,'NUMERICAL_FLAGS'):S.validate_forecasts(f)
    def test_numeric_flag_order_exact(self):
        f=authored_forecasts();f['rows'][0]['numeric_flags']=['claim_numeral_not_in_supplied_context']
        with self.assertRaisesRegex(ValueError,'NUMERIC_DIAGNOSTICS'):S.validate_forecasts(f)
    def test_positive_softmax_no_epsilon(self):
        f=authored_forecasts();f['rows'][0]['probabilities']=[1,0,0]
        with self.assertRaisesRegex(ValueError,'SOFTMAX'):S.validate_forecasts(f)
    def test_window_coverage_flag_not_trusted(self):
        f=authored_forecasts();r=f['rows'][0];r.update(context_window_ids=[0],actual_pair_inferences=1)
        with self.assertRaisesRegex(ValueError,'COVERAGE_FLAGS'):S.validate_forecasts(f)
    def test_source_order_is_prefix(self):
        f=authored_forecasts();r=f['rows'][1];r['context_window_ids']=[1,0]
        with self.assertRaisesRegex(ValueError,'SOURCE_ARM'):S.validate_forecasts(f)
    def test_duplicated_or_exposed_gold_ids_rejected(self):
        f=authored_forecasts();g=[{'case_id':cid,'label':'Entailment'} for cid in S.IDS]
        g[-1]=g[0]
        with self.assertRaisesRegex(ValueError,'GOLD_ROWS'):S.evaluate(f,g)
    def test_case_major_order_required(self):
        f=authored_forecasts();f['rows'][0],f['rows'][1]=f['rows'][1],f['rows'][0]
        with self.assertRaisesRegex(ValueError,'CASE_MAJOR'):S.validate_forecasts(f)


class InertPopenClosureTests(unittest.TestCase):
    class FakePopen:
        def __init__(self,done=False,timeout=False):self.returncode=0 if done else None;self.calls=[];self.timeout=timeout
        def poll(self):return self.returncode
        def terminate(self):self.calls.append('terminate')
        def kill(self):self.calls.append('kill');self.returncode=-9
        def wait(self,timeout):
            self.calls.append('wait')
            if self.timeout and self.returncode is None:raise O.subprocess.TimeoutExpired('authored',timeout)
            self.returncode=0 if self.returncode is None else self.returncode;return self.returncode
    def test_hard_exit_even_when_proc_observation_throws(self):
        class BrokenModule:
            def same_identity(self,*args):raise OSError('authored proc IO fault')
        p=self.FakePopen();exits=[]
        O.hard_kill_owned(p,BrokenModule(),{'pid':123,'startticks':456,'pgid':123,'sid':123},True,exits.append)
        self.assertEqual(p.calls,['kill']);self.assertEqual(exits,[124])
    def test_cleanup_observation_failure_retained(self):
        errors=[]
        def fail():raise OSError('authored proc IO fault')
        self.assertFalse(V.best_effort_observe(fail,errors));self.assertEqual(errors[0]['code'],'PROCESS_OBSERVATION_FAILED_DURING_CLEANUP')
    def test_owned_popen_finally_closes_after_observation_failure(self):
        errors=[];p=self.FakePopen(timeout=True)
        def fail():raise OSError('authored journal fault')
        V.best_effort_observe(fail,errors)
        self.assertTrue(V.close_owned_popen(p,O.time.monotonic()+2,errors));self.assertIn('kill',p.calls)
    def test_uncertain_direct_child_already_reaped(self):
        p=self.FakePopen(done=True);self.assertTrue(O.close_uncertain_direct_child(p,O.time.monotonic()+2));self.assertEqual(p.calls,[])
    def test_uncertain_direct_child_terminates_and_reaps(self):
        p=self.FakePopen();self.assertTrue(O.close_uncertain_direct_child(p,O.time.monotonic()+2));self.assertEqual(p.calls,['terminate','wait'])
    def test_uncertain_direct_child_kills_after_timeout(self):
        p=self.FakePopen(timeout=True);self.assertTrue(O.close_uncertain_direct_child(p,O.time.monotonic()+2));self.assertEqual(p.calls,['terminate','wait','kill','wait'])

class InertIOTests(unittest.TestCase):
    def test_strict_duplicate_and_nonfinite_json(self):
        for m in (S,V,O):
            with self.assertRaises(ValueError):m.parse(b'{"x":1,"x":2}')
            with self.assertRaises(ValueError):m.parse(b'{"x":NaN}')
    def test_stable_refs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='e58-authored-inert-') as directory:
            for i,m in enumerate((S,V,O)):
                p=Path(directory)/('authored-'+str(i)+'.json')
                ref=m.put(p,{'authored':True});self.assertEqual(m.read_ref(ref),p.read_bytes())
                with self.assertRaises(FileExistsError):m.put(p,{'authored':False})
                ref['sha256']='0'*64
                with self.assertRaises(ValueError):m.read_ref(ref)
    def test_symlink_data_ref_rejected(self):
        with tempfile.TemporaryDirectory(prefix='e58-authored-inert-') as directory:
            p=Path(directory)/'authored';p.write_bytes(b'authored')
            link=Path(directory)/'link';link.symlink_to(p)
            for m in (S,V,O):
                with self.assertRaises(ValueError):m.body(link)
    def test_process_identity_only_self_read(self):
        identity=V.process_identity(os.getpid())
        self.assertEqual(identity['pid'],os.getpid())
        self.assertTrue(V.same_identity(identity,True))
        changed=dict(identity,startticks=identity['startticks']+1)
        self.assertFalse(V.same_identity(changed,True))
    def test_pinned_venv_interpreter_leaf(self):
        real=Path(sys.executable).resolve();data=real.read_bytes()
        with tempfile.TemporaryDirectory(prefix='e58-authored-inert-') as directory:
            p=Path(directory)/'python';p.symlink_to(real)
            rr={'path':str(real),'bytes':len(data),'sha256':V.sha(data)}
            leaf=dict(rr,path=str(p))
            self.assertEqual(V.python_executable(leaf,rr),str(p))
            leaf['sha256']='0'*64
            with self.assertRaises(ValueError):V.python_executable(leaf,rr)


if __name__=='__main__':unittest.main(verbosity=2)
