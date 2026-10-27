"""Cross-schema metadata tests with authored probabilities, never dataset labels.

Load the reviewed C scorer's pure evaluate() and exact counter schema. Every
guard, count, forecast and Gold row below is an INERT fixture in a temporary
project; no real NDA text, model, native corpus, network or Gold is accessed.
"""
from pathlib import Path
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import io
import math
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCORER = None
COUNTER_SCHEMA = None
REPORTER_PATH = None

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class ReportingSchemas(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e58-reporting-INERT-')
        self.m = load_module('e58_inert_reporter_' + str(id(self)), REPORTER_PATH)
        self.m.W = Path(self.temp.name); self.m.R = self.m.W / 'project'; self.m.R.mkdir()
        self.ids = list(SCORER.IDS); self.source_sha = 'a' * 64
        self.counts = {key: 0 for key in self.m.COUNTER_KEYS}
        self.cfg = self.prepare()

    def tearDown(self): self.temp.cleanup()

    def fixture(self, path, value, project=True, raw=False):
        p = self.m.R / path if project else self.m.W / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if raw else self.m.encoded(value))
        return self.m.ref(p, project)

    def read(self, row, project=True): return self.m.checked(row, project)

    def prepare(self):
        m = self.m; P = m.P; A = m.A
        plan = self.fixture(P + 'plan.json', {'epoch': 58, 'kind': m.KIND, 'source_checkpoint': m.BASE, 'fixture': 'INERT'})
        engplan = self.fixture(P + 'engineering/plan.json', {'epoch': 58, 'phase': 'engineering', 'fixture': 'INERT'})
        devplan = self.fixture(P + 'dev/plan.json', {'epoch': 58, 'phase': 'dev', 'fixture': 'INERT'})
        code = b"def owner_fence(): return {'INERT_TEST_ONLY': True}\n"
        op_guard = self.fixture('publication-helpers-g26/guard.py', code, False, True)
        public_guard = self.fixture(P + 'publication/guard.py', code, True, True)
        reporter = self.fixture(P + 'reporting/report_actual.py', (REPORTER_PATH).read_bytes(), True, True)
        sf = self.fixture(P + 'SOURCE_FREEZE.json', {'epoch': 58, 'kind': m.KIND, 'source_checkpoint': m.BASE,
                          'files': [plan, engplan, devplan, public_guard, reporter]})
        self.source_sha = sf['sha256']
        source = self.fixture(P + 'publication/publication-source-result.json', {
            'epoch': 58, 'phase': 'source', 'parent': m.BASE, 'commit': '1' * 40,
            'verified_actual_full_tree': True, 'source_plan_sha256': plan['sha256'], 'source_freeze_sha256': sf['sha256']})
        initial = self.fixture('initial_CONT.json', {'epoch57_final': {'used_identities': {'INERT_historical_ref': True},
                               'used_confirmation': {'INERT_historical_ref': True}}}, False)
        ledger = self.fixture('experiments/sota-2026-10-22/epoch57/REJECTION_LEDGER.json', {'rejections': []})
        raw = {'epoch': 58, 'kind': m.KIND, 'phase': 'engineering', 'plan_sha256': engplan['sha256'],
               'source_freeze_sha256': sf['sha256'], 'status': 'INERT_ONLY', 'errors': [], 'rows': [],
               'counters': copy.deepcopy(self.counts), 'selected_query_ids': ['authored:INERT'], 'reserved_ids': ['authored:INERT']}
        raw_ref = self.fixture(P + 'runs/engineering/raw.json', raw)
        audit = {'epoch': 58, 'independent_review_complete': True, 'all_owned_processes_closed': True,
                 'actual_results_ref': raw_ref, 'actual_execution_independently_derived': copy.deepcopy(self.counts),
                 'engineering_qualified': False, 'throughput_gate_passed': False,
                 'status': 'INERT_ENGINEERING_REJECTED', 'reason': 'Authored schema fixture only.',
                 'native_counter_accounting': {'INERT': True}, 'latency_and_resources': {'INERT': True},
                 'throughput_measurements': {'INERT': True}}
        audit_ref = self.fixture(A + 'engineering_actual.json', audit)
        data = self.fixture(P + 'data/DATA_REPORT.json', {
            'epoch': 58, 'kind': m.KIND, 'source_checkpoint': m.BASE, 'identity_namespace': m.NAMESPACE,
            'semantic_gold_decodes': 0, 'used_confirmation_identities': [], 'actual_feature_case_ids': self.ids,
            'source_text_exposed_case_ids': self.ids, 'actual_hypothesis_exposed_case_ids': self.ids,
            'reserved_development_case_ids': self.ids, 'reserved_role': 'DEV_FOREVER', 'INERT_FIXTURE_ONLY': True})
        origin = self.fixture('private/INERT_actual_DATA_report.json', (self.m.R / data['path']).read_bytes(), False, True)
        data_review = self.fixture(A + 'data_actual.json', {'checks_passed': True, 'actual_data_report_ref': origin, 'safe_public_projection_ref': data})
        no = self.fixture(A + 'DEV_nonexecution.json', {'epoch': 58, 'independent_review_complete': True,
            'all_owned_processes_closed': True, 'DEV_CLI_invocations': 0, 'Gold_scoring_calls': 0,
            'semantic_gold_decodes': 0, 'actual_execution_independently_derived': copy.deepcopy(self.counts)})
        return {'epoch': 58, 'kind': m.KIND, 'source_checkpoint': m.BASE, 'source_freeze': sf,
                'reporter_source': reporter, 'publication_guard': op_guard, 'plan': plan,
                'engineering_plan': engplan, 'DEV_plan': devplan, 'initial_CONT': initial,
                'source_receipt': source, 'initial_rejection_ledger': ledger,
                'engineering_result': raw_ref, 'engineering_audit': audit_ref,
                'data_preparation_result': data, 'data_preparation_review': data_review,
                'DEV_pilot': None, 'DEV_nonexecution_review': no,
                'qualification': {'INERT_ONLY': True}, 'official_refresh': {'official_rank': None},
                'pre_science_preparation': {'INERT_ONLY': True}}

    def activate(self): self.m.activate(self.cfg)

    def add_dev(self, with_score=True):
        m = self.m; P = m.P; A = m.A
        rows = []
        for case in self.ids:
            for arm in SCORER.ARMS:
                p = [0.2, 0.7, 0.1]
                rows.append({'case_id': case, 'arm': arm, 'probabilities': p,
                    'log_probabilities': [math.log(x) for x in p], 'elapsed_seconds': 0.01,
                    'actual_model_forward_calls': 1, 'actual_pair_inferences': 1,
                    'context_window_ids': [0], 'all_source_windows_covered': True,
                    'early_exit': False, 'numeric_flags': [], 'prediction': 'Entailment'})
        seal = {'schema_version': 1, 'epoch': 58, 'kind': m.KIND, 'phase': 'dev',
                'plan_sha256': self.cfg['DEV_plan']['sha256'], 'source_freeze_sha256': self.source_sha,
                'labels': list(SCORER.LABELS), 'arms': list(SCORER.ARMS), 'selected_query_ids': self.ids,
                'reserved_ids': self.ids, 'rows': rows}
        seal_ref = self.fixture('private/INERT_sealed_forecasts.json', seal, False)
        # These labels are author-chosen toy labels, not labels read from ContractNLI.
        result = SCORER.evaluate(seal, [{'case_id': case, 'label': 'Entailment'} for case in self.ids])
        result['sealed_forecasts_ref'] = seal_ref
        score_ref = self.fixture(P + 'runs/dev/scored.json', result) if with_score else None
        counts = copy.deepcopy(self.counts); counts['forecasts_completed'] = len(rows)
        counts['NLI_forward_calls_attempted'] = counts['NLI_forward_calls_completed'] = len(rows)
        counts['NLI_pair_inferences_attempted'] = counts['NLI_pair_inferences_completed'] = len(rows)
        raw = {'epoch': 58, 'kind': m.KIND, 'phase': 'dev', 'plan_sha256': self.cfg['DEV_plan']['sha256'],
               'source_freeze_sha256': self.source_sha, 'status': 'INERT_ONLY', 'errors': [], 'rows': rows,
               'counters': counts, 'selected_query_ids': self.ids, 'reserved_ids': self.ids}
        raw_ref = self.fixture(P + 'runs/dev/raw.json', raw)
        gold = self.fixture(P + 'data/GOLD_REPORT.json', {'epoch': 58, 'kind': m.KIND,
            'actual_label_decoded_case_ids': self.ids, 'used_confirmation_identities': [], 'INERT_FIXTURE_ONLY': True})
        audit = {'epoch': 58, 'independent_review_complete': True, 'all_owned_processes_closed': True,
            'actual_results_ref': raw_ref, 'actual_execution_independently_derived': counts,
            'native_counter_accounting': {'INERT': True}, 'latency_and_resources': {'INERT': True},
            'pilot_completed': True, 'scoring_completed': with_score, 'status': 'INERT_DEV_UNPROMOTED',
            'reason': 'Authored schema fixture only.', 'actual_forwarded_case_ids': self.ids,
            'actual_label_decoded_case_ids': self.ids, 'Gold_not_executed_verified': False,
            'label_decode_uncertainty': {'unresolved': False, 'INERT_FIXTURE_ONLY': True},
            'actual_scored_case_ids': self.ids if with_score else [], 'actual_scoring_ref': score_ref,
            'sealed_forecasts_ref': seal_ref, 'quality_metrics': result['arms'] if with_score else None}
        audit_ref = self.fixture(A + 'DEV_actual.json', audit)
        self.cfg['DEV_pilot'] = {'results': raw_ref, 'independent_acceptance': audit_ref,
            'frozen_plan': self.cfg['DEV_plan'], 'sealed_forecasts': seal_ref,
            'scoring': score_ref, 'Gold_report': gold}
        self.cfg['DEV_nonexecution_review'] = None
        return result

    def test_exact_actual_C_counter_schema(self):
        self.assertEqual(self.m.COUNTER_KEYS, frozenset(COUNTER_SCHEMA['keys']))
        for bad in (True, 0.0):
            value = copy.deepcopy(self.counts); value['NLI_forward_calls_completed'] = bad
            with self.assertRaises(ValueError): self.m.counters(value)
        with self.assertRaises(ValueError): self.m.counters(True)
        with self.assertRaises(ValueError): self.m.counters({'engineering': self.counts, 'dev': self.counts})
        value = copy.deepcopy(self.counts); del value['GPU_calls']
        with self.assertRaises(ValueError): self.m.counters(value)

    def test_bool_audit_does_not_masquerade_as_dict(self):
        audit = self.read(self.cfg['engineering_audit']); audit['actual_execution_independently_derived'] = True
        self.cfg['engineering_audit'] = self.fixture(self.cfg['engineering_audit']['path'], audit)
        cfg_ref = self.fixture('private/INERT_reporting_config.json', self.cfg, False)
        with self.assertRaisesRegex(ValueError, 'counter dictionary'):
            self.m.main(['summarize', cfg_ref['path']])
        self.assertFalse((self.m.R / (self.m.P + 'RESULTS.json')).exists())

    def test_independent_reconstruction_must_equal_raw(self):
        audit = self.read(self.cfg['engineering_audit'])
        audit['actual_execution_independently_derived']['forecasts_completed'] = 1
        self.cfg['engineering_audit'] = self.fixture(self.cfg['engineering_audit']['path'], audit)
        self.activate()
        with self.assertRaisesRegex(ValueError, 'counts differ'): self.m.summarize(self.cfg)

    def test_honest_nonzero_failure_counter_is_retained_and_rejected(self):
        raw = self.read(self.cfg['engineering_result'])
        raw['counters']['Python_network_operations_attempted'] = 1
        self.cfg['engineering_result'] = self.fixture(self.cfg['engineering_result']['path'], raw)
        audit = self.read(self.cfg['engineering_audit'])
        audit['actual_results_ref'] = self.cfg['engineering_result']
        audit['actual_execution_independently_derived'] = copy.deepcopy(raw['counters'])
        self.cfg['engineering_audit'] = self.fixture(self.cfg['engineering_audit']['path'], audit)
        self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertEqual(result['actual_engineering_execution']['Python_network_operations_attempted'], 1)
        self.assertEqual(result['status'], 'REJECTED_ACTUAL_COUNTER_POLICY_OR_QUALITY_EVIDENCE')
        self.assertIn('forbidden_worker_activity:Python_network_operations_attempted', result['actual_counter_policy_violations']['engineering'])

    def test_nonempty_raw_errors_reject_quality_but_retain_measured_losses(self):
        expected = self.add_dev(); raw_ref = self.cfg['DEV_pilot']['results']
        raw = self.read(raw_ref); raw['errors'] = ['INERT authored failure']
        self.cfg['DEV_pilot']['results'] = self.fixture(raw_ref['path'], raw)
        audit_ref = self.cfg['DEV_pilot']['independent_acceptance']; audit = self.read(audit_ref)
        audit['actual_results_ref'] = self.cfg['DEV_pilot']['results']
        self.cfg['DEV_pilot']['independent_acceptance'] = self.fixture(audit_ref['path'], audit)
        self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertEqual(result['quality_metrics'], expected['arms'])
        self.assertFalse(result['completed_quality_evidence_checks']['no_worker_errors'])
        self.assertEqual(result['status'], 'REJECTED_ACTUAL_COUNTER_POLICY_OR_QUALITY_EVIDENCE')

    def test_invalid_NEI_coverage_and_raw_seal_disagreement_rejected(self):
        self.add_dev(); raw_ref = self.cfg['DEV_pilot']['results']
        raw = self.read(raw_ref); raw['rows'][0]['prediction'] = 'NotMentioned'
        raw['rows'][0]['all_source_windows_covered'] = False
        self.cfg['DEV_pilot']['results'] = self.fixture(raw_ref['path'], raw)
        audit_ref = self.cfg['DEV_pilot']['independent_acceptance']; audit = self.read(audit_ref)
        audit['actual_results_ref'] = self.cfg['DEV_pilot']['results']
        self.cfg['DEV_pilot']['independent_acceptance'] = self.fixture(audit_ref['path'], audit)
        self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertFalse(result['completed_quality_evidence_checks']['NEI_rows_have_full_source_coverage'])
        self.assertFalse(result['completed_quality_evidence_checks']['sealed_rows_equal_actual_raw_rows'])
        self.assertEqual(result['status'], 'REJECTED_ACTUAL_COUNTER_POLICY_OR_QUALITY_EVIDENCE')

    def test_nonexecuted_DEV_still_retires_prepared_features(self):
        cfg_ref = self.fixture('private/INERT_reporting_config.json', self.cfg, False)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.m.main(['summarize', cfg_ref['path']])
        self.assertEqual(len(json.loads(output.getvalue())['outputs']), 5)
        used = self.read(self.m.ref(self.m.R / (self.m.P + 'USED_IDENTITIES.json'), True))
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertEqual(set(used['roles']['development']), set(self.ids))
        self.assertEqual(used['actual_forwarded_DEV_identities'], [])
        self.assertEqual(used['actual_scored_DEV_identities'], [])
        self.assertEqual(used['roles']['fresh_confirmation'], [])
        self.assertIsNone(result['quality_metrics'])

    def test_actual_C_scorer_schema_flows_to_report(self):
        expected = self.add_dev(); self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertEqual(result['quality_metrics'], expected['arms'])
        self.assertEqual(result['paired_comparisons'], expected['paired_comparisons'])
        self.assertEqual(result['actual_scored_case_ids'], self.ids)
        self.assertFalse(result['candidate_promoted']); self.assertIsNone(result['official_rank'])

    def test_decoded_but_unscored_labels_retained(self):
        self.add_dev(with_score=False); self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertEqual(set(result['label_only_exposed_case_ids']), set(self.ids))
        self.assertEqual(result['actual_scored_case_ids'], [])
        self.assertIsNone(result['quality_metrics'])

    def test_seal_ref_mismatch_stops_reporting(self):
        self.add_dev(); audit = self.read(self.cfg['DEV_pilot']['independent_acceptance'])
        audit['sealed_forecasts_ref'] = dict(audit['sealed_forecasts_ref'], sha256='b' * 64)
        self.cfg['DEV_pilot']['independent_acceptance'] = self.fixture(self.cfg['DEV_pilot']['independent_acceptance']['path'], audit)
        self.activate()
        with self.assertRaisesRegex(ValueError, 'sealed forecasts'): self.m.summarize(self.cfg)

    def test_missing_Gold_report_needs_explicit_uncertainty(self):
        self.add_dev(with_score=False)
        audit_ref = self.cfg['DEV_pilot']['independent_acceptance']; audit = self.read(audit_ref)
        self.cfg['DEV_pilot']['Gold_report'] = None
        audit['actual_label_decoded_case_ids'] = []
        audit['label_decode_uncertainty'] = {'unresolved': True, 'INERT_FIXTURE_ONLY': True}
        self.cfg['DEV_pilot']['independent_acceptance'] = self.fixture(audit_ref['path'], audit)
        self.activate(); self.m.summarize(self.cfg)
        result = self.read(self.m.ref(self.m.R / (self.m.P + 'RESULTS.json'), True))
        self.assertTrue(result['label_decode_uncertainty']['unresolved'])
        self.assertIsNone(result['quality_metrics'])

    def test_manifest_preserves_exact_review_bytes(self):
        self.activate(); self.m.summarize(self.cfg)
        m = self.m
        files = sorted([m.ref(p, True) for root in m.ROOTS for p in (m.R / root).rglob('*') if p.is_file()], key=lambda row: row['path'])
        review = {'epoch': 58, 'checks_passed': True, 'public_inventory_review_passed': True, 'files': files}
        compact = json.dumps(review, separators=(',', ':')).encode() + b'\n'
        self.cfg['independent_public_review'] = self.fixture('private/INERT_public_review.json', compact, False, True)
        out = m.close(self.cfg)
        public = m.R / (m.A + 'FINAL_PUBLIC_SUMMARY_AND_INVENTORY_INDEPENDENT_REVIEW.v1.json')
        self.assertEqual(public.read_bytes(), compact)
        self.assertTrue(m.checked(out, True)['no_project_additions_after_manifest'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reporter', type=Path, required=True); parser.add_argument('--reporter-sha', required=True)
    parser.add_argument('--scorer', type=Path, required=True); parser.add_argument('--scorer-sha', required=True)
    parser.add_argument('--counter-schema', type=Path, required=True); parser.add_argument('--counter-schema-sha', required=True)
    args, remaining = parser.parse_known_args()
    scorer_raw = args.scorer.read_bytes(); schema_raw = args.counter_schema.read_bytes()
    REPORTER_PATH = args.reporter
    if hashlib.sha256(REPORTER_PATH.read_bytes()).hexdigest() != args.reporter_sha: raise ValueError('exact reviewed reporter source required')
    if hashlib.sha256(scorer_raw).hexdigest() != args.scorer_sha or hashlib.sha256(schema_raw).hexdigest() != args.counter_schema_sha:
        raise ValueError('exact reviewed C source/schema required')
    SCORER = load_module('e58_reviewed_C_scorer_inert_only', args.scorer)
    COUNTER_SCHEMA = json.loads(schema_raw)
    unittest.main(argv=[sys.argv[0], *remaining])
