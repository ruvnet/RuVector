"""Inert publication safety tests. No live guard CLI, Git, dataset or model call."""
from pathlib import Path
import copy
import importlib.util
import json
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e58_guard_inert', HERE / 'guard.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
spec = importlib.util.spec_from_file_location('e58_stage_inert', HERE / 'stagebuilder.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

class PublicationSafety(unittest.TestCase):
    def test_exact_denyset_and_context(self):
        self.assertEqual(g.DENY, {'486888de9602bbb5d817a55f707b4fcfd11c65e0', '3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5', 'a68b7ba587332f176f65d0348de3e910aa16e77a'})
        self.assertEqual(g.BASE, 'db7b9b21f4a13e5fc3444a8fe40de26583558e2e')
        self.assertEqual(g.KIND, 'REAL_NATIVE_TYPED_NLI_COVERAGE_ORDER_DEV')
        self.assertEqual(g.ROOTS, ('experiments/sota-2026-10-22/epoch58/', 'experiments/sota-2026-10-22/audit/epoch58/', 'experiments/sota-2026-10-22/runtime/epoch58/'))

    def test_every_forbidden_live_ref_rejected(self):
        for forbidden in g.DENY:
            with self.subTest(sha=forbidden), self.assertRaises(ValueError):
                g.validate_request(g.TOOLS['commit'], {'repository_full_name': g.REPO, 'tree_sha': forbidden, 'parent_sha': g.BASE, 'message': 'inert'})

    def test_only_scoped_nonforce_cas(self):
        good = {'repository_full_name': g.REPO, 'branch_name': g.BRANCH, 'sha': '1' * 40, 'force': False, 'expected_sha': g.BASE}
        g.validate_request(g.TOOLS['ref'], good)
        for change in ({'branch_name': 'main'}, {'force': True}, {'expected_sha': None}, {'repository_full_name': 'other/repo'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                g.validate_request(g.TOOLS['ref'], dict(good, **change))

    def test_sparse_tree_cannot_change_unrelated_path(self):
        good = {'repository_full_name': g.REPO, 'base_tree_sha': g.BASE_ROOT, 'tree_elements': [{'path': g.SCOPE + g.ROOTS[0] + 'example.json', 'mode': '100644', 'type': 'blob', 'sha': '1' * 40}]}
        g.validate_request(g.TOOLS['tree'], good)
        for path in ('README.md', g.SCOPE + 'unrelated.txt', g.SCOPE + g.ROOTS[0] + '../escape.json'):
            value = copy.deepcopy(good); value['tree_elements'][0]['path'] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                g.validate_request(g.TOOLS['tree'], value)

    def test_historical_literals_allowed_in_blob_payload(self):
        value = {'repository_full_name': g.REPO, 'encoding': 'utf-8', 'content': json.dumps({'historical_forbidden_objects': sorted(g.DENY)})}
        g.validate_request(g.TOOLS['blob'], value)

    def test_identifying_blob_ack_exact(self):
        value = {'repository_full_name': g.REPO, 'encoding': 'utf-8', 'content': 'inert'}
        expected = g.git(b'inert')
        self.assertEqual(g.acknowledge(g.TOOLS['blob'], value, {'sha': expected}, expected), expected)
        for answer in ({'success': True}, {'sha': '2' * 40}):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                g.acknowledge(g.TOOLS['blob'], value, answer, expected)

    def test_bare_cas_ack_requires_exact_fresh_ref(self):
        request = {'repository_full_name': g.REPO, 'branch_name': g.BRANCH, 'sha': '1' * 40, 'force': False, 'expected_sha': g.BASE}
        fresh = {'ref': 'refs/heads/' + g.BRANCH, 'object': {'sha': '1' * 40}}
        self.assertIsNone(g.acknowledge(g.TOOLS['ref'], request, {'success': True}, fresh_ref=fresh))
        for bad in (None, {'ref': 'refs/heads/main', 'object': {'sha': '1' * 40}}, {'ref': fresh['ref'], 'object': {'sha': '2' * 40}}):
            with self.subTest(fresh=bad), self.assertRaises(ValueError):
                g.acknowledge(g.TOOLS['ref'], request, {'success': True}, fresh_ref=bad)

    def test_transport_error_and_mcp_error_are_unknown(self):
        for raw in ({'transport_error': 'inert disconnection'}, {'isError': True, 'structuredContent': {'sha': '1' * 40}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                g.unwrap(raw)

    def test_pending_or_failed_journal_is_not_clean(self):
        row = {'phase': 'intent', 'id': 'inert', 'kind': g.TOOLS['blob']}
        good = dict(row, phase='response', ok=True)
        g.clean([row, good])
        for rows in ([row], [row, dict(good, ok=False)], [row, dict(good, id='other')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                g.clean(rows)

    def test_json_duplicate_and_nonfinite_are_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                g.parse(raw)
        self.assertFalse(g.same(True, 1))

    def test_durable_write_is_exclusive_and_body_rejects_symlink(self):
        with tempfile.TemporaryDirectory(prefix='e58-publication-inert-') as temp:
            p = Path(temp) / 'new.json'; g.write(p, b'{}\n')
            with self.assertRaises(FileExistsError): g.write(p, b'changed')
            self.assertEqual(g.body(p), b'{}\n')
            link = Path(temp) / 'link.json'; link.symlink_to(p)
            with self.assertRaises(ValueError): g.body(link)

    def test_public_text_excludes_secret_or_raw_page(self):
        for raw in (('hf_' + 'X' * 25).encode(), ('<' + 'html>raw</' + 'html>').encode()):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                g.public_text(g.ROOTS[0] + 'inert.json', raw)

    def test_empty_tree_known_digest(self):
        self.assertEqual(g.tree_sha([]), '4b825dc642cb6eb9a060e54bf8d69288fbee4904')

    def test_control_preservation_and_append_only_budget(self):
        old = {'goal_met': False, 'official_rank': None, 'champion': {'name': 'native104'}, 'unused_confirmation_sets': {'status': 'UNQUALIFIED_HOLD'}, 'cumulative_budget_ledger': {'old': {'actual': 0}}, 'exposed_development_sets': ['oldDEV'], 'active_run': {'epoch': 57}, 'unrelated': {'typed': [True, 1]}}
        cp = {'project_files': [], 'old_metadata': {'keep': True}}
        active = {'epoch': 58, 'phase': 'final', 'source_checkpoint': g.BASE, 'owner_sha256': g.OWNER['sha256'], 'kind': g.KIND, 'work_active': False, 'experiments_active': False, 'OS_lock_observation_scope': 'PREPUBLICATION_SNAPSHOT_NOT_CURRENT_REMOTE_LEASE', 'publication_is_not_scientific_authorization': True}
        x = {'phase': 'final', 'CONT_patch': {'epoch58_final': {'status': 'inert'}, 'cumulative_budget_ledger': {'old': {'actual': 0}, 'epoch58_actual_execution': {'actual': 0}}, 'exposed_development_sets': ['oldDEV', 'newDEV']}, 'ACTIVE': active, 'CP_phase_record': {'inert': True}}
        result, nextcp, _ = s.preserve_controls(old, cp, x, [])
        self.assertEqual(result['champion'], old['champion']); self.assertEqual(result['unrelated'], old['unrelated'])
        self.assertEqual(nextcp['old_metadata'], cp['old_metadata'])
        bad = copy.deepcopy(x); bad['CONT_patch']['cumulative_budget_ledger']['old']['actual'] = 1
        with self.assertRaises(ValueError): s.preserve_controls(old, cp, bad, [])
        bad = copy.deepcopy(x); bad['CONT_patch']['champion'] = {'name': 'changed'}
        with self.assertRaises(ValueError): s.preserve_controls(old, cp, bad, [])

if __name__ == '__main__': unittest.main()
