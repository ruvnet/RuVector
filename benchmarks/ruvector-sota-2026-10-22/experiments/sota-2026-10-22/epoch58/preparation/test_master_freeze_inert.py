"""Inert master/SF tests in temporary projects; no live owner/Git/model/data."""
from pathlib import Path
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class MasterFreezeSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e58-freeze-INERT-')
        self.m = load('e58_master_freeze_inert_' + str(id(self)), HERE / 'build_master_and_freeze.py')
        self.m.W = Path(self.temp.name); self.m.R = self.m.W / 'project'; self.m.R.mkdir()
        m = self.m; self.ids = ['contractnli:eced6528:dev:17:INERT-' + str(i) for i in range(17)]
        guard_text = (HERE / 'guard.py').read_text().replace('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1', str(m.W))
        start = guard_text.index('def owner_fence():'); end = guard_text.index('\ndef validate_request(', start)
        guard_text = guard_text[:start] + "def owner_fence(): return {'INERT_ONLY': True}\n\n" + guard_text[end:]
        self.guard_ref = self.make('publication-helpers-g26/guard.py', guard_text.encode(), False, True)
        self.g = load('e58_master_guard_inert_' + str(id(self)), Path(self.guard_ref['path']))
        self.make(m.P + 'publication/guard.py', guard_text.encode(), True, True)
        self.make(m.BUILDER_PUBLIC_PATH, (HERE / 'build_master_and_freeze.py').read_bytes(), True, True)
        self.data_ref = self.make(m.P + 'data/DATA_REPORT.json', {'epoch': 58, 'kind': m.KIND,
            'source_checkpoint': m.BASE, 'actual_feature_case_ids': self.ids, 'reserved_development_case_ids': self.ids,
            'semantic_gold_decodes': 0, 'used_confirmation_identities': [], 'INERT_ONLY': True})
        self.catalog_ref = self.make(m.P + 'data/CATALOG_REPORT.json', {'epoch': 58, 'kind': m.KIND,
            'status': 'BLIND_CATALOG_READY', 'selected_query_ids': self.ids, 'all17_claims_capacity_checked': True,
            'frozen_window_count': 2, 'NLI_forward_calls': 0, 'native_corpus_batches': 0,
            'Gold_labels_decoded': 0, 'paid_API_calls': 0, 'GPU_calls': 0, 'INERT_ONLY': True})
        local_plan = self.make(m.P + 'preparation/LOCAL_PLAN.json', {'INERT_ONLY': True})
        local_freeze = self.make(m.P + 'preparation/LOCAL_SOURCE_FREEZE.json', {'INERT_ONLY': True})
        runtime = self.make('private/INERT_weights.onnx', b'INERT placeholder, not a model.', False, True)
        public = sorted([self.project_ref(path) for root in m.ROOTS for path in (m.R / root).rglob('*') if path.is_file()], key=lambda r: r['path'])
        review = self.make('private/INERT_public_review.json', {'epoch': 58, 'checks_passed': True,
            'public_inventory_review_passed': True, 'files': public}, False)
        self.config = {'epoch': 58, 'kind': m.KIND, 'source_checkpoint': m.BASE, 'owner': self.g.OWNER,
            'publication_guard': self.guard_ref, 'public_artifacts': public, 'runtime_sources': [runtime],
            'inherited_public_sources': [], 'independent_public_source_review': review,
            'master_fields': {'development_ids': self.ids, 'frozen_window_count': 2,
                'limits': {'native_corpus_batches': 2, 'CPU_threads': 2, 'inclusive_CLI_seconds_per_batch': 600,
                           'paid_API_calls': 0, 'GPU_calls': 0}},
            'actual_preparation': {'DATA_report': self.data_ref, 'catalog_report': self.catalog_ref,
                                   'local_plan': local_plan, 'local_source_freeze': local_freeze},
            'source_qualification': {'INERT_ONLY': True}}

    def tearDown(self): self.temp.cleanup()

    def make(self, path, value, project=True, raw=False):
        m = self.m; target = m.R / path if project else m.W / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if raw else m.encoded(value))
        row = m.stream_ref(target)
        if project: row['path'] = target.relative_to(m.R).as_posix()
        return row

    def project_ref(self, path):
        row = self.m.stream_ref(path); row['path'] = path.relative_to(self.m.R).as_posix(); return row

    def invoke(self):
        m = self.m; config = self.make('private/INERT_input.json', self.config, False)
        own = m.stream_ref(Path(m.__file__).resolve())
        approval = self.make('private/INERT_approval.json', {'epoch': 58, 'checks_passed': True,
            'source_metadata_preparation_allowed': True, 'phase_input_sha256': config['sha256'],
            'builder_source_sha256': own['sha256']}, False)
        with contextlib.redirect_stdout(io.StringIO()) as result:
            m.main(['--input', config['path'], '--input-sha', config['sha256'],
                    '--approval', approval['path'], '--approval-sha', approval['sha256']])
        return json.loads(result.getvalue())

    def test_full_acyclic_freeze_runtime_hashes_without_body_publication(self):
        result = self.invoke(); m = self.m
        plan = json.loads((m.R / (m.P + 'plan.json')).read_text())
        sf = json.loads((m.R / (m.P + 'SOURCE_FREEZE.json')).read_text())
        manifest = json.loads((m.R / (m.P + 'SOURCE_MANIFEST.json')).read_text())
        self.assertEqual(plan['source_freeze_path'], m.P + 'SOURCE_FREEZE.json')
        self.assertNotIn('source_freeze_sha256', plan)
        self.assertEqual(sf['runtime_sources'], self.config['runtime_sources'])
        self.assertEqual(manifest['runtime_sources'], self.config['runtime_sources'])
        self.assertNotIn(m.P + 'SOURCE_FREEZE.json', {r['path'] for r in sf['files']})
        for row in sf['files']:
            raw = (m.R / row['path']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (row['bytes'], row['sha256']))
        self.assertFalse(any(path.name.endswith('.onnx') for path in m.R.rglob('*')))
        self.assertEqual(result['actual_model_native_Gold_Git_calls'], 0)

    def test_actual_catalog_W_mismatch_stops_before_any_generated_file(self):
        self.config['master_fields']['frozen_window_count'] = 3
        with self.assertRaisesRegex(ValueError, 'master W derives'): self.invoke()
        self.assertFalse((self.m.R / (self.m.P + 'plan.json')).exists())

    def test_runtime_ref_hash_mismatch_stops(self):
        self.config['runtime_sources'][0]['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'actual absolute source body'): self.invoke()
        self.assertFalse((self.m.R / (self.m.P + 'plan.json')).exists())

    def test_no_overwrite_or_replay_of_existing_master(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, 'never overwrite'): self.invoke()

    def test_no_unreviewed_public_file(self):
        self.make(self.m.P + 'unreviewed.txt', b'INERT', True, True)
        with self.assertRaisesRegex(ValueError, 'omitted or extra'): self.invoke()

if __name__ == '__main__': unittest.main()
