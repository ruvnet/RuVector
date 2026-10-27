"""Source-only tests. No subprocess, model, asset GET or native package is used."""
import base64
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e58_native_adapter_inert',HERE/'native_adapter.v2.py')
adapter = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def block(vectors):
    raw = b''.join(struct.pack('<384f',*v) for v in vectors)
    return {'encoding':'base64','dtype':'float32le','shape':[len(vectors),384],
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'data_base64':base64.b64encode(raw).decode()}


def fixture(missing=False):
    config = {'schema_version':1,'epoch':58,'kind':adapter.KIND,'phase':'dev',
              'corpus_document_count':2,'query_count':17,'plan_sha256':'a'*64,
              'native_helper_max_seconds':60,'native_process_timeout_seconds':85,
              'selected_query_ids':['query-%02d'%i for i in range(17)]}
    e0 = [1.0]+[0.0]*383;e1 = [0.0,1.0]+[0.0]*382
    bundle = {'schema_version':1,'epoch':58,'kind':adapter.KIND,'phase':'dev','plan_sha256':'a'*64,
              'corpus':{'ids':['window-0','window-1'],'vectors':block([e0,e1])},
              'queries':{'ids':list(config['selected_query_ids']),'vectors':block([e0]*17)}}
    source_raw = base64.b64decode(bundle['corpus']['vectors']['data_base64'])
    # Deliberately use the native order opposite exact cosine: the adapter must
    # validate and retain this order, not silently sort by its diagnostic.
    hits = []
    for index in ([1] if missing else [1,0]):
        hits.append({'id':'window-%d'%index,'score':float(2-index),
            'vector_receipt':{'source_vector_index':index,'dimensions':384,'dtype':'float32le',
                'sha256':adapter.sha(source_raw[index*1536:(index+1)*1536]),'source_equal_before_stripping':True},
            'metadata_receipt':{'present':False}})
    result = {'schema_version':1,'epoch':58,'kind':adapter.KIND,'phase':'dev','plan_sha256':'a'*64,
              'native_config_sha256':'b'*64,'bundle_sha256':'c'*64,'helper_source_sha256':'d'*64,
              'native_binary_sha256':adapter.BINARY_SHA,'errors':[],'stage':'COMPLETE',
              'corpus_document_count':2,'query_count':17,'top_k':2,'binary_version':'2.3.0',
              'package_version':'0.1.33','platform_package_version':'0.1.31',
              'raw_native_scores_are_not_probabilities':True,'queries':[]}
    result.update(adapter.limits(config))
    result['native_hit_source_validations'] = len(hits)*17
    for query in config['selected_query_ids']:
        result['queries'].append({'query_id':query,'ids':[h['id']for h in hits],
            'scores':[h['score']for h in hits],'rank_scores':list(range(2,2-len(hits),-1)),
            'raw_ids':[h['id']for h in hits],'raw_scores':[h['score']for h in hits],
            'raw_hits':copy.deepcopy(hits),'requested_k':2,'elapsed_seconds':.001,
            'retrieval_miss':missing,'coverage_complete':not missing,
            'missing_source_ids':['window-0']if missing else [],'fallback_required':missing,
            'exact_cosine_diagnostic':[{'id':'window-0','cosine':1.0},{'id':'window-1','cosine':0.0}],
            'exact_full_order_agreement':False,'native_order_used_except_returned_equal_score_ties':True,
            'exact_order_never_substituted':True})
    result['observedHits'] = len(hits)*17
    result['retrieval_miss_count'] = 17 if missing else 0
    return config,bundle,result


class NativeInert(unittest.TestCase):
    def validate(self, config, bundle, result):
        return adapter.validate_result(result,config,bundle,{'sha256':'c'*64},
                                       {'sha256':'b'*64},{'sha256':'d'*64})

    def test_full_native_order_not_replaced_by_exact_cosine(self):
        config,bundle,result = fixture()
        actual = self.validate(config,bundle,result)
        self.assertEqual(actual['native_hit_source_validations'],34)
        self.assertEqual(result['queries'][0]['ids'],['window-1','window-0'])
        self.assertFalse(result['queries'][0]['exact_full_order_agreement'])

    def test_missing_hits_require_supplied_context_fallback(self):
        config,bundle,result = fixture(True)
        self.assertEqual(self.validate(config,bundle,result)['native_hit_source_validations'],17)
        self.assertTrue(all(r['fallback_required'] for r in result['queries']))
        result['queries'][0]['fallback_required'] = False
        with self.assertRaisesRegex(ValueError,'NATIVE_MISS_FULL_CONTEXT_FALLBACK'):
            self.validate(config,bundle,result)

    def test_source_vector_receipt_cannot_be_forged(self):
        config,bundle,result = fixture()
        result['queries'][0]['raw_hits'][0]['vector_receipt']['sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError,'NATIVE_SOURCE_VECTOR_RECEIPT'):
            self.validate(config,bundle,result)

    def test_duplicate_unknown_or_reordered_hits_rejected(self):
        for mutation in ('duplicate','unknown','reorder'):
            config,bundle,result = fixture()
            row = result['queries'][0]
            if mutation == 'duplicate':row['ids'] = ['window-0','window-0']
            elif mutation == 'unknown':row['ids'][0] = 'unprovided-source'
            else:row['ids'].reverse()
            with self.assertRaises(ValueError):self.validate(config,bundle,result)

    def test_true_cosine_diagnostic_is_checked_without_selecting(self):
        config,bundle,result = fixture()
        result['queries'][0]['exact_cosine_diagnostic'][0]['cosine'] = .9
        with self.assertRaisesRegex(ValueError,'DIAGNOSTIC_COSINE_RECALCULATION'):
            self.validate(config,bundle,result)

    def test_bundle_hash_normalization_and_nonfinite_rejected(self):
        config,bundle,_ = fixture()
        bad = copy.deepcopy(bundle)
        bad['queries']['vectors']['sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError,'F32_HASH_BYTES'):adapter.validate_bundle(config,bad)
        for first in (float('nan'),float('inf'),0.0):
            bad = copy.deepcopy(bundle)
            vector = [first]+[0.0]*383
            bad['queries']['vectors'] = block([vector]*17)
            with self.assertRaisesRegex(ValueError,'FINITE_NORMALIZED_F32'):adapter.validate_bundle(config,bad)

    def test_frozen_query_ids_and_features_only(self):
        config,bundle,_ = fixture()
        bundle['queries']['ids'][0] = 'another-question'
        with self.assertRaisesRegex(ValueError,'FIXED_QUERY_IDS'):adapter.validate_bundle(config,bundle)
        config,bundle,_ = fixture();bundle['labels'] = ['a']
        with self.assertRaisesRegex(ValueError,'BUNDLE_BINDING'):adapter.validate_bundle(config,bundle)

    def test_counter_bool_overrun_and_inflight_rejected(self):
        config,_,result = fixture()
        for key,value in [('native_queries_completed',True),('native_queries_attempted',18),
                          ('native_insertions_completed',0)]:
            bad = copy.deepcopy(result);bad[key] = value
            with self.assertRaises(ValueError):adapter.observed_counters(bad,config,terminal=True)

    def test_cpu_and_batch_caps(self):
        config,_,_ = fixture()
        for key,value in [('corpus_document_count',25),('query_count',18),
                          ('native_helper_max_seconds',61),('native_process_timeout_seconds',86)]:
            bad = dict(config);bad[key]=value
            with self.assertRaises(ValueError):adapter.geometry(bad)

    def test_exact_file_ref_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source';path.write_bytes(b'public test source')
            ref = adapter.reference(path)
            self.assertEqual(adapter.checked(ref),path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):adapter.checked(ref)
            link = Path(directory)/'alias';link.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'CANONICAL_NO_SYMLINK_PATH'):adapter.reference(link)

    def test_pid_reuse_is_not_signaled(self):
        identity = {'pid':123456,'startticks':10,'pgid':77,'sid':77}
        with patch.object(adapter,'process_identity',return_value={**identity,'startticks':11,'state':'R'}),\
             patch.object(adapter.os,'kill') as kill:
            self.assertTrue(adapter.physically_closed(identity))
            adapter.signal_exact(identity,15)
            kill.assert_not_called()

    def test_live_process_is_not_reported_closed(self):
        identity = adapter.process_identity(os.getpid())
        self.assertFalse(adapter.physically_closed(identity))

    def test_monotonic_actual_counters_are_not_expected_fill(self):
        config,_,result = fixture(True)
        observed = adapter.observed_counters(result,config,terminal=True)
        target = {k:0 for k in adapter.COUNTER_KEYS}
        adapter.merge_counts(target,observed)
        self.assertEqual(target['native_hit_source_validations'],17)
        self.assertNotEqual(target['native_hit_source_validations'],adapter.limits(config)['native_hit_source_validations'])
        self.assertEqual(target['native_processes_attempted'],0)
        with self.assertRaisesRegex(ValueError,'NATIVE_COUNTS_MONOTONIC'):
            adapter.merge_counts(target,{k:0 for k in adapter.NATIVE_KEYS})

    def test_over_cap_failed_actuals_retained_but_never_accepted(self):
        config,_,result = fixture()
        result['Node_network_operations_attempted'] = 1
        with self.assertRaisesRegex(ValueError,'NATIVE_COUNTER_SCHEMA_CAP'):
            adapter.observed_counters(result,config,terminal=True)
        actual = adapter.observed_counters(result,config,terminal=False,enforce_caps=False)
        target = {k:0 for k in adapter.COUNTER_KEYS}
        adapter.merge_counts(target,actual)
        self.assertEqual(target['Node_network_operations_attempted'],1)


if __name__ == '__main__':
    unittest.main()
