"""Authored toy E57 source/runtime checks; no models/native/benchmark inputs."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e57_runtime_inert',HERE/'runtime_common.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

def ref(p):
    b=p.read_bytes(); return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def gate():
    return {'engineering_gate':{'minimum_speedup':1.25,'maximum_projection_seconds':420,'reserve_seconds':90,'projection_safety_factor':2,'numerical_cosine_floor':.97}}

def measured(values,initialization=1):
    return {'initialization_seconds':initialization,'batch_measurements':[{'group':'corpus','forward_seconds':v} for v in values]}

class RuntimeInertTests(unittest.TestCase):
    def test_stdlib_only_module_import(self):
        self.assertFalse(set(sys.modules)&{'numpy','torch','onnxruntime','tokenizers','transformers','pyarrow'})

    def test_source_ast_all_three_workers(self):
        for name in ['runtime_common.py','engineering_worker.py','pilot_worker.py']:
            ast.parse((HERE/name).read_text())

    def test_exact_ref_hash_and_bytes_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'toy.txt'; p.write_text('authored toy')
            value=ref(p); self.assertEqual(r.verify(value),p)
            with self.assertRaisesRegex(ValueError,'EXACT_REF_SIZE_SHA256'): r.verify({**value,'bytes':999})
            q=Path(d)/'alias'; q.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'REGULAR_NONSYMLINK_REF'): r.verify({**value,'path':str(q)})

    def test_output_is_immutable_and_short_write_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json'; value=r.put(p,{'toy':True})
            self.assertEqual(ref(p),value)
            with self.assertRaises(FileExistsError): r.put(p,{'toy':False})
            with patch.object(r.os,'write',return_value=0):
                with self.assertRaisesRegex(ValueError,'COMPLETE_OUTPUT_WRITE'): r.put(Path(d)/'short.json',{'toy':True})

    def test_exact_reviewed_guard_hash_contract(self):
        self.assertEqual(set(r.GUARD_SHAS),{'inherited_guards','io_guard','search_policy'})
        self.assertEqual(r.GUARD_SHAS['io_guard'],'9bcb717c704114f9d9fb69ca6ed1838482e37a8491d81ca1162f54126056c13b')
        self.assertTrue(all(len(v)==64 for v in r.GUARD_SHAS.values()))

    def test_gold_guard_denies_path_and_payload_before_io(self):
        ctx=r.WorkerContext.__new__(r.WorkerContext); ctx.counters=Counter(); ctx.goldpaths={'/authored/labels.dat'}
        for p in ['/authored/qrels.tsv','/authored/qrels.parquet','/authored/labels.dat']:
            with self.assertRaisesRegex(RuntimeError,'QRELS_LABEL_PAYLOAD_FORBIDDEN_BEFORE_IO'): ctx.no_gold('open',(p,'r'))
        ctx.no_gold('open',('/authored/qrels_metadata.json','r'))
        self.assertEqual(ctx.counters['qrels_or_label_file_access_attempts'],3)

    def test_gold_guard_installed_before_first_plan_or_source_read(self):
        source=(HERE/'runtime_common.py').read_text(); tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WorkerContext')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        enter=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='enter')
        self.assertIn('sys.addaudithook(self.no_gold)',ast.unparse(init))
        self.assertIn('Path(self.args.plan).read_bytes()',ast.unparse(enter))
        self.assertNotIn('read_bytes',ast.unparse(init))

    def test_signal_is_deferred_then_deadline_refuses(self):
        ctx=r.WorkerContext.__new__(r.WorkerContext); ctx.interrupt_signal=None; ctx.args=SimpleNamespace(deadline=r.time.monotonic()+5)
        ctx.defer_signal(signal.SIGTERM,None); self.assertEqual(ctx.interrupt_signal,signal.SIGTERM)
        with self.assertRaisesRegex(ValueError,'DEFERRED_WORKER_SIGNAL'): ctx.deadline()
        fn=next(n for n in ast.walk(ast.parse((HERE/'runtime_common.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='defer_signal')
        self.assertFalse(any(isinstance(n,ast.Raise) for n in ast.walk(fn)))

    def test_finish_retains_exception_and_chain_after_handler_exits(self):
        held=None
        try:
            try: raise ValueError('authored toy cause')
            except ValueError as cause: raise RuntimeError('retained toy outer') from cause
        except RuntimeError as error: held=error
        self.assertEqual(r.traceback.format_exc(),'NoneType: None\n')
        with tempfile.TemporaryDirectory() as d:
            ctx=r.WorkerContext.__new__(r.WorkerContext); ctx.out=Path(d); ctx.raw={'native_state':{},'errors':[]}; ctx.counters=Counter(); ctx.network=None; ctx.denied=None; ctx.memory_snapshot=None; ctx.progress_fd=None; ctx.started=r.time.monotonic(); ctx.interrupt_signal=None
            ctx.finish(held)
            trace=(Path(d)/'worker.exception.txt').read_text()
            self.assertIn('ValueError: authored toy cause',trace); self.assertIn('RuntimeError: retained toy outer',trace)
            self.assertIn('test_finish_retains_exception_and_chain_after_handler_exits',trace)
            self.assertIn('The above exception was the direct cause',trace); self.assertNotIn('NoneType: None',trace)
            self.assertEqual(json.loads((Path(d)/'raw.json').read_bytes())['errors'][0]['type'],'RuntimeError')

    def test_exact_gate_projection_and_speedup(self):
        got=r.engineering_gate(gate(),measured([1.4]*4),measured([.8]*4,2),[.98]*128)
        self.assertAlmostEqual(got['observed_FP32_over_INT8_throughput_ratio'],1.75)
        self.assertAlmostEqual(got['conservative_5199_projection_seconds'],2*.8*163+2+90)
        self.assertTrue(got['local_gate_passed']); self.assertFalse(got['pilot_execution_authorized_by_this_report'])

    def test_gate_refuses_speed_projection_and_drift_separately(self):
        a=r.engineering_gate(gate(),measured([.9]*4),measured([.8]*4),[.98]*128)
        self.assertFalse(a['observed_throughput_speedup_pass']); self.assertFalse(a['local_gate_passed'])
        b=r.engineering_gate(gate(),measured([3]*4),measured([1.1]*4),[.98]*128)
        self.assertFalse(b['conservative_runtime_projection_pass'])
        c=r.engineering_gate(gate(),measured([1.4]*4),measured([.8]*4),[.969]+[.98]*127)
        self.assertFalse(c['paired_cosine_drift_pass'])

    def test_gate_refuses_unfrozen_floor_invalid_or_missing_batches(self):
        bad=gate(); bad['engineering_gate']['numerical_cosine_floor']=.8
        with self.assertRaisesRegex(ValueError,'FROZEN_GATE_LIMITS'): r.engineering_gate(bad,measured([1]*4),measured([.5]*4),[1.0]*128)
        for values in [[.5]*3,[math.nan]*4,[0]*4]:
            with self.assertRaisesRegex(ValueError,'FOUR_OBSERVED_POSITIVE_BATCH_TIMES'): r.engineering_gate(gate(),measured([1]*4),measured(values),[1.0]*128)

    def test_bm25_authored_ties_and_matching_word_evidence(self):
        ids=['z','a','b']; index=r.bm25_index(['cat cat','cat cat','dog dog'])
        order,scores=r.bm25_query('cat',index,ids)
        self.assertEqual([ids[i] for i in order],['a','z','b']); self.assertGreater(scores[0],scores[2])

    def test_percentiles_and_invalid_values(self):
        values=list(range(16)); self.assertEqual(r.percentile(values,.5),7.5); self.assertEqual(r.percentile(values,.95),14.25)
        for values in [[],[-1],[math.inf]]:
            with self.assertRaises(ValueError): r.percentile(values,.95)

    def test_authored_engineering_features_exact_ids_and_no_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'toy.json'; rows=[{'_id':f'd{i:03}','title':'authored','text':'toy words'} for i in range(128)]; queries=[{'_id':f'q{i:02}','text':'authored query'} for i in range(16)]
            p.write_text(json.dumps({'epoch':57,'license':'CC0-1.0','corpus':rows,'queries':queries}))
            ctx=SimpleNamespace(plan={'engineering_features':ref(p),'corpus_document_count':128,'query_count':16,'selected_query_ids':[q['_id'] for q in queries]},raw={})
            self.assertEqual(tuple(map(len,r.features(ctx,True))),(128,16))
            rows[0]['label']='invented'; p.write_text(json.dumps({'epoch':57,'license':'CC0-1.0','corpus':rows,'queries':queries})); ctx.plan['engineering_features']=ref(p)
            with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_CORPUS_FEATURES'): r.features(ctx,True)

    def test_byte_identical_e56_prepared_features_require_frozen_source_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            corpus=Path(d)/'authored-corpus.jsonl'; query=Path(d)/'authored-query.json'
            corpus.write_text(''.join(json.dumps({'_id':f'd{i:04}','title':'authored','text':'toy words'})+'\n' for i in range(5183)))
            rows=[{'_id':f'q{i:02}','text':'authored query'} for i in range(16)]; ids=[q['_id'] for q in rows]
            original=(json.dumps({'epoch':56,'selected_ids':ids,'queries':rows})+'\n').encode(); query.write_bytes(original)
            ctx=SimpleNamespace(plan={'corpus_features':ref(corpus),'query_features':ref(query),'corpus_document_count':5183,'query_count':16,'selected_query_ids':ids,'prepared_feature_source_epoch':56},raw={})
            self.assertEqual(tuple(map(len,r.features(ctx))),(5183,16)); self.assertEqual(query.read_bytes(),original)
            self.assertEqual(ctx.raw['prepared_feature_source_epoch'],56); self.assertTrue(ctx.raw['all_reserved_queries_exposed_development_forever']); self.assertEqual(ctx.raw['actual_feature_query_ids'],ids)
            ctx.plan['prepared_feature_source_epoch']=57
            with self.assertRaisesRegex(ValueError,'FROZEN_E56_PREPARED_EXPOSED_DEV_IDENTITIES'): r.features(ctx)
            ctx.plan['prepared_feature_source_epoch']=56; query.write_text(json.dumps({'epoch':57,'selected_ids':ids,'queries':rows})); ctx.plan['query_features']=ref(query)
            with self.assertRaisesRegex(ValueError,'FROZEN_E56_PREPARED_EXPOSED_DEV_IDENTITIES'): r.features(ctx)

    def test_fp32_never_retrieval_arm_and_pilot_rejects_fp32(self):
        eng=ast.parse((HERE/'engineering_worker.py').read_text()); pil=(HERE/'pilot_worker.py').read_text()
        call=next(n for n in ast.walk(eng) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='matched_forecasts')
        self.assertEqual(ast.unparse(call.args[-1]),'quant'); self.assertIn("'bge_fp32_model' not in p",pil)
        self.assertIn("PHASE='engineering'",(HERE/'engineering_worker.py').read_text())

    def test_fixed_tensor_shape_caps_no_hidden_warmup_or_gold_metrics(self):
        source=(HERE/'runtime_common.py').read_text()
        self.assertIn('enable_padding(length=256',source); self.assertIn('x.shape==(len(batch),256)',source)
        self.assertEqual(source.count('session.run('),1)
        for name in ['engineering_worker.py','pilot_worker.py']:
            s=(HERE/name).read_text(); self.assertIn('ctx.finish(error)',s); self.assertNotIn('load_selected_qrels',s); self.assertNotIn('mrr(',s)

if __name__=='__main__': unittest.main(verbosity=2)
