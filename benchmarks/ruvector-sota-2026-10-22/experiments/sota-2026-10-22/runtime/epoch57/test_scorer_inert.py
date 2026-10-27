"""Authored toy-only scorer isolation tests; no benchmark payload or runtime imports.

Toy relevance rows are invented engineering fixtures and never benchmark scores.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

BASE=Path(__file__).resolve().parent
SCORER=BASE/'score_sealed_retrieval.py'
EXPECTED='c99fd4eac6192c16e4b7d7e5109888e1f0585d8ecf4b638016a36161a395429b'
assert hashlib.sha256(SCORER.read_bytes()).hexdigest()==EXPECTED
spec=importlib.util.spec_from_file_location('authored_inert_sealed_scorer',SCORER)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
OFFICIAL=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada/project/experiments/sota-2026-10-22/runtime/epoch56/retrieval/beir_custom_metrics.py')
assert hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()=='3b315c1473115dc4e41d3038423fe5f1963868c3b9c92d5386839bde33472157'
spec=importlib.util.spec_from_file_location('authored_inert_official_metrics',OFFICIAL)
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
SELECTED=['toy-%02d'%i for i in range(16)]
HEADER=b'query-id\tcorpus-id\tscore\n'


def positive_rows():
 return [('%s\ttoy-doc-%02d\t1\n'%(q,i)).encode()for i,q in enumerate(SELECTED)]


class SelectedGoldTests(unittest.TestCase):
 def run_toy(self,rows,expected_exception=None):
  with tempfile.TemporaryDirectory(prefix='authored-scorer-inert-')as d:
   root=Path(d);gold=root/'authored-toy.tsv';gold.write_bytes(HEADER+b''.join(rows));progress=root/'toy-progress.jsonl'
   fd=os.open(progress,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);result=None;error=None
   try:
    if expected_exception is None:result=s.load_selected_qrels(gold,SELECTED,fd)
    else:
     try:s.load_selected_qrels(gold,SELECTED,fd)
     except expected_exception as exc:error=exc
     else:self.fail('Expected failure did not occur')
   finally:os.close(fd)
   events=[json.loads(x)for x in progress.read_text().splitlines()]
   return result,events,error

 def test_outside_invalid_utf8_and_non_numeric_grade_are_not_decoded(self):
  rows=[b'toy-outside-a\t\xff\xfe\t\xffnot-an-int\n']+positive_rows()+[b'toy-outside-z\t\x80\tbogus\textra-ignored\n']
  result,events,error=self.run_toy(rows)
  qrels,decoded,metadata=result
  self.assertEqual(decoded,16);self.assertEqual(metadata,18);self.assertEqual(set(qrels),set(SELECTED));self.assertIsNone(error)
  self.assertTrue(all(len(x)==1 and list(x.values())==[1]for x in qrels.values()))
  self.assertEqual(len(events),32)
  self.assertFalse(any('outside'in e['query_id']for e in events))

 def test_short_before_write_refuses_before_invalid_utf8_doc_decode(self):
  # If document bytes were decoded first this would raise UnicodeDecodeError,
  # rather than the required progress-write failure.
  with mock.patch.object(s.os,'write',return_value=0)as writes, mock.patch.object(s.os,'fsync')as syncs:
   _,events,error=self.run_toy([b'toy-00\t\xff\xfe\talso-not-an-int\n']+positive_rows()[1:],RuntimeError)
  self.assertEqual(str(error),'SHORT_GOLD_PROGRESS_WRITE');self.assertEqual(writes.call_count,1);self.assertEqual(syncs.call_count,0);self.assertEqual(events,[])

 def test_short_after_write_preserves_durable_before_and_aborts(self):
  with tempfile.TemporaryDirectory(prefix='authored-short-gold-inert-')as d:
   root=Path(d);gold=root/'toy.tsv';gold.write_bytes(HEADER+b''.join(positive_rows()));progress=root/'toy-progress.jsonl';fd=os.open(progress,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   real_write=os.write;real_fsync=os.fsync;calls=[];syncs=[]
   def partial_after(target,data):
    calls.append(json.loads(data)['event'])
    return real_write(target,data)if len(calls)==1 else real_write(target,data[:len(data)//2])
   def durable_before(target):syncs.append(target);return real_fsync(target)
   try:
    with mock.patch.object(s.os,'write',side_effect=partial_after), mock.patch.object(s.os,'fsync',side_effect=durable_before):
     with self.assertRaisesRegex(RuntimeError,'^SHORT_GOLD_PROGRESS_WRITE$'):s.load_selected_qrels(gold,SELECTED,fd)
   finally:os.close(fd)
   raw=progress.read_bytes();first,sep,remainder=raw.partition(b'\n');self.assertEqual(sep,b'\n')
   self.assertEqual(json.loads(first),{'query_id':'toy-00','event':'GOLD_FIELDS_DECODE_BEFORE','prior_completed_row_decodes':0})
   self.assertEqual(calls,['GOLD_FIELDS_DECODE_BEFORE','GOLD_FIELDS_DECODE_AFTER']);self.assertEqual(len(syncs),1);self.assertTrue(remainder);self.assertFalse(remainder.endswith(b'\n'))

 def test_selected_before_after_progress_exact_counts(self):
  result,events,_=self.run_toy(positive_rows())
  self.assertEqual(result[1:],(16,16))
  for i,q in enumerate(SELECTED):
   self.assertEqual(events[2*i],{'query_id':q,'event':'GOLD_FIELDS_DECODE_BEFORE','prior_completed_row_decodes':i})
   self.assertEqual(events[2*i+1],{'query_id':q,'event':'GOLD_FIELDS_DECODE_AFTER','completed_row_decodes':i+1})

 def test_selected_conflicting_duplicate_fails_with_pending_before(self):
  _,events,error=self.run_toy(positive_rows()+[b'toy-00\ttoy-doc-00\t0\n'],AssertionError)
  self.assertIsNotNone(error);self.assertEqual(len(events),33)
  self.assertEqual(events[-1],{'query_id':'toy-00','event':'GOLD_FIELDS_DECODE_BEFORE','prior_completed_row_decodes':16})

 def test_selected_consistent_duplicate_is_counted_not_hidden(self):
  result,events,_=self.run_toy(positive_rows()+[b'toy-00\ttoy-doc-00\t1\n'])
  self.assertEqual(result[1:],(17,17));self.assertEqual(len(result[0]['toy-00']),1);self.assertEqual(len(events),34)

 def test_selected_non_numeric_grade_fails_after_before_only(self):
  _,events,error=self.run_toy([b'toy-00\ttoy-doc-00\tnot-an-integer\n']+positive_rows()[1:],ValueError)
  self.assertIsNotNone(error);self.assertEqual(events,[{'query_id':'toy-00','event':'GOLD_FIELDS_DECODE_BEFORE','prior_completed_row_decodes':0}])

 def test_selected_negative_grade_fails_closed(self):
  _,events,error=self.run_toy([b'toy-00\ttoy-doc-00\t-1\n']+positive_rows()[1:],AssertionError)
  self.assertIsNotNone(error);self.assertEqual(len(events),1)

 def test_selected_invalid_utf8_document_fails_closed(self):
  _,events,error=self.run_toy([b'toy-00\t\xff\t1\n']+positive_rows()[1:],UnicodeDecodeError)
  self.assertIsNotNone(error);self.assertEqual(len(events),1)

 def test_selected_extra_grade_columns_fail_closed(self):
  _,events,error=self.run_toy([b'toy-00\ttoy-doc-00\t1\textra\n']+positive_rows()[1:],AssertionError)
  self.assertIsNotNone(error);self.assertEqual(len(events),1)

 def test_missing_selected_query_fails_before_return(self):
  _,events,error=self.run_toy(positive_rows()[:-1],AssertionError)
  self.assertIsNotNone(error);self.assertEqual(len(events),30)

 def test_selected_all_nonpositive_fails(self):
  _,events,error=self.run_toy([b'toy-00\ttoy-doc-00\t0\n']+positive_rows()[1:],AssertionError)
  self.assertIsNotNone(error);self.assertEqual(len(events),32)

 def test_main_closure_and_seal_assertions_precede_gold_loader(self):
  source=SCORER.read_text();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  load=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='load_selected_qrels')
  assertions=[(n.lineno,ast.get_source_segment(source,n))for n in ast.walk(main)if isinstance(n,ast.Assert)]
  for token in ["raw['forecast_seal']==cfg['forecasts']","outer['all_owned_experiment_processes_physically_closed']","ind['all_owned_processes_closed']","forecasts['semantic_gold_decodes_before_seal']==0","ind['forecasts_sha256']==cfg['forecasts']['sha256']"]:
   found=[line for line,text in assertions if token in text]
   self.assertTrue(found,token);self.assertTrue(all(line<load.lineno for line in found),token)
  helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='load_selected_qrels')
  condition=next(n for n in ast.walk(helper)if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and ast.get_source_segment(source,n.test)=='qid not in selected_set')
  self.assertTrue(any(isinstance(n,ast.Continue)for n in condition.body))

 def test_corpus_alignment_assertion_is_after_gold_and_rejects_unknown_docs(self):
  source=SCORER.read_text();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  load=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='load_selected_qrels')
  condition=next(n for n in ast.walk(main)if isinstance(n,ast.Assert)and "all(doc in corpus_ids"in ast.get_source_segment(source,n))
  self.assertGreater(condition.lineno,load.lineno)
  metric_calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='module'and n.func.attr in {'mrr','top_k_accuracy','recall_cap'}]
  self.assertEqual(len(metric_calls),3);self.assertTrue(all(condition.lineno<n.lineno for n in metric_calls))
  allowed={'len','corpus_ids','all','doc','rows','qrels'}
  self.assertTrue(all(n.id in allowed for n in ast.walk(condition)if isinstance(n,ast.Name)))
  self.assertTrue(all(isinstance(n.func,ast.Name)and n.func.id in {'len','all'}or isinstance(n.func,ast.Attribute)and n.func.attr=='values'for n in ast.walk(condition)if isinstance(n,ast.Call)))
  code=compile(ast.fix_missing_locations(ast.Module(body=[condition],type_ignores=[])),'authored-toy-corpus-assert','exec')
  corpus_ids={'toy-doc-%d'%i for i in range(5183)};qrels={q:{'toy-doc-%d'%i:1}for i,q in enumerate(SELECTED)}
  exec(code,{'corpus_ids':corpus_ids,'qrels':qrels})
  qrels[SELECTED[0]]={'toy-unknown-doc':1}
  with self.assertRaises(AssertionError):exec(code,{'corpus_ids':corpus_ids,'qrels':qrels})
  with self.assertRaises(AssertionError):exec(code,{'corpus_ids':set(list(corpus_ids)[:-1]),'qrels':{q:{'toy-doc-0':1}for q in SELECTED}})

 def test_percentiles_sixteen_observations(self):
  values=[i/1000 for i in reversed(range(16))]
  self.assertAlmostEqual(s.percentile(values,.5),.0075);self.assertAlmostEqual(s.percentile(values,.95),.01425)
  self.assertEqual(s.percentile([.2]*16,.95),.2)

 def test_percentiles_reject_negative_nonfinite_or_empty(self):
  for values in [[],[-1.0]*16,[float('nan')]*16,[float('inf')]*16]:
   with self.assertRaises(AssertionError):s.percentile(values,.95)

 def test_official_functions_toy_rankcodes(self):
  qrels={q:{'toy-positive':1}for q in SELECTED};ranks={}
  for i,q in enumerate(SELECTED):
   ids=['toy-positive']+['toy-negative-%02d'%j for j in range(9)]if i<8 else ['toy-negative-00','toy-positive']+['toy-negative-%02d'%j for j in range(1,9)]
   self.assertEqual(len(ids),10);ranks[q]={d:10-j for j,d in enumerate(ids)}
  self.assertTrue(all(sorted(x.values())==list(range(1,11))for x in ranks.values()))
  self.assertEqual(o.mrr(qrels,ranks,[1,10]),{'MRR@1':.5,'MRR@10':.75})
  self.assertEqual(o.top_k_accuracy(qrels,ranks,[1,10]),{'Accuracy@1':.5,'Accuracy@10':1.0})
  self.assertEqual(o.recall_cap(qrels,ranks,[1,10]),{'R_cap@1':.5,'R_cap@10':1.0})

 def test_imports_have_no_actual_scientific_modules(self):
  import sys
  self.assertFalse(set(sys.modules)&{'numpy','torch','onnxruntime','transformers','tokenizers','pyarrow'})

if __name__=='__main__':unittest.main()
