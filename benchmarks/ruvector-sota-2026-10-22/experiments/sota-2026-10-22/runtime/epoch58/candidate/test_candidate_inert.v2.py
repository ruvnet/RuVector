"""Authored structural tests only; no benchmark labels or runtime/model/native use."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_common as m
import importlib.util
_spec=importlib.util.spec_from_file_location('e58_authored_worker_v2',Path(__file__).resolve().parent/'worker.v2.py')
worker=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(worker)

class Encoding:
    def __init__(self,n):self.ids=[1]+[10]*(n-2)+[2];self.attention_mask=[1]*n;self.type_ids=[0]*n
class FakeTokenizer:
    def encode(self,a,b=None):return Encoding(len(a)+(len(b) if b else 0)+3)
class FakeEvents:
    def __init__(self):self.values={k:0 for k in m.COUNTER_KEYS};self.events=[]
    def event(self,event,**kw):self.events.append((event,kw))
class FakeNLI:
    def __init__(self):self.forward_seconds=[];self.calls=[]
    def nli(self,pairs):
        self.calls.append(pairs);self.forward_seconds.append(.01);self.last_input_receipts={'authored_fixture_only':True};self.last_output_receipt={'authored_fixture_only':True}
        return [[0.,6.,0.] if a.startswith('support') else [0.,0.,4.] for a,b in pairs]

def authored_catalog(W=4,numeric=False):
    text='';windows=[]
    for i in range(W):
        t=('support' if i==W-1 else 'neutral')+' context. ';start=len(text);text+=t
        windows.append({'window_id':i,'start':start,'end':len(text),'source_unit_ids':[i],'text':t,'text_sha256':m.sha(t.encode())})
    cases=[{'id':cid,'claim':'Authored claim'+(' 7' if numeric else ''),'document_id':17,'hypothesis_id':h,'input_sha256':'0'*64} for cid,h in zip(m.DEV_IDS,m.HYPOTHESES)]
    return {'schema_version':1,'epoch':58,'kind':m.KIND,'role':'dev','document_id':17,'selected_query_ids':list(m.DEV_IDS),'reserved_ids':list(m.DEV_IDS),'frozen_window_count':W,'windows':windows,'cases':cases,'source_document_sha256':m.sha(text.encode()),'all17_pair_capacity_checked':True,'max_tokens':512,'max_windows':24,'BGE_query_prefix':m.QUERY_PREFIX}
def native(catalog,miss=False):
    W=catalog['frozen_window_count'];ids=['source-window:'+str(i) for i in reversed(range(W))]
    return {'queries':[{'query_id':cid,'ids':ids[:W-1] if miss else ids,'retrieval_miss':miss} for cid in m.DEV_IDS]}

class Tests(unittest.TestCase):
    def test_softmax(self):
        a=m.stable_scores([1.,2.,0.]);self.assertAlmostEqual(sum(a['probabilities']),1);self.assertEqual(a['prediction'],'Entailment')
    def test_aggregate_preserves_typed_neutral(self):
        self.assertEqual(m.aggregate([[-2.,-3.,0.]])['prediction'],'NotMentioned')
    def test_logodds_threshold(self):
        self.assertTrue(m.may_exit([[0.,math.log(9),0.]],[],False));self.assertFalse(m.may_exit([[0.,math.log(9),0.]],['claim_has_numerals'],False))
    def test_numbering_not_claim_evidence(self):
        self.assertEqual(m.numeric_diagnostic('A visitor signs.','Section 7. Visitor signs.')['flags'],[])
    def test_numerical_guard_diagnostic(self):
        self.assertEqual(m.numeric_diagnostic('Retain 7 copies.','Retain 8 copies.')['flags'],list(m.NUMERIC_FLAGS))
    def test_catalog_exact_partition(self):
        text='A'*700+'B'*20;units=[{'unit_id':0,'start':0,'end':700,'sentence':text[:700]},{'unit_id':1,'start':700,'end':720,'sentence':text[700:]}]
        bank={'schema_version':1,'epoch':58,'kind':m.KIND,'role':'dev','documents':[{'document_id':17,'evidence':units,'source_document_sha256':m.sha(text.encode())}],'cases':[{'id':cid,'claim':'Authored claim'} for cid in m.DEV_IDS]}
        c=m.make_catalog(bank,FakeTokenizer(),FakeTokenizer());self.assertEqual(''.join(w['text'] for w in c['windows']),text);self.assertLessEqual(c['frozen_window_count'],24);m.validate_catalog(c)
    def test_no_hidden_truncation(self):
        with self.assertRaisesRegex(ValueError,'WHOLE_UNTRUNCATED'):m.whole_pair(FakeTokenizer(),'a'*510,'claim')
    def test_roundrobin_order_can_exit_early(self):
        c=authored_catalog();events=FakeEvents();nli=FakeNLI();rows,batches,d=worker.roundrobin_forecasts(c,native(c),nli,events)
        self.assertEqual(len(rows),51);self.assertTrue(rows[0]['early_exit']);self.assertEqual(rows[0]['context_window_ids'],[3]);self.assertEqual(rows[1]['context_window_ids'],[0,1,2,3]);self.assertEqual(rows[2]['context_window_ids'],[0,1,2,3]);self.assertEqual(events.values['forecasts_completed'],51)
        self.assertLessEqual(len(batches),math.ceil(3*17*4/8));self.assertTrue(all(len(x)<=8 for x in nli.calls))
    def test_native_miss_forces_full_coverage(self):
        c=authored_catalog();rows,b,d=worker.roundrobin_forecasts(c,native(c,True),FakeNLI(),FakeEvents());self.assertTrue(all(r['all_source_windows_covered'] for r in rows if r['arm']=='native_order'));self.assertTrue(all(not r['early_exit'] for r in rows if r['arm']=='native_order'))
    def test_numeric_claim_disables_exit(self):
        c=authored_catalog(numeric=True);rows,b,d=worker.roundrobin_forecasts(c,native(c),FakeNLI(),FakeEvents());self.assertTrue(all(r['all_source_windows_covered'] and not r['early_exit'] for r in rows))
    def test_neutral_always_full(self):
        c=authored_catalog()
        for w in c['windows']:w['text']='neutral '+w['text']
        rows,b,d=worker.roundrobin_forecasts(c,native(c),FakeNLI(),FakeEvents());self.assertTrue(all(r['prediction']=='NotMentioned' and r['all_source_windows_covered'] for r in rows))
    def test_all51_order_frozen(self):
        c=authored_catalog(W=1);rows,b,d=worker.roundrobin_forecasts(c,native(c),FakeNLI(),FakeEvents());self.assertEqual([(r['case_id'],r['arm']) for r in rows],[(cid,a) for cid in m.DEV_IDS for a in m.ARMS]);self.assertEqual(len(b),math.ceil(51/8))
    def test_every_observed_logit_archived_once(self):
        c=authored_catalog();events=FakeEvents();rows,batches,d=worker.roundrobin_forecasts(c,native(c),FakeNLI(),events)
        archived=[r for b in batches for r in b['logit_rows']]
        self.assertEqual(len(archived),sum(r['actual_pair_inferences'] for r in rows))
        self.assertEqual(len({(r['case_id'],r['arm'],r['window_id']) for r in archived}),len(archived))
        self.assertEqual(len([x for x in events.events if x[0]=='NLI_BATCH_OUTPUTS_OBSERVED']),len(batches))
        self.assertTrue(all(len(r['raw_logits3'])==3 for r in archived))
    def test_counter_types(self):self.assertEqual(len(m.COUNTER_KEYS),len(set(m.COUNTER_KEYS)));self.assertTrue(all(type(x) is int for x in FakeEvents().values.values()))
if __name__=='__main__':unittest.main()
