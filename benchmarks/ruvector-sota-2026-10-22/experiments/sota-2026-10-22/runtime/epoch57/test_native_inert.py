"""Inert E57 unit tests only: no model/native runtime, benchmark or gold reads."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('native_bridge',ROOT/'native_bridge.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
LEDGER=Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada/project/experiments/sota-2026-10-22/runtime/epoch56/retrieval/native_accounting.py')
LEDGER_REF={'path':str(LEDGER),'bytes':10650,'sha256':'379588fcbe15f9db598c899740c54c280237d1853a3fea6575b72d0898ff21be'}
JS=r'''
const assert=require('node:assert/strict'),m=require(process.argv[1]);
const plans=Object.entries(m.GEOMETRY).map(([phase,g])=>({schema_version:1,epoch:57,kind:g.kind,phase,
 corpus_document_count:g.docs,query_count:16,top_k:10,native_helper_max_seconds:60,
 bge_model:{sha256:m.MODEL_SHA},selected_query_ids:Array.from({length:16},(_,i)=>'authored-q'+i)}));
const result=[];
for(const plan of plans){
 const rows=[],expected=m.expected(plan),c=m.makeCounters(event=>rows.push({event,
 monotonic_milliseconds:rows.length,counters:{...c.counters}}),expected);
 const emit=event=>rows.push({event,monotonic_milliseconds:rows.length,counters:{...c.counters}});
 c.attempt('native_binary_loads','NATIVE_LOAD_BEFORE');c.complete('native_binary_loads');emit('NATIVE_LOAD_COMPLETE');
 c.attempt('native_version_calls','NATIVE_VERSION_BEFORE');c.complete('native_version_calls');
 c.attempt('native_index_initializations','INDEX_BEFORE');c.complete('native_index_initializations');
 c.attempt('native_len_calls',null);c.complete('native_len_calls');
 for(let i=0;i<plan.corpus_document_count;i++){c.attempt('native_insertions','INSERT_BEFORE');c.complete('native_insertions');}
 c.attempt('native_len_calls',null);c.complete('native_len_calls');
 for(let i=0;i<16;i++){c.attempt('native_queries','QUERY_BEFORE');c.complete('native_queries');for(let j=0;j<10;j++)c.hit();emit('QUERY_COMPLETE');}
 c.terminal();emit('COMPLETE');assert.deepEqual(c.counters,expected);
 assert.equal(rows.length,plan.corpus_document_count+37);
 assert.throws(()=>c.attempt('native_insertions','INSERT_BEFORE'),/CAP_OR_INFLIGHT/);
 assert.throws(()=>c.hit(),/HIT_VALIDATION_CAP/);
 assert.throws(()=>m.expected({...plan,epoch:56}),/PLAN_GEOMETRY/);
 assert.throws(()=>m.expected({...plan,bge_model:{sha256:'0'.repeat(64)}}),/PROVENANCE_PIN/);
 assert.throws(()=>m.expected({...plan,corpus_document_count:1}),/PLAN_GEOMETRY/);
 const v=new Float32Array(384);v[0]=1;
 const hit=m.compact({id:'authored-B',score:.4,vector:v},[v],new Map([['authored-B',0]]));
 assert.equal(hit.vector_receipt.source_equal_before_stripping,true);
 assert.throws(()=>m.compact({id:'authored-C',score:.4,vector:v},[v],new Map([['authored-B',0]])),/KNOWN_SOURCE_ID/);
 assert.deepEqual(m.returnedTieOrder([{id:'B',score:.4},{id:'A',score:.4},{id:'C',score:.3}]).map(x=>x.id),['A','B','C']);
 result.push({plan,expected,rows});
}
// Authored mock-fs failures: the attempted operation is never invoked when its
// BEFORE progress write is zero/short; partial bytes and attempted!=completed
// remain evidence, never a fabricated completion.
const fs=require('node:fs'),originalWrite=fs.writeSync,originalFsync=fs.fsyncSync;
for(const mode of ['zero','short']){
 let invoked=false,flushed=false,partial='';
 const line=JSON.stringify({event:'INSERT_BEFORE',authored_toy:true})+'\n';
 try{
  fs.writeSync=(fd,raw)=>{partial=mode==='zero'?'':raw.slice(0,-1);return Buffer.byteLength(partial);};
  fs.fsyncSync=()=>{flushed=true;};
  const c=m.makeCounters(()=>m.writeProgressLine(999,line),m.expected(plans[0]));
  assert.throws(()=>{c.attempt('native_insertions','INSERT_BEFORE');invoked=true;},/SHORT_NATIVE_PROGRESS_WRITE/);
  assert.equal(invoked,false);assert.equal(flushed,false);
  assert.equal(c.counters.native_insertions_attempted,1);assert.equal(c.counters.native_insertions_completed,0);
  assert.equal(partial.length,mode==='zero'?0:line.length-1);
 }finally{fs.writeSync=originalWrite;fs.fsyncSync=originalFsync;}
}
assert(!Object.keys(require.cache).some(x=>/ruvector\.node|\/node_modules\/|onnxruntime|transformers/.test(x)));
console.log(JSON.stringify(result));
'''

class Inert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=json.loads(subprocess.run(['node','-e',JS,str(ROOT/'native_helper.cjs')],
            check=True,capture_output=True,text=True).stdout)
        cls.Ledger=b.load_ledger(LEDGER_REF)

    def capture(self,item,raw=None):
        counters=Counter();ledger=self.Ledger(counters,item['expected'])
        with tempfile.TemporaryDirectory(prefix='e57-inert-')as folder:
            path=Path(folder)/'fake-progress.jsonl'
            path.write_bytes(raw if raw is not None else b''.join((json.dumps(x)+'\n').encode()for x in item['rows']))
            ledger.capture(path)
        return counters,ledger

    def test_both_dynamic_geometries_and_exact_ledger(self):
        for item in self.fixtures:
            self.assertEqual(b.expected_native(item['plan']),item['expected'])
            counters,ledger=self.capture(item)
            ledger.finalize(dict(item['expected'],stage='COMPLETE',errors=[]))
            self.assertTrue(ledger.receipt()['qualified']);self.assertEqual(dict(counters),item['expected'])
            self.assertEqual(ledger.records,item['plan']['corpus_document_count']+37)

    def test_counter_replay_is_assignment_not_addition(self):
        item=self.fixtures[0];counters=Counter();ledger=self.Ledger(counters,item['expected'])
        row=item['rows'][0];ledger.ingest(row);ledger.ingest(row)
        self.assertEqual(counters['native_binary_loads_attempted'],1)
        self.assertFalse(ledger.qualified)

    def test_partial_truncated_capture_retains_observed_prefix(self):
        item=self.fixtures[0];counters=Counter();ledger=self.Ledger(counters,item['expected'])
        raw=(json.dumps(item['rows'][0])+'\n'+json.dumps(item['rows'][1])[:-1]).encode()
        with tempfile.TemporaryDirectory(prefix='e57-inert-')as folder:
            p=Path(folder)/'partial.jsonl';p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'FINAL_NEWLINE'):ledger.capture(p)
        self.assertEqual(counters['native_binary_loads_attempted'],1)
        self.assertEqual(counters['native_binary_loads_completed'],0)
        self.assertEqual(counters['native_insertions_completed'],0)
        self.assertTrue(ledger.receipt()['inflight_native_call_outcome_uncertain'])
        self.assertFalse(ledger.qualified)

    def test_missing_capture_never_fills_expected(self):
        item=self.fixtures[0];counters=Counter();ledger=self.Ledger(counters,item['expected'])
        with tempfile.TemporaryDirectory(prefix='e57-inert-')as folder:
            with self.assertRaisesRegex(ValueError,'CAPTURE_REQUIRED'):ledger.capture(Path(folder)/'missing')
        self.assertEqual(dict(counters),{});self.assertFalse(ledger.qualified)

    def test_changed_geometry_epoch_or_model_fails(self):
        p=self.fixtures[0]['plan']
        for changes in ({'epoch':56},{'corpus_document_count':5183},{'top_k':9},{'phase':'other'},
                        {'bge_model':{'sha256':'0'*64}}):
            with self.assertRaises(ValueError):b.expected_native(dict(p,**changes))

    def test_exact_ledger_pin_rejects_changed_source(self):
        with self.assertRaisesRegex(ValueError,'LEDGER_PIN'):
            b.load_ledger(dict(LEDGER_REF,sha256='0'*64))

    def test_zero_and_short_output_writes_fail_closed_preserving_partial_artifacts(self):
        real_write=b.os.write
        with tempfile.TemporaryDirectory(prefix='e57-inert-')as folder:
            zero=Path(folder)/'zero.json'
            with mock.patch.object(b.os,'write',return_value=0):
                with self.assertRaisesRegex(ValueError,'SHORT_NATIVE_BRIDGE_OUTPUT_WRITE'):b.put(zero,{'authored':'toy'})
            self.assertTrue(zero.is_file());self.assertEqual(zero.stat().st_size,0)
            short=Path(folder)/'short.json'
            with mock.patch.object(b.os,'write',side_effect=lambda fd,raw:real_write(fd,raw[:len(raw)//2])):
                with self.assertRaisesRegex(ValueError,'SHORT_NATIVE_BRIDGE_OUTPUT_WRITE'):b.put(short,{'authored':'toy'})
            self.assertTrue(short.is_file());self.assertGreater(short.stat().st_size,0)
            self.assertLess(short.stat().st_size,len((json.dumps({'authored':'toy'},sort_keys=True)+'\n').encode()))

    def test_progress_bounds(self):
        for item in self.fixtures:
            raw=b''.join((json.dumps(x)+'\n').encode()for x in item['rows'])
            self.assertLessEqual(len(raw),4194304);self.assertLessEqual(len(item['rows']),10000)

    def test_repeated_complete_capture_fails_closed(self):
        item=self.fixtures[0];_,ledger=self.capture(item)
        with self.assertRaisesRegex(ValueError,'ONE_SHOT'):ledger.capture('/missing')
        self.assertFalse(ledger.qualified)

    def test_parent_counter_drift_latches_failure(self):
        item=self.fixtures[0];counters=Counter();ledger=self.Ledger(counters,item['expected'])
        ledger.ingest(item['rows'][0]);counters['native_binary_loads_attempted']=2
        with self.assertRaisesRegex(ValueError,'PARENT_COUNTER_DRIFT'):ledger.ingest(item['rows'][1])
        self.assertTrue(ledger.receipt()['validation_failure_latched'])

if __name__=='__main__':unittest.main(verbosity=2)
