import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from energy import integrate,analyze
from energy_compare import compare_energy
from rig import retention,verify,stage_summary,load_manifest
from sensor_data import prepare,metrics
from sensor_lab import compile_model
from transport import Device,stable
from e2e import ROOT

class EnergyTests(unittest.TestCase):
    def trace(self):
        return [dict(time_s=i*.01,voltage_v=3,current_a=.02,marker=int(1<=i<11)) for i in range(13)]
    def test_known_energy_and_idle(self):
        r=integrate(self.trace(),100,.005)
        self.assertAlmostEqual(r['gross_joules_per_decision'],.00006)
        self.assertAlmostEqual(r['idle_adjusted_joules_per_decision'],.000045)
        self.assertEqual(r['decisions'],100)
    def test_reject_bad_traces(self):
        for key,value in [('time_s',-.5),('voltage_v',float('nan')),('current_a',-.1),('marker',2)]:
            rows=self.trace();rows[5][key]=value
            with self.assertRaises(ValueError):integrate(rows,100)
        rows=self.trace();rows[0]['marker']=1
        with self.assertRaises(ValueError):integrate(rows,100)
        with self.assertRaises(ValueError):integrate(self.trace(),0)
    def test_energy_provenance_and_duration(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);trace=d/'trace.csv';report=d/'run.json'
            trace.write_text('time_s,voltage_v,current_a,marker\n'+''.join(f"{r['time_s']},3,.02,{r['marker']}\n" for r in self.trace()))
            run={'execution':'native','physical_hardware':False,'model_sha256':'model','rounds':[{'candidate':{
                'energy_batch':{'energy_batch':True,'marker_gpio':4,'runs':100,'batch_us':100000},
                'image':None,'meta':{'kernel_sha256':'kernel'}}}]}
            report.write_text(json.dumps(run));r=analyze(trace,report,0,'candidate')
            self.assertFalse(r['physical_hardware']);self.assertEqual(r['scope'],'kernel_batch')
            run['rounds'][0]['candidate']['energy_batch']['batch_us']=900000
            report.write_text(json.dumps(run))
            with self.assertRaises(ValueError):analyze(trace,report,0,'candidate')

class GateTests(unittest.TestCase):
    def test_stage_summaries_do_not_invent_capture_time(self):
        summary=stage_summary([dict(capture_us=None,parse_us=3,preprocess_us=1,inference_us=x) for x in range(1,101)])
        self.assertIsNone(summary['capture_us'])
        self.assertEqual(summary['inference_us']['p99'],99)
        self.assertEqual(summary['inference_us']['p95'],95)
    def test_packaged_manifest_paths_are_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text(json.dumps({'variants':{'candidate':{'images':{'esp32s3':{'file':'s3.bin'}}}}}))
            self.assertEqual(load_manifest(p)['variants']['candidate']['images']['esp32s3']['file'],str(Path(d)/'s3.bin'))
    def rounds(self,ratios):
        return [{'baseline':{'profile':{'mean_cycles':100*r}},'candidate':{'profile':{'mean_cycles':100}}} for r in ratios]
    def test_gains_cannot_override_provenance_correctness_or_sample_count(self):
        rounds=self.rounds([1.2]*11)
        self.assertTrue(retention(rounds,'mean_cycles','physical',True)['retain'])
        for execution,correct in [('native',True),('emulator',True),('physical',False)]:
            self.assertFalse(retention(rounds,'mean_cycles',execution,correct)['retain'])
        self.assertFalse(retention(rounds[:2],'mean_cycles','physical',True)['retain'])
        self.assertFalse(retention(self.rounds([.9,1.3]*6),'mean_cycles','physical',True)['retain'])
    def test_verify_requires_exact_model_kernel_and_target(self):
        manifest={'model_sha256':'model','variants':{'candidate':{'kernel_sha256':'kernel'}}}
        good=dict(selftest_pass=True,model_sha256='model',kernel_sha256='kernel',target='esp32c6',dynamic_frequency=False,fixed_affinity=True)
        verify(good,manifest,'candidate','physical','esp32c6')
        for key,value in [('model_sha256','wrong'),('kernel_sha256','wrong'),('target','host'),('dynamic_frequency',True),('fixed_affinity',False)]:
            bad={**good,key:value}
            with self.assertRaises(ValueError):verify(bad,manifest,'candidate','physical','esp32c6')
    def test_energy_requires_independent_consistent_physical_trials(self):
        pairs=[]
        for i in range(11):
            pairs.append({arm:dict(trace_sha256=f'{i}-{arm}',arm=arm,model_sha256='m',target='esp32c6',scope='kernel_batch',
                decisions=256,physical_hardware=True,execution='physical',gross_joules_per_decision=value)
                for arm,value in [('baseline',.00012),('candidate',.0001)]})
        self.assertTrue(compare_energy(pairs,True)['retain'])
        self.assertFalse(compare_energy(pairs,False)['retain'])
        pairs[0]['candidate']['execution']='simulated'
        self.assertFalse(compare_energy(pairs,True)['retain'])
        pairs[1]['candidate']['trace_sha256']=pairs[0]['candidate']['trace_sha256']
        with self.assertRaises(ValueError):compare_energy(pairs,True)

class SensorTests(unittest.TestCase):
    def test_temporal_split_and_training_only_statistics(self):
        data,provenance=prepare(ROOT/'build-data/occupancy.zip')
        day=provenance['validation_day']
        self.assertTrue(all(r['date'][:10]!=day for r in data['train']))
        self.assertTrue(all(r['date'][:10]==day for r in data['validation']))
        dates=[{r['date'] for r in data[k]} for k in ('train','validation','test')]
        self.assertFalse(dates[0]&dates[1] or dates[0]&dates[2] or dates[1]&dates[2])
        self.assertEqual(sum(map(len,dates)),20560)
    def test_profile_protocol_sensor_equivalence_and_recovery(self):
        folder=ROOT/'build-sensor';snapshot=json.loads((folder/'snapshot.json').read_text())
        rows=json.loads((folder/'validation.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            binary,_=compile_model(snapshot,rows,Path(d),16)
            device=Device(command=[binary])
            try:
                self.assertTrue(device.meta['sensor_pipeline'])
                raw=rows[0]['raw'];x=rows[0]['features']
                self.assertEqual(stable(device.query('sensor '+' '.join(map(str,raw)))),stable(device.query('infer '+' '.join(map(str,x)))))
                p=device.query('profile 256')
                self.assertLessEqual(p['p50_us'],p['p95_us']);self.assertLessEqual(p['p95_us'],p['p99_us'])
                self.assertEqual(p['runs'],256);self.assertEqual(device.query('energy 128')['runs'],128)
                for command in ('profile 0','profile -1','profile 2049','energy 257','profile 3junk','sensor nan','sensor 1'):
                    self.assertFalse(device.query(command)['accepted'])
                self.assertTrue(device.query('selftest')['selftest_pass'])
            finally:device.close()
    def test_incomplete_capture_cannot_reuse_previous_features(self):
        folder=ROOT/'build-sensor';snapshot=json.loads((folder/'snapshot.json').read_text())
        rows=json.loads((folder/'validation.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);stub=d/'incomplete.c'
            stub.write_text('#include "app.h"\nbool rd_sensor_read(float *v,size_t n){(void)n;v[0]=23;return true;}\n')
            binary,_=compile_model(snapshot,rows,d,16,extra_sources=[stub]);device=Device(command=[binary])
            try:
                device.query('sensor '+' '.join(map(str,rows[0]['raw'])))
                self.assertFalse(device.query('sample')['accepted'])
                self.assertTrue(device.query('selftest')['selftest_pass'])
            finally:device.close()
    def test_metrics_measure_truth_separately_from_acceptance(self):
        rows=[{'label':0},{'label':1}]
        answers=[{'index':0,'probabilities':[.8,.2],'accepted':False},{'index':0,'probabilities':[.7,.3],'accepted':True}]
        r=metrics(rows,answers)
        self.assertEqual(r['accuracy'],.5);self.assertEqual(r['coverage'],.5);self.assertEqual(r['accepted_accuracy'],0)
        self.assertAlmostEqual(r['brier_score'],.265)
    def test_capture_adapter_is_explicit_and_timed(self):
        from quantize import number
        folder=ROOT/'build-sensor';snapshot=json.loads((folder/'snapshot.json').read_text())
        rows=json.loads((folder/'validation.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);stub=d/'capture.c'
            stub.write_text('#include "app.h"\n#include <string.h>\nbool rd_sensor_read(float *v,size_t n) {'
                ' const float x[]={'+','.join(map(number,rows[0]['raw']))+'}; if(n!=5)return false;memcpy(v,x,sizeof(x));return true;}\n'
                'const char *rd_sensor_name(void){return "test_fixture_only";}\n')
            binary,_=compile_model(snapshot,rows,d,16,extra_sources=[stub])
            device=Device(command=[binary])
            try:
                answer=device.query('sample');self.assertGreaterEqual(answer['capture_us'],0)
                self.assertEqual(device.meta['capture_driver'],'test_fixture_only')
                self.assertEqual(stable(answer),stable(device.query('sensor '+' '.join(map(str,rows[0]['raw'])))))
            finally:device.close()

if __name__=='__main__':unittest.main()
