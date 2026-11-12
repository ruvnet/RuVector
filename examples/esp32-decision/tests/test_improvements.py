import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from coverage_lab import split_calibration,quality,passes,select
from sensor_data import prepare
from sensor_lab import compile_model,replay
from transport import Device,stable
from e2e import ROOT

class CoverageTests(unittest.TestCase):
    def test_selection_requires_both_classes_and_ignores_test_scores(self):
        rows=[{'label':i%2} for i in range(200)]
        answers=[dict(index=i%2,accepted=True,probabilities=[.99,.01] if i%2==0 else [.01,.99],
                      confidence=.99,abstain=0) for i in range(200)]
        good=quality(rows,answers);self.assertTrue(passes(good))
        for i,a in enumerate(answers):a['accepted']=i%2==0
        one_class=quality(rows,answers);self.assertFalse(passes(one_class))
        arms=[dict(name='strict',tau=.35,parity_pass=True,calibration=one_class,test={'accuracy':1}),
              dict(name='relaxed',tau=0.,parity_pass=True,calibration=good,test={'accuracy':0})]
        self.assertEqual(select(arms),'relaxed')
        arms[1]['parity_pass']=False;self.assertIsNone(select(arms))
    def test_small_support_and_wrong_accepted_predictions_fail(self):
        rows=[{'label':i%2} for i in range(200)]
        answers=[dict(index=i%2,accepted=i<20,probabilities=[.5,.5],confidence=.6,abstain=.4) for i in range(200)]
        self.assertFalse(passes(quality(rows,answers)))
        for a in answers:a.update(index=0,accepted=True)
        self.assertFalse(passes(quality(rows,answers)))
    def test_calibration_day_cannot_change_training_preprocessing(self):
        original,_=prepare(ROOT/'build-data/occupancy.zip')
        before=copy.deepcopy(original);data,day=split_calibration(original)
        self.assertEqual(original,before)
        changed=copy.deepcopy(original)
        for row in changed['train']:
            if row['date'][:10]==day:row['raw'][0]+=1000
        changed,_=split_calibration(changed)
        self.assertEqual(data['preprocessing'],changed['preprocessing'])
        self.assertEqual(data['train'],changed['train'])
        self.assertFalse({r['date'] for r in data['train']}&{r['date'] for r in data['calibration']})
        self.assertEqual(len(data['train'])+len(data['calibration']),len(original['train']))

class ProfileStorageTests(unittest.TestCase):
    def test_zero_and_bounded_storage_preserve_replies_and_energy(self):
        folder=ROOT/'build-sensor';snapshot=json.loads((folder/'snapshot.json').read_text())
        rows=json.loads((folder/'validation.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            reference=None
            for capacity in (2048,64,0):
                binary,_=compile_model(snapshot,rows,Path(tmp)/str(capacity),8,profile_samples=capacity)
                device=Device(command=[binary])
                try:
                    self.assertEqual(device.meta['profile_buffer_bytes'],capacity*8)
                    self.assertEqual(device.meta['profile_capacity'],capacity)
                    if capacity:
                        self.assertEqual(device.query('profile '+str(capacity))['runs'],capacity)
                        self.assertIn('error',device.query('profile '+str(capacity+1)))
                    else:self.assertEqual(device.query('profile 64')['error'],'profile_disabled')
                    self.assertEqual(device.query('energy 256')['runs'],256)
                    self.assertTrue(device.query('selftest')['selftest_pass'])
                finally:device.close()
                actual=[stable(a) for a in replay(binary,rows)]
                if reference is None:reference=actual
                else:self.assertEqual(actual,reference)
                symbols=subprocess.check_output(['nm','-S',str(binary)],text=True)
                allocated=sum(int(line.split()[1],16) for line in symbols.splitlines()
                              if len(line.split())==4 and line.split()[-1] in ('times','cycles'))
                self.assertEqual(allocated,capacity*8)

if __name__=='__main__':unittest.main()
