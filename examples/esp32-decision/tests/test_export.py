import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from quantize import export, quantize_row, validate

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.model = json.loads((Path(__file__).resolve().parents[1]/'build-host/fixtures/probe.json').read_text())
    def test_reject_bad_envelopes(self):
        for key,value in [('version',2),('dims',769),('dims',0),('temperature',0),
                          ('abstain_scale',0),('logit_scale',float('nan')),('weights',[]),
                          ('bias',[]),('kind','other'),('head','logistic')]:
            with self.subTest(key=key,value=value):
                m=copy.deepcopy(self.model); m[key]=value
                with self.assertRaises((ValueError,TypeError)):validate(m)
    def test_reject_nonfinite_and_wrong_rows(self):
        for row in [[0.0]*32,[1.0],[float('inf')]*32]:
            m=copy.deepcopy(self.model); m['prototypes'][0]=row
            with self.assertRaises(ValueError):validate(m)
    def test_symmetric_quantization_and_zero_weights(self):
        for bits in [8,16]:
            q,scale=quantize_row([-1,0,1],bits)
            self.assertEqual(q,[-((1<<(bits-1))-1),0,(1<<(bits-1))-1])
            self.assertGreater(scale,0)
            self.assertEqual(quantize_row([0,0],bits),([0,0],1.0))
    def test_deterministic_and_precision_bound_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.h'; q=Path(d)/'b.h'
            r1=export(self.model,p); r2=export(self.model,q)
            self.assertEqual(p.read_bytes(),q.read_bytes())
            self.assertEqual(r1,r2)
            self.assertNotEqual(r1['sha256'],export(self.model,q,bits=8)['sha256'])
            self.assertLess(r1['quantized_parameter_bytes'],r1['float_parameter_bytes'])

if __name__=='__main__':unittest.main()
