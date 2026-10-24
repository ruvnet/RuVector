import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from validation_gate import official_validation,select,checkpoint_bytes,load_adapted


def evidence(rank,count=64):
    return dict(queryHashes=[str(i) for i in range(count)],ranks=[[rank,rank]]*count,
                metrics={"mrr":1/rank})


class ValidationGateTests(unittest.TestCase):
    def test_selects_only_clear_validation_gain(self):
        self.assertEqual(select(evidence(4),evidence(2))["selected"],"adapted")
        self.assertEqual(select(evidence(2),evidence(4))["selected"],"frozen")
        self.assertEqual(select(evidence(2),evidence(2))["selected"],"frozen")

    def test_small_sample_keeps_frozen(self):
        self.assertEqual(select(evidence(4,8),evidence(1,8))["selected"],"frozen")

    def test_unstable_mean_gain_keeps_frozen(self):
        candidate=evidence(100)
        candidate["ranks"][0]=[1,1]
        candidate["metrics"]["mrr"]=(1+63/100)/64
        result=select(evidence(50),candidate)
        self.assertGreater(result["relativeGain"],.1)
        self.assertLess(result["bootstrapLower"],0)
        self.assertEqual(result["selected"],"frozen")

    def test_mismatched_or_fabricated_evidence_rejected(self):
        a,b=evidence(2),evidence(1);b["queryHashes"][0]="other"
        with self.assertRaises(ValueError):select(a,b)
        b=evidence(1);b["metrics"]["mrr"]=.5
        with self.assertRaises(ValueError):select(a,b)

    def test_official_validation_never_uses_export_valid_rows(self):
        raw=b'a r b\nc r d\n';h=hashlib.sha256(raw).hexdigest()
        support=[dict(s="e",r="r",o="f",split="train")]
        queries,truths=official_validation(raw,h,support)
        self.assertEqual(len(queries),2)
        self.assertEqual(len(truths),3)
        with self.assertRaises(ValueError):official_validation(raw,"0"*64,support)
        with self.assertRaises(ValueError):official_validation(raw,h,[dict(s="a",r="r",o="b")])

    def test_corrupt_checkpoint_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights.pth";path.write_bytes(b"untrusted")
            with self.assertRaises(ValueError):checkpoint_bytes(path,"0"*64)
            with self.assertRaises(ValueError):checkpoint_bytes(path,"bad")


@unittest.skipUnless(os.environ.get("ULTRA_CHECKOUT"),"optional checkpoint integration")
class CheckpointTests(unittest.TestCase):
    def test_saved_weights_roundtrip_predictions(self):
        from ultra_adapter import UltraAdapter
        payload=dict(triples=[dict(s="a",r="r",o="b",split="train"),
                              dict(s="b",r="r",o="c",split="test")])
        model=UltraAdapter(payload,os.environ["ULTRA_CHECKOUT"])
        before=model.predict(dict(s="a",r="r",k=3))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights.pth"
            model.torch.save(model.model.state_dict(),path)
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            restored=UltraAdapter(payload,os.environ["ULTRA_CHECKOUT"],checkpoint=False,seed=9)
            load_adapted(restored,path,h)
            self.assertEqual(before,restored.predict(dict(s="a",r="r",k=3)))

    def test_rejected_candidate_and_error_restore_incumbent(self):
        from guarded_adaptation import apply_candidate
        from ultra_adapter import UltraAdapter
        from sparse_transfer import state_hash
        payload=dict(triples=[dict(s="a",r="r",o="b",split="train"),
                              dict(s="b",r="r",o="c",split="test")])
        model=UltraAdapter(payload,os.environ["ULTRA_CHECKOUT"])
        candidate=UltraAdapter(payload,os.environ["ULTRA_CHECKOUT"],checkpoint=False,seed=9)
        queries=[dict(s="b",r="r",o="a")]
        truths=[payload["triples"][0]]+queries
        before=state_hash(model)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights.pth"
            model.torch.save(candidate.model.state_dict(),path)
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            result=apply_candidate(model,path,h,queries,truths)
            self.assertEqual(result["decision"]["selected"],"frozen")
            self.assertEqual(state_hash(model),before)
            with self.assertRaises(ValueError):apply_candidate(model,path,"0"*64,queries,truths)
            self.assertEqual(state_hash(model),before)
            with patch("guarded_adaptation.evaluate_queries",side_effect=[evidence(2),RuntimeError("evaluation failed after loading")]):
                with self.assertRaises(RuntimeError):apply_candidate(model,path,h,queries,truths)
            self.assertEqual(state_hash(model),before)
            bad=dict(candidate.model.state_dict());bad.pop(next(iter(bad)))
            model.torch.save(bad,path)
            with self.assertRaises(ValueError):load_adapted(model,path,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(state_hash(model),before)


if __name__=="__main__":unittest.main()
