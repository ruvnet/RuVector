import copy
import os
import unittest
from sparse_transfer import sparse_payload, state_hash, train
from ultra_adapter import UltraAdapter
from summarize_sparse import summarize


def fixture():
    rows=[dict(s=f"n{i}",r="r",o=f"n{i+1}",split="train") for i in range(20)]
    rows += [dict(s="n0",r="q",o="n20",split="train"),
             dict(s="n1",r="q",o="n19",split="valid"),
             dict(s="n0",r="r",o="n2",split="test")]
    return dict(name="synthetic-test",triples=rows)


class SparseContractTests(unittest.TestCase):
    def test_nested_and_coverage_floor(self):
        previous=set()
        for fraction in [.01,.05,.1,.25]:
            data=sparse_payload(fixture(),fraction)
            support={(r["s"],r["r"],r["o"]) for r in data["triples"] if r["split"]=="train"}
            self.assertTrue(previous <= support)
            self.assertEqual({r[1] for r in support},{"q","r"})
            self.assertEqual(len(support),data["support"]["count"])
            self.assertEqual(sum(r["split"]=="test" for r in data["triples"]),1)
            previous=support
        self.assertGreater(sparse_payload(fixture(),.01)["support"]["actualFraction"],.01)

    def test_validation_cannot_change_support(self):
        a=fixture(); b=copy.deepcopy(a)
        b["triples"][-2]["o"]="n18"
        self.assertEqual(sparse_payload(a,.25)["support"],sparse_payload(b,.25)["support"])


class ReceiptGateTests(unittest.TestCase):
    def fixture(self):
        records=[]
        for fraction in [.01,.05,.1,.25]:
            for arm in ["frozen","adapted","scratch","kge50","kgeTimed"]:
                for seed in ([None] if arm=="frozen" else [101,202,303]):
                    records.append(dict(graph="countries_s1",fraction=fraction,arm=arm,seed=seed,
                                        support={"sha256":"same"},inputHash="same",
                                        training={"seconds":.01 if arm=="kge50" else 2},
                                        evaluation={"queryHashes":["query"],"metrics":{"count":2,
                                            "mrr":.5 if arm=="adapted" else .4}}))
        return dict(complete=True,promotion=False,records=records,
                    manifest={"inputHashes":{"/data/countries_s1.json":"hash"}})

    def test_fixed_epoch_control_excluded_from_time_gate(self):
        self.assertTrue(summarize(self.fixture())["confirmationPassed"])

    def test_timing_mismatch_blocks_acceptance(self):
        data=self.fixture()
        next(r for r in data["records"] if r["arm"]=="kgeTimed")["training"]["seconds"]=3
        result=summarize(data)["graphs"]["countries_s1"]
        self.assertTrue(result["qualityGate"])
        self.assertFalse(result["acceptancePassed"])

    def test_missing_and_nonfinite_evidence_rejected(self):
        data=self.fixture(); data["records"].pop()
        with self.assertRaises(ValueError): summarize(data)
        data=self.fixture(); data["records"][0]["evaluation"]["metrics"]["mrr"]=float("nan")
        with self.assertRaises(ValueError): summarize(data)

    def test_one_bad_fraction_blocks_acceptance(self):
        data=self.fixture()
        for r in data["records"]:
            if r["arm"]=="adapted" and r["fraction"]==.01:
                r["evaluation"]["metrics"]["mrr"]=.3
        self.assertFalse(summarize(data)["confirmationPassed"])


@unittest.skipUnless(os.environ.get("ULTRA_CHECKOUT"),"optional training integration")
class TrainingTests(unittest.TestCase):
    def test_real_gradient_update_and_truth_isolation(self):
        a=fixture(); b=copy.deepcopy(a)
        b["triples"][-2]["o"]="n18"
        outputs=[]
        for payload in (a,b):
            model=UltraAdapter(payload,os.environ["ULTRA_CHECKOUT"],checkpoint=False,seed=17)
            before=state_hash(model)
            record=train(model,seed=17,seconds=30,max_steps=2)
            self.assertEqual(record["steps"],2)
            self.assertNotEqual(before,state_hash(model))
            outputs.append(record["finalHash"])
        self.assertEqual(*outputs)


if __name__=="__main__":
    unittest.main()
