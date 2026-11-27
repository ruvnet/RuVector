import copy
import unittest
from adaptation_budget import PROTOCOL, summarize


def fixture():
    records=[]
    for graph in PROTOCOL["graphs"]:
        for arm in ("frozen","adapted","scratch"):
            for seed in ([None] if arm=="frozen" else PROTOCOL["seeds"]):
                rank=1 if arm=="adapted" else 2
                records.append(dict(graph=graph,arm=arm,seed=seed,inputHash="same",
                                    training={"seconds":20},evaluation=dict(
                                        queryHashes=[str(x) for x in range(64)],ranks=[[rank,rank]]*64,
                                        metrics={"mrr":1/rank})))
    return dict(complete=True,promotion=False,manifest={"protocol":copy.deepcopy(PROTOCOL)},records=records)


class DevelopmentGateTests(unittest.TestCase):
    def test_passing_development_never_promotes(self):
        result=summarize(fixture())
        self.assertTrue(result["developmentPassed"])
        self.assertFalse(result["promotion"])
        self.assertFalse(result["confirmationRun"])

    def test_frozen_comparator_blocks_even_when_scratch_loses(self):
        data=fixture()
        row=next(r for r in data["records"] if r["arm"]=="frozen")
        row["evaluation"]["ranks"]=[[1,1]]*64
        row["evaluation"]["metrics"]["mrr"]=1.
        self.assertFalse(summarize(data)["developmentPassed"])

    def test_timing_and_missing_evidence_block(self):
        data=fixture();data["records"][-1]["training"]["seconds"]=23
        self.assertFalse(summarize(data)["developmentPassed"])
        data["records"].pop()
        with self.assertRaises(ValueError): summarize(data)

    def test_tampered_metric_and_query_rejected(self):
        data=fixture();data["records"][0]["evaluation"]["metrics"]["mrr"]+=.1
        with self.assertRaises(ValueError): summarize(data)
        data=fixture();data["records"][0]["evaluation"]["queryHashes"][0]="changed"
        with self.assertRaises(ValueError): summarize(data)


if __name__ == "__main__":
    unittest.main()
