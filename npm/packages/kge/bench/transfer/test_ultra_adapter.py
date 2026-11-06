import copy
import math
import os
import unittest
from ultra_adapter import validate_payload, filtered_rank, metrics, UltraAdapter
from summarize import decision


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"triples": [dict(s="a", r="r", o="b", split="train"),
                                    dict(s="b", r="r", o="c", split="test")]}

    def test_explicit_splits(self):
        rows, nodes, rels = validate_payload(self.payload)
        self.assertEqual((len(rows), nodes, rels), (2, ["a", "b", "c"], ["r"]))

    def test_leakage_and_duplicate_rejected(self):
        for split in ("test", "train"):
            payload = copy.deepcopy(self.payload)
            payload["triples"].append(dict(s="a", r="r", o="b", split=split))
            with self.assertRaises(ValueError):
                validate_payload(payload)

    def test_unlabelled_or_unsupported_relations_rejected(self):
        for field, value in (("split", "transfer"), ("r", "unseen-in-support"), ("s", None)):
            payload = copy.deepcopy(self.payload)
            payload["triples"][1][field] = value
            with self.assertRaises(ValueError):
                validate_payload(payload)

    def test_filter_and_pessimistic_ties(self):
        self.assertEqual(filtered_rank([9, 5, 5, 2], 1, {0, 1}), 2)
        self.assertEqual(filtered_rank([9, 5, 5, 2], 1, {0, 1, 2}), 1)
        self.assertEqual(filtered_rank([5], 0, set()), 1)

    def test_nonfinite_rejected(self):
        for v in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                filtered_rank([v, 1], 1, set())
        with self.assertRaises(ValueError):
            metrics([])

    def test_metrics(self):
        self.assertEqual(metrics([1, 2])["mrr"], .75)

    def test_decision_follows_evidence(self):
        result = dict(pretrainedMRR=.5, randomMeanMRR=.2, kgeScratchMeanMRR=.8)
        self.assertIn("NOT_SUPERIOR", decision([result]))
        result["randomMeanMRR"] = .6
        self.assertEqual(decision([result]), "TRANSFER_NOT_CONSISTENTLY_DEMONSTRATED")
        result.update(randomMeanMRR=.2, kgeScratchMeanMRR=.4)
        self.assertEqual(decision([result]), "EXPLORATORY_TRANSFER_SIGNAL_REQUIRES_CONFIRMATION")


@unittest.skipUnless(os.environ.get("ULTRA_CHECKOUT"), "optional ULTRA runtime not requested")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"triples": [dict(s="a", r="r", o="b", split="train"),
                                    dict(s="b", r="r", o="c", split="train"),
                                    dict(s="b", r="q", o="a", split="train"),
                                    dict(s="c", r="q", o="b", split="valid"),
                                    dict(s="a", r="r", o="c", split="test")]}
        self.model = UltraAdapter(self.payload, os.environ["ULTRA_CHECKOUT"])

    def test_truth_isolation(self):
        altered = copy.deepcopy(self.payload)
        altered["triples"][3] = dict(s="a", r="q", o="c", split="valid")
        other = UltraAdapter(altered, os.environ["ULTRA_CHECKOUT"])
        torch = self.model.torch
        self.assertTrue(torch.equal(self.model.graph.edge_index, other.graph.edge_index))
        self.assertTrue(torch.equal(self.model.graph.relation_graph.edge_index, other.graph.relation_graph.edge_index))
        self.assertEqual(self.model.scores("a", "r", "c"), other.scores("a", "r", "c"))
        self.assertEqual(self.model.graph.num_edges, 6)

    def test_prediction_and_permutation(self):
        for query, side in ((dict(s="a", r="r", k=3), "tail"), (dict(o="c", r="r", k=3), "head")):
            result = self.model.predict(query)
            scores = self.model.scores("a", "r", "c", side)
            expected = sorted(zip(self.model.entities, scores), key=lambda x: (-x[1], x[0]))
            self.assertEqual([(x["entity"], x["score"]) for x in result["candidates"]], expected)
        self.assertEqual(self.model.permutation_check()["queries"], 2)

    def test_unknown_label_and_invalid_query_fail(self):
        for query in ({"s":"missing", "r":"r"}, {"s":"a", "o":"c", "r":"r"}, {"s":"a", "r":"r", "k":0}):
            with self.assertRaises((KeyError, ValueError)):
                self.model.predict(query)


if __name__ == "__main__":
    unittest.main()
