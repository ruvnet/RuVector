"""Optional CPU ULTRA adapter for RuVector {s,r,o,split} graph exports.

No network, provider calls, arbitrary checkpoint unpickling or automatic install.
ULTRA remains a separately installed, pinned MIT upstream implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ULTRA_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"
CHECKPOINT_SHA256 = "fdedc01b0045fc089d2ad5da08569466b7b221691fe978a18e91082bbb133c18"
PRETRAIN_GRAPHS = ["FB15k237", "WN18RR", "CoDExMedium"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_payload(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("triples"), list):
        raise ValueError("expected object with triples array")
    rows = payload["triples"]
    if not rows or len(rows) > 100000:
        raise ValueError("expected 1..100000 triples")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in ("train", "valid", "test"):
            raise ValueError("explicit train/valid/test split required")
        if any(not isinstance(row.get(k), str) or not row[k] or len(row[k]) > 1024 for k in ("s", "r", "o")):
            raise ValueError("nonempty bounded string labels required")
        key = (row["s"], row["r"], row["o"])
        if key in seen:
            raise ValueError("duplicate or cross-split overlapping triple")
        seen.add(key)
    if not all(any(x["split"] == split for x in rows) for split in ("train", "test")):
        raise ValueError("nonempty train and test required")
    entities = sorted({x[k] for x in rows for k in ("s", "o")})
    relations = sorted({x["r"] for x in rows})
    if len(entities) > 5000 or len(relations) > 512:
        raise ValueError("CPU pilot graph exceeds 5000 entities or 512 relations")
    train_relations = {x["r"] for x in rows if x["split"] == "train"}
    if train_relations != set(relations):
        raise ValueError("each target relation needs support edges")
    return rows, entities, relations


def filtered_rank(scores, target, truths):
    if not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError("nonfinite or empty scores")
    if target not in range(len(scores)):
        raise ValueError("target outside candidate universe")
    return 1 + sum(score >= scores[target] for i, score in enumerate(scores)
                   if i != target and i not in truths)


def metrics(ranks):
    if not ranks or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError("invalid ranks")
    return {"count": len(ranks), "mrr": sum(1 / r for r in ranks) / len(ranks),
            "hits1": sum(r == 1 for r in ranks) / len(ranks),
            "hits10": sum(r <= 10 for r in ranks) / len(ranks)}


class UltraAdapter:
    """Scores every candidate; vocabulary has no trainable ID embeddings.

    One instance is sequential only: upstream caches query tensors on its model.
    """
    def __init__(self, payload, upstream, checkpoint=True, seed=101):
        rows, self.entities, self.relations = validate_payload(payload)
        upstream = Path(upstream).resolve()
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
        if commit != ULTRA_COMMIT or dirty:
            raise ValueError("ULTRA checkout must match the pinned clean source")
        ckpt = upstream / "ckpts/ultra_3g.pth"
        if digest(ckpt) != CHECKPOINT_SHA256:
            raise ValueError("checkpoint SHA256 mismatch")
        sys.path.insert(0, str(upstream))
        import torch
        from torch_geometric.data import Data
        from ultra.models import Ultra
        from ultra import tasks
        import ultra.models
        if Path(ultra.models.__file__).resolve() != upstream / "ultra/models.py" or Path(tasks.__file__).resolve() != upstream / "ultra/tasks.py":
            raise ValueError("imported ULTRA module does not match pinned checkout")
        self.torch, self.Data, self.tasks = torch, Data, tasks
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        self.entity_ids = {x: i for i, x in enumerate(self.entities)}
        self.relation_ids = {x: i for i, x in enumerate(self.relations)}
        self.rows = rows
        self.graph = self.make_graph(self.entity_ids, self.relation_ids)
        cfg = dict(input_dim=64, hidden_dims=[64]*6, message_func="distmult",
                   aggregate_func="sum", short_cut=True, layer_norm=True)
        self.model = Ultra(dict(cfg, **{"class": "RelNBFNet"}),
                           dict(cfg, **{"class": "EntityNBFNet"}))
        if checkpoint:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

    def make_graph(self, entity_ids, relation_ids):
        triples = [(entity_ids[x["s"]], entity_ids[x["o"]], relation_ids[x["r"]])
                   for x in self.rows if x["split"] == "train"]
        triples = self.torch.tensor(triples, dtype=self.torch.long)
        edge_index = self.torch.cat([triples[:, :2].T, triples[:, :2].T.flip(0)], dim=1)
        edge_type = self.torch.cat([triples[:, 2], triples[:, 2]+len(relation_ids)])
        graph = self.Data(edge_index=edge_index, edge_type=edge_type,
                          num_nodes=len(entity_ids), num_relations=2*len(relation_ids))
        return self.tasks.build_relation_graph(graph)

    def scores(self, s, r, o, side="tail", graph=None, entity_ids=None, relation_ids=None):
        if side not in ("head", "tail"):
            raise ValueError("side must be head or tail")
        e, rel = entity_ids or self.entity_ids, relation_ids or self.relation_ids
        batch = self.torch.tensor([[e[s], e[o], rel[r]]])
        tail, head = self.tasks.all_negative(graph or self.graph, batch)
        with self.torch.inference_mode():
            scores = self.model(graph or self.graph, tail if side == "tail" else head)[0]
        result = scores.tolist()
        if not all(math.isfinite(x) for x in result):
            raise ValueError("model returned nonfinite scores")
        return result

    def predict(self, query):
        """RuVector predict-compatible query/result, higher score is better."""
        if ("s" in query) == ("o" in query):
            raise ValueError("provide exactly one of s and o")
        k = query.get("k", 10)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        # Dummy endpoint is used solely to construct all candidate triples.
        s, o = query.get("s", self.entities[0]), query.get("o", self.entities[0])
        scores = self.scores(s, query["r"], o, "tail" if "s" in query else "head")
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.entities[i]))[:k]
        return {"candidates": [{"entity": self.entities[i], "score": scores[i]} for i in order],
                "exact": True, "ann": False, "backend": "ultra-cpu"}

    def evaluate(self):
        # Ground truth labels enter ONLY these filtering maps, never make_graph.
        tails, heads = {}, {}
        for row in self.rows:
            tails.setdefault((row["s"], row["r"]), set()).add(self.entity_ids[row["o"]])
            heads.setdefault((row["o"], row["r"]), set()).add(self.entity_ids[row["s"]])
        ranks, query_ids = [], []
        start = time.perf_counter()
        for row in self.rows:
            if row["split"] != "test":
                continue
            pair = []
            for side, target, truth in (("tail", row["o"], tails[(row["s"], row["r"])]),
                                        ("head", row["s"], heads[(row["o"], row["r"])])):
                scores = self.scores(row["s"], row["r"], row["o"], side)
                pair.append(filtered_rank(scores, self.entity_ids[target], truth))
            ranks.append(pair)
            query_ids.append(hashlib.sha256(json.dumps([row["s"], row["r"], row["o"]]).encode()).hexdigest())
        return {"metrics": metrics([r for pair in ranks for r in pair]), "ranks": ranks,
                "queryHashes": query_ids, "seconds": time.perf_counter()-start}

    def permutation_check(self):
        import random
        rng = random.Random(314159)
        ep, rp = list(range(len(self.entities))), list(range(len(self.relations)))
        rng.shuffle(ep)
        rng.shuffle(rp)
        e = dict(zip(self.entities, ep))
        r = dict(zip(self.relations, rp))
        graph = self.make_graph(e, r)
        worst, count = 0.0, 0
        for row in [x for x in self.rows if x["split"] == "test"][:5]:
            for side in ("head", "tail"):
                count += 1
                before = self.scores(row["s"], row["r"], row["o"], side)
                after = self.scores(row["s"], row["r"], row["o"], side, graph, e, r)
                aligned = [after[e[label]] for label in self.entities]
                worst = max(worst, max(abs(a-b) for a, b in zip(before, aligned)))
                if not self.torch.allclose(self.torch.tensor(before), self.torch.tensor(aligned), atol=1e-4, rtol=1e-4):
                    raise ValueError("entity/relation permutation changed scores")
        return {"maxAbsoluteError": worst, "atol": 1e-4, "rtol": 1e-4, "queries": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", help='JSON query, e.g. {"s":"a","r":"r","k":5}')
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    model = UltraAdapter(payload, args.upstream)
    if args.query:
        result = model.predict(json.loads(args.query))
    else:
        # Freeze protocol before any evaluation. No test-driven tuning.
        result = {"schema": "ruvector.ultra-transfer.v1", "inputSha256": digest(args.input),
                  "adapterSha256": digest(__file__), "upstreamCommit": ULTRA_COMMIT,
                  "checkpointSha256": CHECKPOINT_SHA256, "declaredPretrainingGraphs": PRETRAIN_GRAPHS,
                  "graph": payload.get("name"), "entities": len(model.entities),
                  "relations": len(model.relations), "targetTrainingSteps": 0,
                  "candidatePolicy": "all-entities", "ties": "pessimistic",
                  "permutation": model.permutation_check(), "pretrained": model.evaluate(),
                  "random": [], "promotion": False}
        for seed in (101, 202, 303):
            random_model = UltraAdapter(payload, args.upstream, checkpoint=False, seed=seed)
            result["random"].append({"seed": seed, **random_model.evaluate()})
        result["limits"] = "Exploratory external checkpoint transfer. Pretraining provenance is upstream-declared. Random is not target-trained scratch. No replay comparison or generalization claim."
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": args.output, "status": "completed"}))


if __name__ == "__main__":
    main()
