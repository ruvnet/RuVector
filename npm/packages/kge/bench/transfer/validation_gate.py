"""Select frozen/adapted ULTRA using explicit validation facts, never test rows.

Trusted local checkpoint receipts are required. Hashes provide integrity,
not authenticity. This is an offline experimental selector, not a service.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
from ultra_adapter import filtered_rank, metrics

POLICY = dict(minimumFacts=32, maximumFacts=64, relativeGain=.10,
              bootstrapReplicates=2000, bootstrapSeed=271828,
              rule="adapted only if validation MRR >=1.1*frozen and paired bootstrap 2.5 percentile gain >0",
              filtering="retained support plus official validation only; no test truths",
              limits="Bootstrap is a descriptive stability guard, not proof of independent graph generalization")


def triple_key(row):
    return tuple(row[k] for k in ("s", "r", "o"))


def query_hash(row):
    return hashlib.sha256(json.dumps(list(triple_key(row))).encode()).hexdigest()


def official_validation(raw, expected_sha, support):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("official validation hash mismatch")
    rows = []
    seen = {triple_key(r) for r in support}
    for line in raw.decode().splitlines():
        fields = line.split()
        if len(fields) != 3 or any(len(x)>1024 for x in fields):
            raise ValueError("invalid validation triple")
        s,r,o = fields
        if (s,r,o) in seen:
            raise ValueError("validation duplicate or support overlap")
        seen.add((s,r,o))
        rows.append(dict(s=s,r=r,o=o))
    queries = sorted(rows,key=query_hash)[:POLICY["maximumFacts"]]
    return queries, support+rows


def evaluate_queries(adapter, queries, truths):
    """All candidates come from adapter vocabulary; filtering is explicitly scoped."""
    if not queries or len({triple_key(r) for r in queries}) != len(queries):
        raise ValueError("empty or duplicate queries")
    tails,heads = {},{}
    for row in truths:
        tails.setdefault((row["s"],row["r"]),set()).add(adapter.entity_ids[row["o"]])
        heads.setdefault((row["o"],row["r"]),set()).add(adapter.entity_ids[row["s"]])
    ranks=[]
    for row in queries:
        pair=[]
        for side,target,truth in (("tail",row["o"],tails[(row["s"],row["r"])]),
                                  ("head",row["s"],heads[(row["o"],row["r"])])):
            scores=adapter.scores(row["s"],row["r"],row["o"],side)
            pair.append(filtered_rank(scores,adapter.entity_ids[target],truth))
        ranks.append(pair)
    return dict(queryHashes=[query_hash(r) for r in queries],ranks=ranks,
                metrics=metrics([rank for pair in ranks for rank in pair]))


def reciprocal_pairs(evaluation):
    ids,pairs = evaluation["queryHashes"],evaluation["ranks"]
    if len(ids) != len(pairs) or len(set(ids)) != len(ids) or not pairs:
        raise ValueError("missing or duplicate query evidence")
    if any(len(pair)!=2 or any(type(r) is not int or r<1 for r in pair) for pair in pairs):
        raise ValueError("invalid ranks")
    values=[(1/pair[0]+1/pair[1])/2 for pair in pairs]
    supplied=evaluation["metrics"]["mrr"]
    if not math.isfinite(supplied) or abs(statistics.mean(values)-supplied)>1e-12:
        raise ValueError("metric differs from ranks")
    return values


def select(frozen, adapted):
    if frozen["queryHashes"] != adapted["queryHashes"]:
        raise ValueError("unpaired validation queries")
    baseline,candidate=reciprocal_pairs(frozen),reciprocal_pairs(adapted)
    if len(baseline)<POLICY["minimumFacts"]:
        return dict(selected="frozen",reason="insufficient validation facts",facts=len(baseline))
    delta=[a-b for a,b in zip(candidate,baseline)]
    rng=random.Random(POLICY["bootstrapSeed"])
    means=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(POLICY["bootstrapReplicates"]))
    lower=means[int(.025*len(means))]
    relative=statistics.mean(candidate)/statistics.mean(baseline)-1
    accepted=statistics.mean(candidate)>=1.1*statistics.mean(baseline) and lower>0
    return dict(selected="adapted" if accepted else "frozen",facts=len(delta),
                reason="validation gain and stability passed" if accepted else "validation gain or stability failed",
                relativeGain=relative,pairedMeanGain=statistics.mean(delta),bootstrapLower=lower)


def checkpoint_bytes(path, expected_sha):
    if not isinstance(expected_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",expected_sha):
        raise ValueError("expected SHA256 required")
    with Path(path).open("rb") as stream:
        raw=stream.read(64*1024*1024+1)
    if len(raw)>64*1024*1024 or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError("checkpoint size or hash mismatch")
    return raw


def load_adapted(adapter,path,expected_sha):
    raw=checkpoint_bytes(path,expected_sha)
    state=adapter.torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if not isinstance(state,dict) or any(not isinstance(v,adapter.torch.Tensor) or not adapter.torch.isfinite(v).all()
                                         for v in state.values()):
        raise ValueError("checkpoint requires finite tensors")
    expected=adapter.model.state_dict()
    if state.keys()!=expected.keys() or any(state[k].shape!=v.shape or state[k].dtype!=v.dtype
                                            for k,v in expected.items()):
        raise ValueError("checkpoint schema differs; model unchanged")
    adapter.model.load_state_dict(state,strict=True)
    adapter.model.eval()
