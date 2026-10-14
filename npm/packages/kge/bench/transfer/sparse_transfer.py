"""Bounded sparse-support transfer experiment; freeze manifest before --run.

Primary comparison: pretrained versus scratch ULTRA under the same wall-time
ceiling, not equal FLOPs. No test-based model selection or automatic promotion.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
from ultra_adapter import UltraAdapter, digest, validate_payload

FRACTIONS = [.01, .05, .10, .25]
SEEDS = [101, 202, 303]
PROTOCOL = dict(fractions=FRACTIONS, seeds=SEEDS, seconds=2.0, maxSteps=1000, trainingThreads=1,
                batchSize=8, negatives=16, lr=.0005, weightDecay=.01,
                primary="equal-weight mean adapted-minus-scratch MRR across four fractions and three seeds",
                acceptance="relative aggregate MRR gain >= 10% versus scratch ULTRA AND both KGE controls; all four fractions positive versus scratch; adapted/scratch/kgeTimed durations within 10% of 2 seconds (kge50 excluded)",
                training="support-only strict negatives; remove batch positive edges and inverse edges before rebuilding relation graph; BCE equal positive/negative group weights",
                selection="nested hash sample with one support fact per relation floor",
                querySelection="fixed prior 64 SHA-selected facts on diagnostic graphs; all 24 official Countries S1 test facts for confirmation",
                limits="Timing checked between updates permits one-step overrun. Wall time is not equal compute. No production promotion.")


def key(row):
    return (row["s"], row["r"], row["o"])


def sparse_payload(payload, fraction):
    if fraction not in FRACTIONS:
        raise ValueError("fraction outside frozen protocol")
    validate_payload(payload)
    train = sorted((r for r in payload["triples"] if r["split"] == "train"),
                   key=lambda r: hashlib.sha256(json.dumps(key(r)).encode()).hexdigest())
    covered, keep = set(), set()
    for row in train:
        if row["r"] not in covered:
            covered.add(row["r"])
            keep.add(key(row))
    count = max(len(keep), math.ceil(fraction*len(train)))
    for row in train:
        if len(keep) >= count:
            break
        keep.add(key(row))
    result = copy.deepcopy(payload)
    for row in result["triples"]:
        if row["split"] == "train" and key(row) not in keep:
            row["split"] = "valid"  # evaluation filtering only; never tuning
    result["support"] = dict(nominalFraction=fraction, actualFraction=len(keep)/len(train),
                             count=len(keep), originalCount=len(train), relationFloor=len(covered),
                             sha256=hashlib.sha256(json.dumps(sorted(keep)).encode()).hexdigest())
    return result


def state_hash(adapter):
    h = hashlib.sha256()
    for name, tensor in sorted(adapter.model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(adapter, seed, seconds=2.0, max_steps=1000):
    preparation_started = time.perf_counter()
    if not 0 < seconds <= 60 or not 1 <= max_steps <= 1000:
        raise ValueError("invalid bounded training budget")
    torch = adapter.torch
    # Single-thread rspmm backward avoids OpenMP reduction order variation.
    torch.set_num_threads(PROTOCOL["trainingThreads"])
    from torch.nn import functional as F
    triples = [(adapter.entity_ids[r["s"]], adapter.entity_ids[r["o"]], adapter.relation_ids[r["r"]])
               for r in adapter.rows if r["split"] == "train"]
    tails, heads = {}, {}
    for s, o, r in triples:
        tails.setdefault((s, r), set()).add(o)
        heads.setdefault((o, r), set()).add(s)
    universe = set(range(len(adapter.entities)))
    candidates = [(t, sorted(universe-tails[(t[0], t[2])]), sorted(universe-heads[(t[1], t[2])]))
                  for t in triples]
    candidates = [x for x in candidates if x[1] and x[2]]
    if not candidates:
        raise ValueError("no valid support-only negative samples")
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weightDecay"])
    rng = random.Random(seed)
    # This architecture has LayerNorm, no dropout or BatchNorm. eval() disables
    # upstream's second edge removal; autograd and optimizer updates remain on.
    adapter.model.eval()
    initial = state_hash(adapter)
    losses, batch_hash = [], hashlib.sha256()
    preparation_seconds = time.perf_counter()-preparation_started
    started = time.perf_counter()
    while len(losses) < max_steps and time.perf_counter()-started < seconds:
        selected = [rng.choice(candidates) for _ in range(PROTOCOL["batchSize"])]
        positive = {x[0] for x in selected}
        # Both real and synthetic inverse query edges are removed before
        # constructing the relation graph. Other support facts remain visible.
        remaining = [t for t in triples if t not in positive]
        if not remaining:
            raise ValueError("batch removed the entire support graph")
        edges = torch.tensor(remaining, dtype=torch.long)
        graph = adapter.Data(edge_index=torch.cat([edges[:, :2].T, edges[:, :2].T.flip(0)], dim=1),
                             edge_type=torch.cat([edges[:, 2], edges[:, 2]+len(adapter.relations)]),
                             num_nodes=len(adapter.entities), num_relations=2*len(adapter.relations))
        graph = adapter.tasks.build_relation_graph(graph)
        batch = []
        for i, ((s, o, r), neg_t, neg_h) in enumerate(selected):
            negatives = rng.choices(neg_t if i < len(selected)//2 else neg_h, k=PROTOCOL["negatives"])
            batch.append([(s, o, r)]+[(s, n, r) if i < len(selected)//2 else (n, o, r) for n in negatives])
        batch_hash.update(json.dumps(batch).encode())
        optimizer.zero_grad(set_to_none=True)
        scores = adapter.model(graph, torch.tensor(batch, dtype=torch.long))
        target = torch.zeros_like(scores)
        target[:, 0] = 1
        bce = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
        loss = ((bce[:, 0]+bce[:, 1:].mean(dim=1))/2).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.model.parameters()):
            raise ValueError("nonfinite gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter()-started
    final = state_hash(adapter)
    if not losses or final == initial:
        raise ValueError("no verified parameter update")
    return dict(steps=len(losses), examples=len(losses)*PROTOCOL["batchSize"], seconds=elapsed,
                preparationSeconds=preparation_seconds,
                secondsBudget=seconds, overrun=max(0, elapsed-seconds), lastLoss=losses[-1],
                initialHash=initial, finalHash=final, batchesHash=batch_hash.hexdigest(),
                eligibleSupport=len(candidates))


def manifest(inputs, binding, native):
    return dict(protocol=PROTOCOL, inputHashes={str(p.resolve()):digest(p) for p in inputs},
                binding={"wrapperPath":str(Path(binding).resolve()), "wrapperHash":digest(binding),
                         "nativePath":str(Path(native).resolve()), "nativeHash":digest(native)},
                sourceHashes={name:digest(Path(__file__).with_name(name)) for name in
                              ("sparse_transfer.py", "ultra_adapter.py", "sparse_kge.mjs")})


def execute(args, frozen):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        for fraction in FRACTIONS:
            subset = sparse_payload(payload, fraction)
            stem = f"{payload['name']}-{round(fraction*100)}"
            data_path = root/(stem+".json")
            data_path.write_text(json.dumps(subset, indent=2)+"\n")
            frozen_model = UltraAdapter(subset, args.upstream)
            zero = frozen_model.evaluate()
            records.append(dict(graph=payload["name"], fraction=fraction, support=subset["support"],
                                arm="frozen", seed=None, evaluation=zero))
            for seed in SEEDS:
                # Counterbalance order by seed, without using outcomes.
                order = ["adapted", "scratch"] if seed % 2 else ["scratch", "adapted"]
                for arm in order:
                    setup = time.perf_counter()
                    model = UltraAdapter(subset, args.upstream, checkpoint=arm=="adapted", seed=seed)
                    setup_seconds = time.perf_counter()-setup
                    training = train(model, seed, PROTOCOL["seconds"], PROTOCOL["maxSteps"])
                    # Freeze before evaluation; no checkpoint selection.
                    checkpoint = root/f"{stem}-{arm}-{seed}.pth"
                    model.torch.save(model.model.state_dict(), checkpoint)
                    evaluation = model.evaluate()
                    if state_hash(model) != training["finalHash"]:
                        raise ValueError("evaluation mutated parameters")
                    records.append(dict(graph=payload["name"], fraction=fraction, support=subset["support"],
                                        arm=arm, seed=seed, setupSeconds=setup_seconds,
                                        inputHash=digest(data_path), training=training,
                                        checkpointHash=digest(checkpoint), evaluation=evaluation))
                    print(json.dumps({"graph":payload["name"], "fraction":fraction,"arm":arm,"seed":seed,
                                      "steps":training["steps"],"mrr":evaluation["metrics"]["mrr"]}), flush=True)
            kge_output = root/(stem+"-kge.json")
            subprocess.run(["node", str(Path(__file__).with_name("sparse_kge.mjs")), str(data_path),
                            args.binding, str(kge_output), args.native], check=True)
            kge = json.loads(kge_output.read_text())
            if kge["inputHash"] != digest(data_path):
                raise ValueError("KGE input mismatch")
            for arm in kge["arms"]:
                records.append(dict(graph=payload["name"], fraction=fraction, support=subset["support"],
                                    inputHash=digest(data_path), engineVersion=kge["engineVersion"],
                                    calibration=kge["calibration"], **arm))
            (root/"receipt.json").write_text(json.dumps(dict(manifest=frozen, records=records, complete=False, promotion=False),indent=2)+"\n")
    receipt = dict(manifest=frozen, records=records, complete=True, promotion=False)
    (root/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--upstream")
    parser.add_argument("--binding")
    parser.add_argument("--native")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose exactly one of --freeze or --run")
    if not args.binding or not args.native:
        parser.error("binding and native paths required for provenance")
    current = manifest(args.inputs, args.binding, args.native)
    if args.freeze:
        with args.manifest.open("x") as out:
            json.dump(current, out, indent=2)
        print(digest(args.manifest))
    else:
        frozen = json.loads(args.manifest.read_text())
        if current != frozen:
            raise ValueError("frozen manifest or source/input identity changed")
        if not all((args.upstream, args.binding, args.output)):
            parser.error("run requires upstream, binding and output")
        execute(args, frozen)


if __name__ == "__main__":
    main()
