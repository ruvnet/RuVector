"""Development-only check of longer adaptation against retaining frozen ULTRA.

Reuse the frozen sparse trainer without changing its source or earlier results.
Fail the development gate before consuming a fresh confirmation graph.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from sparse_transfer import sparse_payload, state_hash, train
from ultra_adapter import UltraAdapter, digest

PROTOCOL = dict(graphs=["nations", "kinships", "umls"], fraction=.25,
                seeds=[101, 202, 303], seconds=20., maxSteps=1000,
                minimumRelativeGain=.10, timeTolerance=.10,
                decision="Each graph must improve mean MRR >=10% over frozen and scratch; all training durations within 10% of 20s",
                role="Development only. No fresh graph accessed unless this gate passes.",
                limitations="Selected support fraction; existing diagnostic queries; no KGE comparison or promotion")


def identity(inputs):
    payloads = [json.loads(p.read_text()) for p in inputs]
    if sorted(p["name"] for p in payloads) != sorted(PROTOCOL["graphs"]):
        raise ValueError("exactly the three development graphs are required")
    return dict(protocol=PROTOCOL,
                inputHashes={str(p.resolve()):digest(p) for p in inputs},
                sourceHashes={n:digest(Path(__file__).with_name(n)) for n in
                              ("adaptation_budget.py", "sparse_transfer.py", "ultra_adapter.py")})


def summarize(receipt):
    if receipt.get("complete") is not True or receipt.get("promotion") is not False:
        raise ValueError("incomplete or promoting receipt")
    if receipt["manifest"]["protocol"] != PROTOCOL:
        raise ValueError("protocol differs")
    rows = receipt["records"]
    if len(rows) != 21 or {r["graph"] for r in rows} != set(PROTOCOL["graphs"]):
        raise ValueError("wrong record count or graph set")
    results = {}
    for graph in PROTOCOL["graphs"]:
        group = [r for r in rows if r["graph"] == graph]
        expected = {("frozen", None)} | {(a,s) for a in ("adapted", "scratch") for s in PROTOCOL["seeds"]}
        if len(group) != 7 or {(r["arm"],r["seed"]) for r in group} != expected:
            raise ValueError("missing or duplicate arm")
        if len({r["inputHash"] for r in group}) != 1:
            raise ValueError("incompatible input")
        reference = group[0]["evaluation"]["queryHashes"]
        for row in group:
            ev = row["evaluation"]
            if ev["queryHashes"] != reference or len(reference) != 64:
                raise ValueError("incompatible queries")
            ranks = [rank for pair in ev["ranks"] for rank in pair]
            if len(ranks) != 128 or any(not isinstance(x,int) or x < 1 for x in ranks):
                raise ValueError("invalid ranks")
            mrr = ev["metrics"]["mrr"]
            if not math.isfinite(mrr) or abs(statistics.mean(1/x for x in ranks)-mrr) > 1e-12:
                raise ValueError("MRR does not match ranks")
        means = {a:statistics.mean(r["evaluation"]["metrics"]["mrr"] for r in group if r["arm"]==a)
                 for a in ("frozen", "adapted", "scratch")}
        gains = {a:means["adapted"]/means[a]-1 for a in ("frozen", "scratch")}
        mismatches = []
        for row in group:
            if row["arm"] == "frozen":
                continue
            seconds = row["training"]["seconds"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid timing")
            if not 18 <= seconds <= 22:
                mismatches.append(dict(arm=row["arm"],seed=row["seed"],seconds=seconds))
        quality = all(means["adapted"] >= 1.1*means[a] for a in ("frozen", "scratch"))
        results[graph] = dict(mrr=means,relativeGains=gains,timingMismatches=mismatches,
                              qualityPassed=quality,developmentPassed=quality and not mismatches)
    return dict(graphs=results,developmentPassed=all(g["developmentPassed"] for g in results.values()),
                promotion=False,confirmationRun=False)


def execute(args, manifest):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(manifest=manifest,records=[],complete=False,promotion=False)
    for path in args.inputs:
        payload = sparse_payload(json.loads(path.read_text()), PROTOCOL["fraction"])
        graph = payload["name"]
        data = root/(graph+".json")
        data.write_text(json.dumps(payload,indent=2)+"\n")
        frozen = UltraAdapter(payload,args.upstream)
        receipt["records"].append(dict(graph=graph,arm="frozen",seed=None,inputHash=digest(data),
                                        support=payload["support"],evaluation=frozen.evaluate()))
        for seed in PROTOCOL["seeds"]:
            for arm in (["adapted","scratch"] if seed % 2 else ["scratch","adapted"]):
                model = UltraAdapter(payload,args.upstream,checkpoint=arm=="adapted",seed=seed)
                training = train(model,seed,PROTOCOL["seconds"],PROTOCOL["maxSteps"])
                checkpoint = root/f"{graph}-{arm}-{seed}.pth"
                model.torch.save(model.model.state_dict(),checkpoint)
                evaluation = model.evaluate()
                if state_hash(model) != training["finalHash"]:
                    raise ValueError("evaluation changed weights")
                row = dict(graph=graph,arm=arm,seed=seed,inputHash=digest(data),support=payload["support"],
                           training=training,checkpointHash=digest(checkpoint),evaluation=evaluation)
                receipt["records"].append(row)
                (root/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
                print(json.dumps(dict(graph=graph,arm=arm,seed=seed,steps=training["steps"],
                                      seconds=training["seconds"],mrr=evaluation["metrics"]["mrr"])),flush=True)
    receipt["complete"] = True
    result = summarize(receipt)
    (root/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    (root/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,nargs="+",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--upstream")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose freeze or run")
    current = identity(args.inputs)
    if args.freeze:
        with args.manifest.open("x") as out:
            json.dump(current,out,indent=2)
    else:
        if current != json.loads(args.manifest.read_text()):
            raise ValueError("frozen source or input identity changed")
        if not args.upstream or not args.output:
            parser.error("run requires upstream and output")
        execute(args,current)


if __name__ == "__main__":
    main()
