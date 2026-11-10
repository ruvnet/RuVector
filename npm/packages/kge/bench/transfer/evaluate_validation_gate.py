"""Development evaluation of validation-only selection using existing weights."""
import argparse
import json
from pathlib import Path
import statistics
from adaptation_budget import summarize as verify_development
from sparse_transfer import state_hash
from ultra_adapter import UltraAdapter,digest
from validation_gate import POLICY,official_validation,evaluate_queries,select,load_adapted


def manifest(args):
    receipt=json.loads((args.run/"receipt.json").read_text())
    verify_development(receipt)
    inputs={}
    for graph in receipt["manifest"]["protocol"]["graphs"]:
        data=args.run/(graph+".json")
        payload=json.loads(data.read_text())
        rows=[r for r in receipt["records"] if r["graph"]==graph]
        if any(digest(data)!=r["inputHash"] for r in rows):
            raise ValueError("support input differs from trained checkpoint receipt")
        valid=args.pykeen/"src/pykeen/datasets"/graph/"valid.txt"
        if digest(valid)!=payload["fileHashes"]["valid"]:
            raise ValueError("official validation source differs")
        inputs[graph]=dict(inputHash=digest(data),validHash=digest(valid),
                           checkpoints={str(r["seed"]):r["checkpointHash"] for r in rows if r["arm"]=="adapted"})
        for seed,h in inputs[graph]["checkpoints"].items():
            if digest(args.run/f"{graph}-adapted-{seed}.pth")!=h:
                raise ValueError("checkpoint differs from training receipt")
    return dict(policy=POLICY,inputs=inputs,trainingReceiptHash=digest(args.run/"receipt.json"),
                sourceHashes={n:digest(Path(__file__).with_name(n)) for n in
                              ("validation_gate.py","evaluate_validation_gate.py","ultra_adapter.py","sparse_transfer.py","adaptation_budget.py")})


def execute(args,frozen_manifest):
    receipt=json.loads((args.run/"receipt.json").read_text())
    records=[]
    for graph,identity in frozen_manifest["inputs"].items():
        payload=json.loads((args.run/(graph+".json")).read_text())
        support=[r for r in payload["triples"] if r["split"]=="train"]
        raw=(args.pykeen/"src/pykeen/datasets"/graph/"valid.txt").read_bytes()
        queries,truths=official_validation(raw,identity["validHash"],support)
        # Vocabulary remains fixed, but no filter-only/test labels enter selection.
        model=UltraAdapter(payload,args.upstream)
        baseline=evaluate_queries(model,queries,truths)
        for seed,h in identity["checkpoints"].items():
            load_adapted(model,args.run/f"{graph}-adapted-{seed}.pth",h)
            before=state_hash(model)
            candidate=evaluate_queries(model,queries,truths)
            if state_hash(model)!=before:
                raise ValueError("validation mutated weights")
            decision=select(baseline,candidate)
            # Decision is already final before diagnostic test metrics are read.
            selected=next(r for r in receipt["records"] if r["graph"]==graph and r["arm"]==decision["selected"]
                          and r["seed"]==(None if decision["selected"]=="frozen" else int(seed)))
            records.append(dict(graph=graph,seed=int(seed),decision=decision,
                                validationFrozen=baseline,validationAdapted=candidate,
                                selectedCheckpointHash=h if decision["selected"]=="adapted" else None,
                                diagnosticTestMRR=selected["evaluation"]["metrics"]["mrr"]))
            print(json.dumps(dict(graph=graph,seed=int(seed),**decision)),flush=True)
    summary={}
    for graph in frozen_manifest["inputs"]:
        selected=[r for r in records if r["graph"]==graph]
        source=[r for r in receipt["records"] if r["graph"]==graph]
        summary[graph]=dict(selectedAdapted=sum(r["decision"]["selected"]=="adapted" for r in selected),
                            selectedTestMRR=statistics.mean(r["diagnosticTestMRR"] for r in selected),
                            frozenTestMRR=next(r["evaluation"]["metrics"]["mrr"] for r in source if r["arm"]=="frozen"),
                            adaptedTestMRR=statistics.mean(r["evaluation"]["metrics"]["mrr"] for r in source if r["arm"]=="adapted"))
    result=dict(manifest=frozen_manifest,records=records,summary=summary,promotion=False,confirmationRun=False,
                limits="Development policy selected after prior test inspection. Cached test metrics are diagnostic only. No fresh transfer claim.")
    with args.output.open("x") as out:
        json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--pykeen",type=Path,required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    current=manifest(args)
    if args.freeze:
        with args.manifest.open("x") as out: json.dump(current,out,indent=2)
    else:
        if not args.upstream or not args.output or args.output.exists():
            parser.error("requires upstream and a new output file")
        if current!=json.loads(args.manifest.read_text()):
            raise ValueError("frozen manifest differs")
        execute(args,current)


if __name__=="__main__":
    main()
