"""Fail-closed completion and acceptance checks for the frozen sparse pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

FRACTIONS = [.01,.05,.1,.25]
SEEDS = [101,202,303]
ARMS = ["frozen","adapted","scratch","kge50","kgeTimed"]


def validate_cell(records):
    expected = {("frozen",None)} | {(arm,seed) for arm in ARMS[1:] for seed in SEEDS}
    actual = [(r["arm"],r["seed"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("missing, duplicate or unexpected arm/seed")
    counts = {r["evaluation"]["metrics"]["count"] for r in records}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("incompatible query counts")
    if len({r["support"]["sha256"] for r in records}) != 1:
        raise ValueError("support mismatch")
    query_hashes = [r["evaluation"]["queryHashes"] for r in records if r["arm"] in ("frozen","adapted","scratch")]
    if any(q != query_hashes[0] for q in query_hashes):
        raise ValueError("query mismatch")
    input_hashes = {r["inputHash"] for r in records if r["arm"] != "frozen"}
    if len(input_hashes) != 1:
        raise ValueError("input mismatch")
    for row in records:
        m = row["evaluation"]["metrics"]["mrr"]
        if not isinstance(m,(int,float)) or not math.isfinite(m) or not 0 < m <= 1:
            raise ValueError("invalid MRR")
        if row["arm"] != "frozen":
            t = row["training"]["seconds"]
            if not isinstance(t,(int,float)) or not math.isfinite(t) or t <= 0:
                raise ValueError("invalid training duration")


def summarize(receipt):
    if receipt.get("complete") is not True or receipt.get("promotion") is not False:
        raise ValueError("receipt must be complete and non-promoting")
    expected_graphs = {Path(p).stem for p in receipt["manifest"]["inputHashes"]}
    actual_graphs = {r["graph"] for r in receipt["records"]}
    if actual_graphs != expected_graphs:
        raise ValueError("graph set differs from manifest")
    output = {}
    for graph in sorted(expected_graphs):
        all_rows = [r for r in receipt["records"] if r["graph"] == graph]
        if len(all_rows) != 52:
            raise ValueError("expected exactly 52 records per graph")
        table=[]
        for fraction in FRACTIONS:
            rows=[r for r in all_rows if r["fraction"]==fraction]
            validate_cell(rows)
            means={arm:statistics.mean(r["evaluation"]["metrics"]["mrr"] for r in rows if r["arm"]==arm) for arm in ARMS}
            time_rows=[r for r in rows if r["arm"] in ("adapted","scratch","kgeTimed")]
            mismatches=[dict(arm=r["arm"],seed=r["seed"],seconds=r["training"]["seconds"])
                        for r in time_rows if abs(r["training"]["seconds"]-2)/2 > .1]
            table.append(dict(nominalFraction=fraction, support=rows[0]["support"], mrr=means,
                              timingMismatches=mismatches,
                              trainingSeconds={arm:statistics.mean(r["training"]["seconds"] for r in rows if r["arm"]==arm) for arm in ARMS[1:]}))
        aggregate={arm:statistics.mean(t["mrr"][arm] for t in table) for arm in ARMS}
        relative={arm:aggregate["adapted"]/aggregate[arm]-1 for arm in ("scratch","kge50","kgeTimed")}
        positive_all=all(t["mrr"]["adapted"] > t["mrr"]["scratch"] for t in table)
        time_match=all(not t["timingMismatches"] for t in table)
        quality=positive_all and all(g >= .1 for g in relative.values())
        output[graph]=dict(table=table,aggregateMRR=aggregate,relativeGains=relative,
                          allFractionsBeatScratch=positive_all,timingMatched=time_match,
                          qualityGate=quality,acceptancePassed=quality and time_match)
    return dict(graphs=output,confirmationGraph="countries_s1",
                confirmationPassed=output.get("countries_s1",{}).get("acceptancePassed",False),
                promotion=False,
                limits=["A capped wall-time experiment, not equal FLOPs or equal examples",
                        "Coverage floors can exceed nominal fractions; use recorded actual fractions",
                        "Countries is a new whole-graph check with only 24 test facts, not production evidence",
                        "Diagnostic graphs reuse prior 64 queries; no independent confirmation claim for them",
                        "No cross-graph source replay arm; evaluation filters know withheld truths, training does not"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raw=args.receipt.read_bytes()
    receipt=json.loads(raw)
    result=summarize(receipt)
    # Describe information availability without changing the frozen decision.
    for name, graph in result["graphs"].items():
        for cell in graph["table"]:
            fraction=cell["nominalFraction"]
            path=args.receipt.parent/f"{name}-{round(100*fraction)}.json"
            raw_input=path.read_bytes()
            expected=next(r["inputHash"] for r in receipt["records"]
                          if r["graph"]==name and r["fraction"]==fraction and r["arm"]!="frozen")
            if hashlib.sha256(raw_input).hexdigest()!=expected:
                raise ValueError("support coverage input mismatch")
            rows=json.loads(raw_input)["triples"]
            observed={r[k] for r in rows if r["split"]=="train" for k in ("s","o")}
            universe={r[k] for r in rows for k in ("s","o")}
            test=[r for r in rows if r["split"]=="test"]
            cell["coverage"]=dict(supportedNodes=len(observed),allNodes=len(universe),
                                  testFactsBothEndpointsSupported=sum(r["s"] in observed and r["o"] in observed for r in test),
                                  testFacts=len(test))
    result["receiptSha256"]=hashlib.sha256(raw).hexdigest()
    result["manifest"]=receipt["manifest"]
    result["records"]=receipt["records"]
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for graph,value in result["graphs"].items():
        print(graph,json.dumps({k:v for k,v in value.items() if k != "table"}))


if __name__=="__main__":
    main()
