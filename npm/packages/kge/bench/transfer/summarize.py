"""Join hash-matched ULTRA / native KGE receipts; never assert promotion."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import statistics
import subprocess


def decision(summaries):
    if not summaries:
        raise ValueError("no graph summaries")
    if all(x["pretrainedMRR"] > x["randomMeanMRR"] for x in summaries):
        if all(x["pretrainedMRR"] < x["kgeScratchMeanMRR"] for x in summaries):
            return "EXPLORATORY_TRANSFER_SIGNAL_NOT_SUPERIOR_TO_TARGET_TRAINED_KGE"
        return "EXPLORATORY_TRANSFER_SIGNAL_REQUIRES_CONFIRMATION"
    return "TRANSFER_NOT_CONSISTENTLY_DEMONSTRATED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binding", required=True, help="Measured native .node binary")
    args = parser.parse_args()
    root = Path(args.results)
    graphs = []
    for name in ("nations", "kinships", "umls"):
        raw = (root / (name+".json")).read_bytes()
        data = json.loads(raw)
        ultra = json.loads((root / (name+"-ultra.json")).read_text())
        kge = json.loads((root / (name+"-kge.json")).read_text())
        expected = hashlib.sha256(raw).hexdigest()
        if ultra["inputSha256"] != expected or kge["inputSha256"] != expected:
            raise ValueError("comparison input hash mismatch")
        for arm in ultra["random"]:
            if arm["queryHashes"] != ultra["pretrained"]["queryHashes"]:
                raise ValueError("paired query mismatch")
        # Bootstrap original triples: preserve head/tail pairs and average the
        # three random controls first. Conditional query uncertainty, not graph
        # or pretraining-seed uncertainty; shared graph queries are dependent.
        def reciprocal(pair):
            return statistics.mean(1/r for r in pair)
        delta = [reciprocal(pair)-statistics.mean(reciprocal(a["ranks"][i]) for a in ultra["random"])
                 for i, pair in enumerate(ultra["pretrained"]["ranks"])]
        rng = random.Random(8675309)
        boot = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
        graphs.append({"dataset": {k:v for k,v in data.items() if k != "triples"},
                       "summary": {"pretrainedMRR": ultra["pretrained"]["metrics"]["mrr"],
                                   "randomMeanMRR": statistics.mean(a["metrics"]["mrr"] for a in ultra["random"]),
                                   "kgeScratchMeanMRR": statistics.mean(a["evaluation"]["report"]["combined"]["mrr"] for a in kge["arms"]),
                                   "pretrainedMinusRandom": statistics.mean(delta),
                                   "conditionalQueryBootstrap95": [boot[50],boot[1949]]},
                       "ultra": ultra, "kge": kge})
    result = {"schema":"ruvector.cross-graph-pilot.v1", "graphs":graphs,
              "host": {"os":platform.system(), "arch":platform.machine(), "python":platform.python_version()},
              "nativeBinarySha256":hashlib.sha256(Path(args.binding).read_bytes()).hexdigest(),
              "dependencyFreeze":subprocess.check_output([__import__('sys').executable,"-m","pip","freeze"],text=True).splitlines(),
              "decision":decision([g["summary"] for g in graphs]), "promotion":False,
              "limits":["64 selected test facts per graph, not full benchmark scores", "Three small graphs; no production graph evaluated",
                        "Random weights are not trained ULTRA scratch", "No replay or equal target fine-tuning budget comparison",
                        "Bootstrap is descriptive conditional query uncertainty; graph facts are dependent",
                        "No provider calls; existing CPU resources, prior pretraining cost excluded"]}
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for graph in graphs:
        print(graph["dataset"]["name"], json.dumps(graph["summary"]))


if __name__ == "__main__":
    main()
