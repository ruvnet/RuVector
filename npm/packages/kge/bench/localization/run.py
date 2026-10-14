"""Freeze and run a CPU SWERank retrieval pilot against a lexical control."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
import statistics
import time
from retrieval import Encoder,MODEL_REVISION,CODE_REVISION,PREFIX,bm25,digest,evaluate,file_ranking

PROTOCOL=dict(maxTokens=512,batchSize=8,threads=2,scoring="normalized CLS dot product; maximum chunk score per file",
              baseline="BM25 k1=1.5 b=.75 over identical chunks",candidatePolicy="all production Python files under declared package prefixes",
              scope="six development smoke cases; no reranker, graph scoring, fine tuning or transfer confirmation",
              license="SWERank model card declares CC BY NC 4.0; research comparator, no commercial deployment claim",
              promotion=False)


def manifest(args):
    data=json.loads((args.data/"data-manifest.json").read_text())
    if len(data["cases"])!=6:raise ValueError("expected six selected cases")
    for case in data["cases"]:
        if not re.fullmatch(r"[a-zA-Z0-9_]+-[0-9]+",case["id"]):raise ValueError("invalid instance id")
        for suffix,key in [(".json","inputHash"),(".labels.json","labelHash")]:
            if digest(args.data/(case["id"]+suffix))!=case[key]:raise ValueError("case identity differs")
    artifacts={str(p.resolve()):digest(p) for root in [args.model,args.code] for p in root.iterdir() if p.is_file()}
    required=[args.model/n for n in ["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.txt"]]
    required += [args.code/n for n in ["configuration_hf_nomic_bert.py","modeling_hf_nomic_bert.py"]]
    if any(str(p.resolve()) not in artifacts for p in required):raise ValueError("missing pinned artifact")
    return dict(protocol=PROTOCOL,modelRevision=MODEL_REVISION,codeRevision=CODE_REVISION,data=data,
                artifacts=artifacts,sourceHashes={n:digest(Path(__file__).with_name(n)) for n in ["prepare.py","retrieval.py","run.py"]},
                versions={n:importlib.metadata.version(n) for n in ["torch","transformers","huggingface-hub","einops","safetensors","pyarrow"]})


def execute(args,frozen):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    model=Encoder(args.model,args.code,frozen["artifacts"],PROTOCOL["maxTokens"])
    initialization=time.perf_counter()-start
    records=[]
    for item in frozen["data"]["cases"]:
        case=json.loads((args.data/(item["id"]+".json")).read_text())
        if set(case)!={"id","repo","baseCommit","issue","documents"}:raise ValueError("unexpected encoder input fields")
        documents=case["documents"]
        t=time.perf_counter();lexical=file_ranking(documents,bm25(case["issue"],documents));lexical_time=time.perf_counter()-t
        embeddings,indexing=model.encode([d["text"] for d in documents])
        query,query_time=model.encode([PREFIX+case["issue"]])
        t=time.perf_counter();dense=file_ranking(documents,(embeddings@query[0]).tolist());ranking_time=time.perf_counter()-t
        predictions=dict(bm25=lexical,swerank=dense)
        # Persist unscored predictions before loading any patch-derived labels.
        output=args.output/(case["id"]+".predictions.json")
        output.write_text(json.dumps(predictions,indent=2,allow_nan=False)+"\n")
        labels=json.loads((args.data/(case["id"]+".labels.json")).read_text())
        if labels["id"]!=case["id"]:raise ValueError("label mismatch")
        row=dict(id=case["id"],repo=case["repo"],baseCommit=case["baseCommit"],inputHash=item["inputHash"],
                 labelHash=item["labelHash"],predictionHash=digest(output),documents=len(documents),files=len(dense),
                 fallbackChunks=sum(d.get("kind")=="syntax-fallback" for d in documents),
                 bm25=evaluate(lexical,labels["goldFiles"]),swerank=evaluate(dense,labels["goldFiles"]),
                 indexing=indexing,queryEncoding=query_time,rankingSeconds=ranking_time,bm25Seconds=lexical_time)
        records.append(row)
        print(json.dumps(row),flush=True)
    aggregate={arm:{metric:statistics.mean(r[arm][metric] for r in records) for metric in ["mrr","hit1","recall5","recall10"]}
               for arm in ["bm25","swerank"]}
    result=dict(manifest=frozen,records=records,aggregate=aggregate,initializationSeconds=initialization,
                promotion=False,limits="Six preselected public benchmark cases. Training overlap unknown. 512-token float32 CPU variant, not official paper reproduction. No reranker or RuVector graph arm yet.")
    (args.output/"receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(aggregate),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["data","model","code","manifest"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--freeze",action="store_true");p.add_argument("--output",type=Path)
    args=p.parse_args();current=manifest(args)
    if args.freeze:
        with args.manifest.open("x") as out:json.dump(current,out,indent=2)
    else:
        if not args.output:p.error("output required")
        if current!=json.loads(args.manifest.read_text()):raise ValueError("frozen identities differ")
        execute(args,current)


if __name__=="__main__":main()
