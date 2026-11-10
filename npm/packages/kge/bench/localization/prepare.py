"""Prepare six deterministic development cases without running repository code."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import tarfile
import urllib.request

DATA_REVISION="6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
REPOSITORIES={"psf/requests":"requests/","pallets/flask":"src/flask/"}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def chunks(path,source):
    lines=source.splitlines()
    result=[dict(id=path+":module",path=path,text=path+"\n"+"\n".join(lines[:80]))]
    try:tree=ast.parse(source)
    except SyntaxError:
        # Older snapshots can contain Python 2 syntax. Preserve every file,
        # using explicit raw windows instead of silently dropping candidates.
        return [dict(id=f"{path}:raw:{start+1}",path=path,kind="syntax-fallback",
                     text=path+"\n"+"\n".join(lines[start:start+80])) for start in range(0,len(lines),80)]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            result.append(dict(id=f"{path}:{node.lineno}",path=path,
                               text=path+"\n"+"\n".join(lines[start:node.end_lineno])))
    return result


def snapshot_documents(raw,prefix):
    documents=[];files={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
        for member in archive.getmembers():
            parts=PurePosixPath(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                raise ValueError("unsafe archive path")
            path="/".join(parts[1:])
            if not member.isfile() or not path.startswith(prefix) or not path.endswith(".py"):
                continue
            if member.size>2_000_000:raise ValueError("source file too large")
            data=archive.extractfile(member).read()
            source=data.decode("utf-8-sig")
            if path in files:raise ValueError("duplicate archive path")
            files[path]=sha(data)
            documents.extend(chunks(path,source))
    if not files or len(documents)>5000:raise ValueError("invalid bounded corpus")
    return sorted(documents,key=lambda d:d["id"]),dict(sorted(files.items()))


def gold_files(patch,prefix):
    paths=[]
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            path=line[6:]
            if path.startswith(prefix) and path.endswith(".py"):paths.append(path)
    return sorted(set(paths))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import pyarrow.parquet as pq
    rows=pq.read_table(args.parquet).to_pylist()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(dataset="princeton-nlp/SWE-bench_Lite",revision=DATA_REVISION,
                  parquetHash=sha(args.parquet.read_bytes()),selection="first 3 instance_id SHA256 per declared repository",
                  role="Development smoke pilot; pretrained overlap unknown; no transfer confirmation",cases=[])
    for repo,prefix in REPOSITORIES.items():
        selected=sorted([r for r in rows if r["repo"]==repo],key=lambda r:sha(r["instance_id"].encode()))[:3]
        if len(selected)!=3:raise ValueError("missing selected cases")
        for row in selected:
            if not re.fullmatch(r"[a-zA-Z0-9_]+-[0-9]+",row["instance_id"]):raise ValueError("invalid instance id")
            commit=row["base_commit"]
            if not re.fullmatch(r"[0-9a-f]{40}",commit):raise ValueError("invalid commit")
            url=f"https://codeload.github.com/{repo}/tar.gz/{commit}"
            with urllib.request.urlopen(url,timeout=60) as response:raw=response.read(50_000_001)
            if len(raw)>50_000_000:raise ValueError("archive too large")
            documents,files=snapshot_documents(raw,prefix)
            case=dict(id=row["instance_id"],repo=repo,baseCommit=commit,issue=row["problem_statement"],documents=documents)
            # Label material stays in a separate file, unavailable to encoders.
            labels=dict(id=row["instance_id"],goldFiles=gold_files(row["patch"],prefix))
            if not labels["goldFiles"] or not set(labels["goldFiles"])<=set(files):
                raise ValueError("target outside complete production file corpus")
            data=args.output/(case["id"]+".json")
            target=args.output/(case["id"]+".labels.json")
            data.write_text(json.dumps(case,indent=2)+"\n");target.write_text(json.dumps(labels,indent=2)+"\n")
            manifest["cases"].append(dict(id=case["id"],repo=repo,baseCommit=commit,snapshotUrl=url,
                archiveHash=sha(raw),sourceFileHashes=files,inputHash=sha(data.read_bytes()),labelHash=sha(target.read_bytes()),
                documentCount=len(documents),fileCount=len(files)))
            print(case["id"],len(files),len(documents),flush=True)
    (args.output/"data-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
