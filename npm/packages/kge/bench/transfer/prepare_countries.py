"""Convert downloaded, pinned Countries S1 TSV files without network access.

Download train.txt, valid.txt and test.txt from BASE_URL into --directory.
The resulting JSON must match the sparse manifest input hash before evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from ultra_adapter import validate_payload

SOURCE_COMMIT="3f8c12defd62c94cc512a693228906b019ff73b9"
BASE_URL=f"https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/{SOURCE_COMMIT}/Countries/Countries_S1/"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows,hashes,seen=[],{},{}
    duplicates=0
    for split in ("train","valid","test"):
        data=(args.directory/(split+".txt")).read_bytes()
        hashes[split]=hashlib.sha256(data).hexdigest()
        for line in data.decode().splitlines():
            s,r,o=line.split()
            key=(s,r,o)
            if key in seen:
                if seen[key] != split:
                    raise ValueError("cross-split overlap")
                duplicates+=1
                continue
            seen[key]=split
            rows.append(dict(s=s,r=r,o=o,split=split))
    payload=dict(name="countries_s1",source=BASE_URL,sourceCommit=SOURCE_COMMIT,fileHashes=hashes,
                 selection="all official test facts; untouched whole-graph check",triples=rows,
                 normalization={"removedExactWithinSplitDuplicates":duplicates,"crossSplitOverlap":"rejected"})
    validate_payload(payload)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":
    main()
