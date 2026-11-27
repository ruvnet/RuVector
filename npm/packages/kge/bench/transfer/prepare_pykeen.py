"""Export one PyKEEN bundled dataset to explicit RuVector split-tagged triples."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from ultra_adapter import validate_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--dataset", choices=("nations", "kinships", "umls"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.checkout).resolve()
    rows, hashes = [], {}
    for split in ("train", "valid", "test"):
        path = root / "src/pykeen/datasets" / args.dataset / (split+".txt")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            s, r, o = line.split()
            rows.append(dict(s=s, r=r, o=o, split=split))
    test = sorted([x for x in rows if x["split"] == "test"], key=lambda x:
                  hashlib.sha256(json.dumps([x["s"], x["r"], x["o"]]).encode()).hexdigest())
    keep = {tuple(x[k] for k in ("s", "r", "o")) for x in test[:64]}
    for row in rows:
        if row["split"] == "test" and tuple(row[k] for k in ("s", "r", "o")) not in keep:
            row["split"] = "valid"
    payload = dict(name=args.dataset, source="https://github.com/pykeen/pykeen",
                   sourceCommit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                   fileHashes=hashes,
                   selection="first 64 test triples by sha256 of JSON [s,r,o]; other test facts filter-only valid",
                   originalTestCount=len(test), triples=rows)
    validate_payload(payload)
    Path(args.output).write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
