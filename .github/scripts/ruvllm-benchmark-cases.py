#!/usr/bin/env python3
"""Prepare native Criterion executables, enumerate cases, or run one exact case.

Full characterization is opt-in: one case per job retains Criterion's configured
sample count while excluding compilation from the 30/45 minute measurement budget.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BENCHES = ("matmul_bench", "attention_bench", "ane_bench")


def case_names(binary):
    result = subprocess.run([str(binary.resolve()), "--list"], check=True,
                            capture_output=True, text=True)
    names = [line.removesuffix(": benchmark") for line in result.stdout.splitlines()
             if line.endswith(": benchmark")]
    if not names or len(names) != len(set(names)) or len(names) > 256:
        raise ValueError(f"Invalid case inventory for {binary}: {len(names)} (limit 256)")
    return names


def prepare(features):
    benches = BENCHES if features else BENCHES[:2]
    command = ["cargo", "bench", "--locked", "-p", "ruvllm", "--no-run",
               "--message-format=json"]
    for bench in benches:
        command.extend(["--bench", bench])
    if features:
        command.extend(["--features", features])
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    directory = Path("benchmark-binaries")
    directory.mkdir(exist_ok=True)
    executables = {}
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            name = message["target"]["name"]
            if name in benches:
                executables[name] = message["executable"]
    if set(executables) != set(benches):
        raise ValueError(f"Missing benchmark executables: {executables}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for bench in benches:
            binary = directory / bench
            shutil.copy2(executables[bench], binary)
            names = case_names(binary)
            (directory / f"{bench}.cases.json").write_text(json.dumps(names, indent=2))
            matrix = {"include": [{"case": name} for name in names]}
            output.write(f"{bench}={json.dumps(matrix, separators=(',', ':'))}\n")
            print(f"{bench}: {len(names)} independently budgeted cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--features", default="")
    parser.add_argument("--bench", choices=BENCHES)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.features)
    else:
        if not args.bench:
            parser.error("run requires --bench")
        binary = Path("benchmark-binaries") / args.bench
        name = os.environ["BENCHMARK_CASE"]
        inventory = json.loads((binary.parent / f"{args.bench}.cases.json").read_text())
        if name not in inventory:
            raise ValueError("Requested case is absent from the built inventory")
        binary.chmod(binary.stat().st_mode | 0o100)
        subprocess.run([str(binary.resolve()), "--bench", f"^{re.escape(name)}$",
                        "--output-format", "bencher"], check=True)


if __name__ == "__main__":
    main()
