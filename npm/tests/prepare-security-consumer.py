"""Project the changed packages' exact lock closure for a scripts-disabled smoke test.
Usage: python3 npm/tests/prepare-security-consumer.py /new/temporary/directory
Then: cd /new/temporary/directory && npm ci --ignore-scripts && node consumer.cjs
This validates only the affected dependency closure, not the full workspace.
"""
import json
import shutil
import sys
from pathlib import Path

npm = Path(__file__).resolve().parents[1]
packages = json.loads((npm / "package-lock.json").read_text())["packages"]
names = ("@modelcontextprotocol/sdk", "axios", "compression", "fast-copy",
         "ip-address", "proxy-addr", "qs")
target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=False)
selected = {}
pending = ["node_modules/" + name for name in names]

def resolve(at, name):
    parts = at.split("/")
    for i in range(len(parts), -1, -1):
        if i and parts[i - 1] == "node_modules":
            continue
        candidate = "/".join(parts[:i] + ["node_modules", name])
        if candidate in packages:
            return candidate
    return None

while pending:
    key = pending.pop()
    if key in selected:
        continue
    entry = packages[key]
    if entry.get("link"):
        raise ValueError("Workspace link outside dependency test scope")
    selected[key] = entry
    for group in ("dependencies", "optionalDependencies", "peerDependencies"):
        for name in entry.get(group, {}):
            resolved = resolve(key, name)
            if resolved:
                pending.append(resolved)
            elif group == "dependencies":
                raise ValueError(f"Missing required dependency: {key} -> {name}")

dependencies = {name: packages["node_modules/" + name]["version"] for name in names}
manifest = {"name": "ruvector-security-consumer", "version": "1.0.0",
            "private": True, "dependencies": dependencies,
            "overrides": {"qs": dependencies["qs"]}}
lock = {"name": manifest["name"], "version": "1.0.0", "lockfileVersion": 3,
        "requires": True, "packages": {"": manifest, **dict(sorted(selected.items()))}}
(target / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
(target / "package-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
shutil.copyfile(npm / "tests/security-dependency-consumer.cjs", target / "consumer.cjs")
print(f"Projected {len(selected)} exact dependency records; install scripts remain disabled.")
