#!/usr/bin/env bash
# CLI ↔ Rust parity check, in two halves.
#
# `npm/packages/rvforge` (TypeScript) writes `.rvf` containers and a local
# registry; `crates/rvf-forge-core` and `crates/rvforge-registry` (Rust) read
# them. Both sides implement the same specs and were built independently — so
# "both pass their own tests" says nothing about whether they interoperate.
#
# **Container parity.** `rvforge create` writes a real, signed `.rvf`, and
# `rvforge-rvf-check` inspects and verifies it through `rvf-forge-core`'s public
# API, with the publisher's public key supplied so the Ed25519 signatures are
# actually checked rather than skipped. Nothing is shared between the two
# implementations but the format itself: the segment layout, the SHAKE-256
# content hashes, the signature footers, and the Level-0 root manifest page all
# have to agree byte for byte or this fails.
#
# **Registry parity.** The run then drives pack → publish against that same
# artifact, publishes a *second* release so it exercises lineage rather than a
# single object, and hands the directory to `rvforge-registry-check`, which
# re-derives every content address, recomputes the Merkle log, re-applies the
# publication rules, and walks the witness chains.
#
# Prints `PARITY OK` and exits 0 when the Rust readers accept what the CLI
# wrote; otherwise prints the violation list and exits non-zero.
#
# Usage: scripts/rvforge-parity-check.sh [--keep]
#   --keep   leave the temporary registry on disk and print its path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_DIR="$REPO_ROOT/npm/packages/rvforge"
CRATE_DIR="$REPO_ROOT/crates/rvforge-registry"

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

for required in "$CLI_DIR/package.json" "$CRATE_DIR/Cargo.toml"; do
  if [[ ! -f "$required" ]]; then
    echo "parity check needs $required; skipping" >&2
    exit 0
  fi
done

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/rvforge-parity.XXXXXX")"
cleanup() {
  if [[ "$KEEP" -eq 1 ]]; then
    echo "kept: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

PROJECT_DIR="$WORKDIR/project"
REGISTRY_DIR="$WORKDIR/registry"
mkdir -p "$PROJECT_DIR"

echo "==> building the CLI"
# --workspaces=false: the package sits inside the npm/ workspace root, and a
# plain install resolves the whole workspace and fails EBADPLATFORM on
# platform-pinned siblings.
(cd "$CLI_DIR" && npm install --workspaces=false --silent && npm run build --silent)

CLI="$CLI_DIR/dist/cli.js"
KEY_FILE="$PROJECT_DIR/rvforge-publisher.key"

echo "==> rvforge init --keygen"
# init scaffolds rvforge.json too, so it runs first; the fixture generator then
# replaces that scaffold with a project that passes every pack check.
(cd "$PROJECT_DIR" && node "$CLI" init --keygen --quiet >/dev/null)
node "$REPO_ROOT/scripts/rvforge-parity-fixture.cjs" "$CLI_DIR/dist" "$PROJECT_DIR" >/dev/null

echo "==> rvforge create agent.rvf"
# The fixture no longer ships a synthetic .rvf: the CLI writes the artifact
# under test. A .wasm payload makes it carry an executable segment, so the
# run covers the signed-executable path rather than metadata alone.
mkdir -p "$PROJECT_DIR/payload"
printf '\0asm\1\0\0\0' > "$PROJECT_DIR/payload/agent.wasm"
(cd "$PROJECT_DIR" && node "$CLI" create agent.rvf \
  --from payload --key-file "$KEY_FILE" --quiet >/dev/null)

echo "==> rvforge-rvf-check (Rust reads the container the CLI wrote)"
# The public key is passed as hex so the Rust side needs no base64 decoder;
# without it every signature check would record "skipped", which would prove
# nothing about whether the CLI's signatures verify.
PUBLIC_KEY_HEX="$(node -e '
  const { readFileSync } = require("node:fs");
  const key = JSON.parse(readFileSync(process.argv[1], "utf8"));
  process.stdout.write(Buffer.from(key.publicKey, "base64").toString("hex"));
' "$KEY_FILE")"

if ! (cd "$REPO_ROOT" && cargo run --quiet -p rvf-forge-core \
        --bin rvforge-rvf-check -- "$PROJECT_DIR/agent.rvf" \
        --public-key "$PUBLIC_KEY_HEX"); then
  echo
  echo "PARITY FAILED — crates/rvf-forge-core rejected the .rvf that" >&2
  echo "npm/packages/rvforge wrote; the two disagree about the RVF format" >&2
  exit 1
fi

echo "==> rvforge pack agent.rvf"
(cd "$PROJECT_DIR" && node "$CLI" pack agent.rvf --quiet >/dev/null)

echo "==> rvforge publish agent.rvf (1.0.0)"
(cd "$PROJECT_DIR" && node "$CLI" publish agent.rvf \
  --registry "$REGISTRY_DIR" --key-file "$KEY_FILE" --quiet >/dev/null)

# A second release is what makes the check meaningful: it exercises the
# predecessor link, the release index as a chain, a two-leaf Merkle tree, and a
# witness receipt that has to name the previous one.
echo "==> rvforge publish agent.rvf (1.1.0)"
node -e '
  const { readFileSync, writeFileSync } = require("node:fs");
  const path = process.argv[1];
  const project = JSON.parse(readFileSync(path, "utf8"));
  project.version = "1.1.0";
  writeFileSync(path, `${JSON.stringify(project, null, 2)}\n`);
' "$PROJECT_DIR/rvforge.json"
(cd "$PROJECT_DIR" && node "$CLI" publish agent.rvf \
  --registry "$REGISTRY_DIR" --key-file "$KEY_FILE" --quiet >/dev/null)

echo "==> rvforge-registry-check (Rust)"
if (cd "$REPO_ROOT" && cargo run --quiet -p rvforge-registry \
      --bin rvforge-registry-check -- "$REGISTRY_DIR"); then
  echo
  echo "PARITY OK — the Rust side reads everything the CLI wrote:"
  echo "  container  rvf-forge-core inspected and verified agent.rvf"
  echo "  registry   rvforge-registry replayed the whole publication log"
  exit 0
fi

echo
echo "PARITY FAILED — the violations above are places the CLI and" >&2
echo "crates/rvforge-registry disagree about registry-model.md" >&2
exit 1
