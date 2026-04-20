#!/usr/bin/env bash
# Install the obsidian-brain plugin into a local vault.
# Usage:
#   ./scripts/setup.sh <path/to/vault>
# Requires: node >= 18, npm.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VAULT="${1:-}"

if [[ -z "$VAULT" ]]; then
  echo "Usage: $0 <path/to/vault>" >&2
  exit 2
fi
if [[ ! -d "$VAULT/.obsidian" ]]; then
  echo "Not an Obsidian vault: $VAULT (missing .obsidian/)" >&2
  exit 2
fi

echo "==> Installing npm deps in $HERE"
cd "$HERE"
npm install --no-audit --fund=false

echo "==> Building plugin (production)"
npm run build

DEST="$VAULT/.obsidian/plugins/obsidian-brain"
mkdir -p "$DEST"
cp main.js manifest.json styles.css versions.json "$DEST/"

echo "==> Installed to $DEST"
echo "Next:"
echo "  1. In Obsidian, enable 'RuVector Brain' under Community plugins."
echo "  2. Make sure the local brain ($(grep -E '^[[:space:]]*ExecStart' "$HERE/systemd/ruvector-brain.service" | head -1)) and embedder are running."
echo "  3. Settings → RuVector Brain → Test connection."
