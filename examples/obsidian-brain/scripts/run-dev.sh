#!/usr/bin/env bash
# Launch a live Obsidian dev session:
#   1. Spin up the local brain (mcp-brain-server-local) + a mock embedder
#   2. Create/refresh a scratch vault at $VAULT with the built plugin
#      pre-installed, enabled, and pointed at the scratch brain
#   3. Launch the real Obsidian app
#
# Usage:
#   ./scripts/run-dev.sh [path/to/vault]
#
# Defaults:
#   VAULT=$HOME/obsidian-brain-vault

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VAULT="${1:-$HOME/obsidian-brain-vault}"
BRAIN_PORT="${BRAIN_PORT:-19876}"
EMBED_PORT="${EMBED_PORT:-19877}"

BRAIN_BIN="${RUVBRAIN_BIN:-$REPO/target/release/mcp-brain-server-local}"
if [[ ! -x "$BRAIN_BIN" ]]; then
  echo "==> Building mcp-brain-server-local (release)"
  (cd "$REPO" && cargo build --release -p mcp-brain-server --features local --bin mcp-brain-server-local)
fi

# Make sure the plugin bundle is up to date.
if [[ ! -f "$HERE/main.js" ]]; then
  echo "==> Building obsidian-brain plugin"
  (cd "$HERE" && npm install --no-audit --fund=false && npm run build)
fi

APPIMAGE_DIR="$HOME/.cache/obsidian-brain-e2e/Obsidian-1.6.5-extracted/squashfs-root"
if [[ ! -x "$APPIMAGE_DIR/obsidian" ]]; then
  CACHE="$HOME/.cache/obsidian-brain-e2e"
  mkdir -p "$CACHE"
  if [[ ! -f "$CACHE/Obsidian-1.6.5.AppImage" ]]; then
    echo "==> Downloading Obsidian 1.6.5 AppImage"
    curl -fL -o "$CACHE/Obsidian-1.6.5.AppImage" \
      https://github.com/obsidianmd/obsidian-releases/releases/download/v1.6.5/Obsidian-1.6.5.AppImage
    chmod +x "$CACHE/Obsidian-1.6.5.AppImage"
  fi
  echo "==> Extracting AppImage"
  mkdir -p "$CACHE/Obsidian-1.6.5-extracted"
  (cd "$CACHE/Obsidian-1.6.5-extracted" && "$CACHE/Obsidian-1.6.5.AppImage" --appimage-extract >/dev/null)
fi

mkdir -p "$VAULT/.obsidian/plugins/obsidian-brain" "$VAULT/.brain-data/blobs"

# Seed a demo set if the vault is (practically) empty.
md_count=$(find "$VAULT" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
if [[ $md_count -lt 3 ]]; then
  write_note() {
    local path="$1"; local cat="$2"; local title="$3"; shift 3
    mkdir -p "$(dirname "$VAULT/$path")"
    {
      printf '%s\n' "---"
      printf 'brain-category: %s\n' "$cat"
      printf 'tags: [brain/%s]\n' "$cat"
      printf '%s\n' "---"
      printf '\n# %s\n\n' "$title"
      printf '%s\n' "$@"
    } > "$VAULT/$path"
  }

  write_note "Welcome.md" "welcome" "RuVector Brain dev vault" \
    "This vault is wired to a local brain at http://127.0.0.1:${BRAIN_PORT} and an embedder at http://127.0.0.1:${EMBED_PORT}." \
    "" \
    "Try:" \
    "- [[Semantic search]] — Cmd+Shift+B (DiskANN over your memories)" \
    "- [[Related panel]] — ribbon brain icon" \
    "- [[Bulk sync]] — command palette → \`Brain: Bulk-sync vault → brain\`" \
    "- [[Graph overlay]] — command palette → \`Brain: Graph overlay — apply category colors\`" \
    "- [[DPO pairs]] — mark chosen / pair with rejected"
  write_note "Semantic search.md" "feature" "Semantic search" \
    "Press **Cmd+Shift+B** to open the modal. The plugin sends your query to \`POST /brain/search\` and ranks matches by cosine similarity over DiskANN." \
    "" \
    "Results open matching [[Welcome]] notes directly, or show content inline when no vault file is known."
  write_note "Related panel.md" "feature" "Related memories side panel" \
    "Docked in the right sidebar. On every active-leaf change it re-queries the brain for the first 2 KB of the current note." \
    "" \
    "Uses the same \`POST /brain/search\` endpoint as [[Semantic search]]; categories are colour-tagged via [[Graph overlay]]."
  write_note "Bulk sync.md" "feature" "Bulk-sync vault → brain" \
    "One-shot ingest of every markdown file matching your include/exclude filters. Notes are hashed; unchanged ones skip re-embedding." \
    "" \
    "Runs AIDefence scan per note before POSTing to \`/memories\`, so an [[Injection]] or [[PII]] leak fails closed."
  write_note "Graph overlay.md" "feature" "Graph overlay — category colours" \
    "Writes \`tag:#brain/<category>\` colour groups to \`.obsidian/graph.json\`, and stamps each indexed note with the matching tag via \`processFrontMatter\`." \
    "" \
    "Reversible — \`Brain: Graph overlay — clear\` strips only the \`#brain/*\` entries, leaving your other groups intact."
  write_note "DPO pairs.md" "feature" "DPO / preference pairs" \
    "Mark the *chosen* note, then open another and run \`Brain: create pair with current note (rejected)\`. Pairs are persisted server-side and can be exported as a markdown table under \`Brain/Exports/\`." \
    "" \
    "Direction label defaults to \`quality\` but accepts anything (e.g. \`tone\`, \`relevance\`, \`recency\`)."
  write_note "AIDefence.md" "security" "AIDefence scan" \
    "Every note is scanned by 13 injection patterns + 4 PII patterns compiled into the Rust brain. When the brain is offline, auto-index fails **closed** rather than leaking notes." \
    "" \
    "Related: [[Injection]], [[PII]]."
  write_note "Injection.md" "security" "Prompt injection examples" \
    "The brain blocks patterns like *\"ignore all previous instructions\"* or *\"disregard safety\"* with HTTP 422. The plugin surfaces these as toasts instead of silently dropping the note." \
    "" \
    "Tested end-to-end in \`tests/protocol/brain-server.test.ts\`."
  write_note "PII.md" "security" "PII detection" \
    "Patterns for credit cards, emails, phone numbers, SSNs are enforced before embedding. See \`aidefence_scan\` in the Rust server." \
    "" \
    "Links: [[AIDefence]]."
  write_note "Architecture.md" "architecture" "Plugin ↔ brain architecture" \
    "Plugin talks to \`mcp-brain-server-local\` (Rust / axum) over HTTP. The server owns the [[DiskANN index]], AIDefence regex set, and the SQLite content store." \
    "" \
    "No embedded vector DB on the Obsidian side — the brain is shared across CLI / MCP / agent runtime."
  write_note "DiskANN index.md" "architecture" "DiskANN Vamana graph" \
    "Brute-force cosine under 2K vectors; above that, a 64-degree Vamana graph routes queries with beam width \`k·8\` (min 40)." \
    "" \
    "Over-fetches + dedups to handle phantom index entries. See [[Architecture]]."
  write_note "Roadmap.md" "project" "Obsidian-brain roadmap" \
    "- [x] Phase 1 — MVP: semantic search, auto-index, settings, status bar" \
    "- [x] Phase 2 — Related panel, bulk sync" \
    "- [x] Phase 3 — Graph overlay, DPO pairs" \
    "- [ ] Phase 4 — streaming inference, vault-aware recall"
fi

# Copy the built plugin into the vault.
cp "$HERE"/main.js "$HERE"/manifest.json "$HERE"/styles.css "$HERE"/versions.json \
  "$VAULT/.obsidian/plugins/obsidian-brain/"

# Enable the plugin.
cat > "$VAULT/.obsidian/community-plugins.json" <<EOF
["obsidian-brain"]
EOF

# Disable Obsidian 1.12+ "Bases" core plugin — it auto-creates Untitled.base
# on every fresh launch which clutters the editor with a default database tab.
if [[ ! -f "$VAULT/.obsidian/core-plugins.json" ]]; then
  cat > "$VAULT/.obsidian/core-plugins.json" <<'EOF'
{
  "file-explorer": true,
  "global-search": true,
  "switcher": true,
  "graph": true,
  "backlink": true,
  "outgoing-link": true,
  "tag-pane": true,
  "page-preview": true,
  "command-palette": true,
  "editor-status": true,
  "bookmarks": true,
  "outline": true,
  "word-count": true,
  "file-recovery": true,
  "bases": false,
  "canvas": false,
  "properties": false,
  "daily-notes": false,
  "templates": false,
  "note-composer": false,
  "slash-command": false,
  "markdown-importer": false,
  "zk-prefixer": false,
  "random-note": false,
  "slides": false,
  "audio-recorder": false,
  "workspaces": false,
  "publish": false,
  "sync": false,
  "footnotes": false,
  "webviewer": false
}
EOF
fi
# And sweep up any .base file Bases may have dropped previously.
rm -f "$VAULT"/Untitled.base "$VAULT"/*.base 2>/dev/null || true

# Seed a workspace layout that auto-opens Welcome.md on first launch.
# Obsidian persists its own layout after the user moves things around,
# so we only write this when the file is missing.
if [[ ! -f "$VAULT/.obsidian/workspace.json" ]]; then
  cat > "$VAULT/.obsidian/workspace.json" <<'EOF'
{
  "main": {
    "id": "root",
    "type": "split",
    "children": [
      {
        "id": "leaf-welcome",
        "type": "leaf",
        "state": {
          "type": "markdown",
          "state": { "file": "Welcome.md", "mode": "source" }
        }
      }
    ],
    "direction": "vertical"
  },
  "left": {
    "id": "left-root",
    "type": "split",
    "children": [
      {
        "id": "left-tabs",
        "type": "tabs",
        "children": [
          { "id": "left-explorer", "type": "leaf", "state": { "type": "file-explorer", "state": {} } }
        ]
      }
    ],
    "direction": "horizontal",
    "width": 260
  },
  "right": {
    "id": "right-root",
    "type": "split",
    "children": [
      {
        "id": "right-tabs",
        "type": "tabs",
        "children": [
          { "id": "right-brain", "type": "leaf", "state": { "type": "obsidian-brain-related", "state": {} } }
        ]
      }
    ],
    "direction": "horizontal",
    "width": 300,
    "collapsed": false
  },
  "active": "leaf-welcome",
  "lastOpenFiles": ["Welcome.md"]
}
EOF
fi

# Point it at the dev brain instance.
cat > "$VAULT/.obsidian/plugins/obsidian-brain/data.json" <<EOF
{
  "settings": {
    "brainUrl": "http://127.0.0.1:${BRAIN_PORT}",
    "embedderUrl": "http://127.0.0.1:${EMBED_PORT}",
    "defaultCategory": "obsidian",
    "autoIndex": true,
    "autoIndexDebounceMs": 3000,
    "indexMinChars": 20,
    "enableAIDefence": true,
    "searchLimit": 8,
    "relatedLimit": 8,
    "bulkSyncBatchSize": 16,
    "bulkSyncIncludeFolders": "",
    "bulkSyncExcludeFolders": ".obsidian,.trash,.brain-data",
    "storeMapping": {},
    "dpoDirection": "quality"
  },
  "indexState": { "pathToHash": {}, "hashToId": {}, "idToPath": {}, "lastSync": 0 }
}
EOF

# Embedder — prefer the real ruvultra-embedder if it's already running
# (port 9877 by default; bge-small-en-v1.5 @ dim 384). Fall back to a tiny
# stub so the script still works on machines without the real one.
REAL_EMBED_URL="http://127.0.0.1:9877"
EMBED_SCRIPT=""
EMBEDDER_PID=""
if real=$(curl -sf --max-time 2 "${REAL_EMBED_URL}/health" 2>/dev/null) \
   && printf '%s' "$real" | grep -q '"status":"ok"'; then
  dim=$(printf '%s' "$real" | grep -oE '"dim":[0-9]+' | cut -d: -f2)
  model=$(printf '%s' "$real" | grep -oE '"model":"[^"]+"' | cut -d\" -f4)
  echo "==> using real embedder at ${REAL_EMBED_URL} (model=${model:-?} dim=${dim:-?})"
  EMBED_URL="$REAL_EMBED_URL"
else
  echo "==> real embedder not found; starting stub at 127.0.0.1:${EMBED_PORT}"
  EMBED_SCRIPT="$(mktemp /tmp/obsidian-brain-embedder-XXXXXX.mjs)"
  cat > "$EMBED_SCRIPT" <<'JS'
import http from "node:http";
const DIM = 16;
const server = http.createServer((req, res) => {
  if (req.method !== "POST" || !req.url?.startsWith("/embed")) {
    res.writeHead(404).end();
    return;
  }
  const chunks = [];
  req.on("data", c => chunks.push(c));
  req.on("end", () => {
    try {
      const { texts } = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      const vectors = texts.map(t => {
        const v = new Array(DIM).fill(0);
        for (let i = 0; i < t.length; i++) v[i % DIM] += t.charCodeAt(i) / 256;
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
        return v.map(x => x / norm);
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ vectors, dim: DIM, model: "stub" }));
    } catch (e) {
      res.writeHead(400).end(JSON.stringify({ error: String(e) }));
    }
  });
});
server.listen(parseInt(process.env.PORT, 10), "127.0.0.1", () => {
  console.log(`[embedder-stub] listening on 127.0.0.1:${process.env.PORT}`);
});
JS
  PORT="$EMBED_PORT" node "$EMBED_SCRIPT" &
  EMBEDDER_PID=$!
  EMBED_URL="http://127.0.0.1:${EMBED_PORT}"
fi

cleanup() {
  echo "==> shutting down"
  [[ -n "${EMBEDDER_PID:-}" ]] && kill "$EMBEDDER_PID" 2>/dev/null || true
  [[ -n "${BRAIN_PID:-}" ]] && kill "$BRAIN_PID" 2>/dev/null || true
  [[ -n "$EMBED_SCRIPT" ]] && rm -f "$EMBED_SCRIPT"
}
trap cleanup EXIT INT TERM

RUVBRAIN_PORT="$BRAIN_PORT" \
RUVBRAIN_DB="$VAULT/.brain-data/brain.sqlite" \
RUVBRAIN_BLOBS="$VAULT/.brain-data/blobs" \
RUVBRAIN_STORE="rvf" \
RUVBRAIN_EMBEDDER_URL="$EMBED_URL" \
RUST_LOG="${RUST_LOG:-info}" \
"$BRAIN_BIN" &
BRAIN_PID=$!

# Wait for brain health.
for _ in {1..100}; do
  if curl -sf "http://127.0.0.1:${BRAIN_PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done

# Seed the local brain + populate indexState + pull pi.ruv.io data, if a
# BRAIN_API_KEY is present in the environment.
PI_TOKEN="${PI_TOKEN:-${BRAIN_API_KEY:-}}"
PI_URL="${PI_URL:-https://pi.ruv.io}"
PI_LIMIT="${PI_LIMIT:-20}"
PI_QUERY="${PI_QUERY:-}"

PI_ARGS=(--pi-url "$PI_URL" --pi-limit "$PI_LIMIT")
if [[ -n "$PI_TOKEN" ]]; then
  PI_ARGS+=(--pi-token "$PI_TOKEN")
fi
if [[ -n "$PI_QUERY" ]]; then
  PI_ARGS+=(--pi-query "$PI_QUERY")
fi

python3 "$HERE/scripts/seed-dev.py" \
  "$VAULT" \
  "http://127.0.0.1:${BRAIN_PORT}" \
  "$EMBED_URL" \
  "${PI_ARGS[@]}"

# Isolated HOME so we don't mutate the user's real Obsidian vault registry.
# Persisted inside $VAULT/.obsidian-home so theme/layout survive between runs.
FAKE_HOME="$VAULT/.obsidian-home"
mkdir -p "$FAKE_HOME/.config/obsidian"

# Stable vault id so repeated launches reuse the same entry.
VAULT_ID="$(printf '%s' "$VAULT" | sha1sum | cut -c1-16)"

cat > "$FAKE_HOME/.config/obsidian/obsidian.json" <<EOF
{
  "vaults": {
    "${VAULT_ID}": {
      "path": "${VAULT}",
      "ts": $(date +%s)000,
      "open": true
    }
  },
  "insider": false
}
EOF

echo "==> brain ready at http://127.0.0.1:${BRAIN_PORT}"
echo "==> vault:  $VAULT"
echo "==> launching Obsidian (close the app to stop all services)"

HOME="$FAKE_HOME" \
XDG_CONFIG_HOME="$FAKE_HOME/.config" \
XDG_DATA_HOME="$FAKE_HOME/.local/share" \
XDG_CACHE_HOME="$FAKE_HOME/.cache" \
"$APPIMAGE_DIR/obsidian" --no-sandbox
