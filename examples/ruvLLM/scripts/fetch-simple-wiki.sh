#!/usr/bin/env bash
# fetch-simple-wiki.sh — download + extract Simple-English-Wikipedia dump
# into shard-*.txt files consumable by ruvllm-pretrain --corpus.
#
# Requires:
#   - bash, curl, bzip2  (system tools)
#   - python3 + wikiextractor  (`pip install wikiextractor`)
#
# Usage:
#   ./scripts/fetch-simple-wiki.sh [OUT_DIR]
#   default OUT_DIR = ./data/simple-wiki
#
# Idempotent: skips download/extract if the target file already exists.

set -euo pipefail

OUT_DIR="${1:-./data/simple-wiki}"
DUMP_URL="https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DUMP_BZ2="${OUT_DIR}/simplewiki-latest-pages-articles.xml.bz2"
DUMP_XML="${OUT_DIR}/simplewiki-latest-pages-articles.xml"
EXTRACT_DIR="${OUT_DIR}/extracted"

mkdir -p "${OUT_DIR}"

# 1. Download.
if [[ -f "${DUMP_BZ2}" || -f "${DUMP_XML}" ]]; then
  echo "✓ dump already present, skipping download"
else
  echo "→ downloading ${DUMP_URL}"
  curl -L --fail --output "${DUMP_BZ2}" "${DUMP_URL}"
fi

# 2. Decompress.
if [[ -f "${DUMP_XML}" ]]; then
  echo "✓ XML already extracted"
elif [[ -f "${DUMP_BZ2}" ]]; then
  echo "→ decompressing bz2"
  bzip2 -dk "${DUMP_BZ2}"
fi

# 3. Verify wikiextractor is available.
if ! command -v wikiextractor >/dev/null 2>&1; then
  echo "ERROR: wikiextractor not found on PATH." >&2
  echo "       Install it with: pip install wikiextractor" >&2
  exit 2
fi

# 4. Extract.
if [[ -d "${EXTRACT_DIR}" ]] && [[ -n "$(find "${EXTRACT_DIR}" -name 'wiki_*' -print -quit 2>/dev/null)" ]]; then
  echo "✓ wikiextractor output already present"
else
  echo "→ running wikiextractor (this can take a while)"
  rm -rf "${EXTRACT_DIR}"
  wikiextractor --no-templates --processes 4 --output "${EXTRACT_DIR}" "${DUMP_XML}"
fi

# 5. Flatten extractor output into shard-XXXX.txt.
# wikiextractor produces AA/wiki_00, AA/wiki_01, ... we strip <doc> tags and
# keep one paragraph per line, blank line separating articles.
echo "→ producing shard-*.txt"
shard_idx=0
shard_lines=0
shard_max_lines=20000
shard_path() { printf "%s/shard-%04d.txt" "${OUT_DIR}" "${shard_idx}"; }
: > "$(shard_path)"

# Use python for robust XML-tag stripping.
python3 - "${EXTRACT_DIR}" "${OUT_DIR}" "${shard_max_lines}" <<'PY'
import os, sys, re
from pathlib import Path

extract_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
shard_max_lines = int(sys.argv[3])

doc_re = re.compile(r"^<doc[^>]*>$")
end_re = re.compile(r"^</doc>$")

shard_idx = 0
shard_lines = 0
shard_path = out_dir / f"shard-{shard_idx:04d}.txt"
out = open(shard_path, "w")

for p in sorted(extract_dir.rglob("wiki_*")):
    with open(p, encoding="utf-8") as fh:
        in_doc = False
        for line in fh:
            line = line.rstrip("\n")
            if doc_re.match(line):
                in_doc = True
                continue
            if end_re.match(line):
                if in_doc:
                    out.write("\n")  # blank line separates articles
                    shard_lines += 1
                in_doc = False
                if shard_lines >= shard_max_lines:
                    out.close()
                    shard_idx += 1
                    shard_lines = 0
                    shard_path = out_dir / f"shard-{shard_idx:04d}.txt"
                    out = open(shard_path, "w")
                continue
            if in_doc and line.strip():
                out.write(line + "\n")
                shard_lines += 1

out.close()
print(f"wrote {shard_idx + 1} shards to {out_dir}")
PY

echo "✓ done — shards in ${OUT_DIR}/shard-*.txt"
