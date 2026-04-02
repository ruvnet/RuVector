#!/usr/bin/env bash
# claude-code-rvf-corpus.sh - Build binary RVF containers for every major
# Claude Code CLI release.
#
# Downloads the latest patch of each major.minor series from npm, extracts
# the CLI bundle, splits into modules, and creates a binary RVF container
# with vector embeddings and witness chains.
#
# Usage:
#   ./scripts/claude-code-rvf-corpus.sh [--dry-run] [--series 0.2,1.0,2.0,2.1]
#
# Output: docs/research/claude-code-rvsource/versions/<vX.Y.z>/
#   - claude-code-vX.Y.rvf          Binary RVF container
#   - claude-code-vX.Y.rvf.manifest.json  Container manifest
#   - source/                        Extracted JS modules
#   - README.md                      Version metadata

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${ROOT_DIR}/docs/research/claude-code-rvsource/versions"
TMP_DIR="/tmp/cc-rvf-corpus-$$"
DRY_RUN=false
FILTER_SERIES=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${CYAN}[*]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[-]${NC} $*" >&2; }

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --series)   FILTER_SERIES="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--series 0.2,1.0,2.0,2.1]"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Fetch all versions from npm and group by major.minor
get_version_groups() {
    log "Fetching Claude Code versions from npm..." >&2
    local versions_json
    versions_json=$(npm view @anthropic-ai/claude-code versions --json 2>/dev/null)

    # Use node to group versions and pick latest patch per major.minor
    node -e "
const versions = $versions_json;
const groups = {};

for (const v of versions) {
    const parts = v.split('.');
    const key = parts[0] + '.' + parts[1];
    const patch = parseInt(parts[2], 10);

    if (!groups[key] || patch > groups[key].patch) {
        groups[key] = { version: v, patch, key };
    }
}

// Sort by semver
const sorted = Object.values(groups).sort((a, b) => {
    const [aMaj, aMin] = a.key.split('.').map(Number);
    const [bMaj, bMin] = b.key.split('.').map(Number);
    return aMaj !== bMaj ? aMaj - bMaj : aMin - bMin;
});

for (const g of sorted) {
    console.log(g.key + ' ' + g.version);
}
"
}

# Download and extract a specific version
download_version() {
    local version="$1"
    local dest_dir="$2"

    mkdir -p "$dest_dir"
    info "  Downloading @anthropic-ai/claude-code@${version}..."

    local tgz_dir="${TMP_DIR}/tarballs"
    mkdir -p "$tgz_dir"

    npm pack "@anthropic-ai/claude-code@${version}" --pack-destination "$tgz_dir" \
        >/dev/null 2>&1

    # Find the tarball (naming varies between npm versions)
    local tgz
    tgz=$(ls "$tgz_dir"/anthropic-ai-claude-code-*.tgz 2>/dev/null | head -1)
    if [[ -z "$tgz" ]]; then
        err "  Failed to download version ${version}"
        return 1
    fi

    # Try to extract cli.js, then cli.mjs (don't list the tarball, just try)
    tar xf "$tgz" -C "$dest_dir" --strip-components=1 package/cli.js 2>/dev/null || true
    tar xf "$tgz" -C "$dest_dir" --strip-components=1 package/cli.mjs 2>/dev/null || true
    tar xf "$tgz" -C "$dest_dir" --strip-components=1 package/package.json 2>/dev/null || true

    # Rename cli.mjs -> cli.js for consistency
    if [[ -f "${dest_dir}/cli.mjs" ]] && [[ ! -f "${dest_dir}/cli.js" ]]; then
        mv "${dest_dir}/cli.mjs" "${dest_dir}/cli.js"
    fi

    if [[ ! -f "${dest_dir}/cli.js" ]]; then
        warn "  No cli.js or cli.mjs found in ${version}"
        return 1
    fi

    rm -f "$tgz"
    local size
    size=$(du -sh "${dest_dir}/cli.js" 2>/dev/null | cut -f1)
    info "  Extracted cli.js (${size})"
    return 0
}

# Split a CLI bundle into modules
split_modules() {
    local cli_path="$1"
    local source_dir="$2"

    info "  Splitting into modules..."
    node "${SCRIPT_DIR}/lib/module-splitter.mjs" "$cli_path" "$source_dir" 2>/dev/null
}

# Build a binary RVF container
build_rvf() {
    local source_dir="$1"
    local rvf_path="$2"
    local version="$3"
    local series="$4"

    info "  Building binary RVF container..."
    node "${SCRIPT_DIR}/lib/rvf-builder.mjs" \
        "$source_dir" "$rvf_path" \
        --meta "version=${version}" \
        --meta "series=${series}" \
        --meta "package=@anthropic-ai/claude-code" \
        --meta "corpus=claude-code-rvsource" \
        2>/dev/null
}

# Generate a README for a version directory
generate_readme() {
    local ver_dir="$1"
    local series="$2"
    local version="$3"
    local rvf_file="$4"

    local metrics_file="${ver_dir}/source/metrics.json"
    local manifest_file="${rvf_file}.manifest.json"

    # Read metrics
    local bundle_size="unknown"
    local classes="?"
    local functions="?"
    local modules_count="?"

    if [[ -f "$metrics_file" ]]; then
        bundle_size=$(node -e "const m=JSON.parse(require('fs').readFileSync('$metrics_file','utf-8')); console.log((m.sizeBytes/1024/1024).toFixed(1)+'MB')")
        classes=$(node -e "const m=JSON.parse(require('fs').readFileSync('$metrics_file','utf-8')); console.log(m.classes)")
        functions=$(node -e "const m=JSON.parse(require('fs').readFileSync('$metrics_file','utf-8')); console.log(m.functions)")
        modules_count=$(node -e "const m=JSON.parse(require('fs').readFileSync('$metrics_file','utf-8')); console.log(Object.keys(m.modules||{}).length)")
    fi

    local rvf_size="N/A"
    local rvf_vectors="N/A"
    local rvf_id="N/A"
    if [[ -f "$manifest_file" ]]; then
        rvf_size=$(node -e "const m=JSON.parse(require('fs').readFileSync('$manifest_file','utf-8')); console.log((m.fileSizeBytes/1024).toFixed(1)+'KB')")
        rvf_vectors=$(node -e "const m=JSON.parse(require('fs').readFileSync('$manifest_file','utf-8')); console.log(m.totalVectors)")
        rvf_id=$(node -e "const m=JSON.parse(require('fs').readFileSync('$manifest_file','utf-8')); console.log(m.fileId)")
    fi

    cat > "${ver_dir}/README.md" <<READMEEOF
# Claude Code v${version} (${series} series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | ${version} |
| Series | ${series} |
| Bundle size | ${bundle_size} |
| RVF size | ${rvf_size} |
| Vectors | ${rvf_vectors} |
| RVF File ID | \`${rvf_id}\` |
| Classes | ${classes} |
| Functions | ${functions} |
| Modules | ${modules_count} |
| Extracted | $(date -Iseconds) |

## Files

- \`claude-code-v${series}.rvf\` - Binary RVF container with HNSW index + witness chain
- \`claude-code-v${series}.rvf.manifest.json\` - Container manifest (vector ID map, metadata)
- \`source/\` - Extracted JavaScript module fragments

## RVF Container Details

The \`.rvf\` file is a real binary container created with the \`@ruvector/rvf-node\`
native backend. It contains:

- **128-dimensional fingerprint vectors** for each code fragment
- **HNSW index** (M=16, ef_construction=200) for fast similarity search
- **Cosine distance** metric
- **Witness chain** for provenance verification

To query this container:

\`\`\`typescript
import { RvfDatabase } from '@ruvector/rvf';

const db = await RvfDatabase.openReadonly('./claude-code-v${series}.rvf');
const results = await db.query(queryVector, 10);
await db.close();
\`\`\`
READMEEOF
}

# Generate the top-level index README
generate_index() {
    local base_dir="$1"
    shift
    local entries=("$@")

    cat > "${base_dir}/README.md" <<'INDEXHEADER'
# Claude Code RVF Corpus

Binary RVF containers for every major Claude Code CLI release, with
HNSW-indexed vector embeddings and witness chains for provenance.

## Versions

| Series | Version | Bundle | RVF Size | Vectors | File ID |
|--------|---------|--------|----------|---------|---------|
INDEXHEADER

    for entry in "${entries[@]}"; do
        echo "$entry" >> "${base_dir}/README.md"
    done

    cat >> "${base_dir}/README.md" <<'INDEXFOOTER'

## How to Use

```bash
# Build the corpus
./scripts/claude-code-rvf-corpus.sh

# Build only specific series
./scripts/claude-code-rvf-corpus.sh --series 2.0,2.1
```

## Format

Each version directory contains:
- A binary `.rvf` container (128-dim cosine-distance HNSW index)
- A `.manifest.json` sidecar with vector-to-fragment mapping
- Extracted JavaScript modules in `source/`

Generated by `scripts/claude-code-rvf-corpus.sh` using `@ruvector/rvf-node`.
INDEXFOOTER
}

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

main() {
    echo -e "${BOLD}Claude Code RVF Corpus Builder${NC}"
    echo -e "${BOLD}==============================${NC}"
    echo ""

    mkdir -p "$TMP_DIR" "$OUTPUT_BASE"

    # Get version groups
    local groups
    groups=$(get_version_groups)

    if [[ -z "$groups" ]]; then
        err "No versions found on npm"
        exit 1
    fi

    local total_groups
    total_groups=$(echo "$groups" | wc -l)
    log "Found ${total_groups} major.minor series"

    # Apply filter if specified
    if [[ -n "$FILTER_SERIES" ]]; then
        local filtered=""
        IFS=',' read -ra FILTER_ARRAY <<< "$FILTER_SERIES"
        while IFS= read -r line; do
            local series
            series=$(echo "$line" | awk '{print $1}')
            for f in "${FILTER_ARRAY[@]}"; do
                if [[ "$series" == "$f" ]]; then
                    filtered+="${line}"$'\n'
                fi
            done
        done <<< "$groups"
        groups="$filtered"
        total_groups=$(echo -n "$groups" | grep -c '^' || echo 0)
        log "Filtered to ${total_groups} series: ${FILTER_SERIES}"
    fi

    if $DRY_RUN; then
        warn "DRY RUN - would process these versions:"
        echo "$groups" | while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local series version
            series=$(echo "$line" | awk '{print $1}')
            version=$(echo "$line" | awk '{print $2}')
            echo "  v${series}.x -> ${version}"
        done
        exit 0
    fi

    local processed=0
    local failed=0

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue

        local series version
        series=$(echo "$line" | awk '{print $1}')
        version=$(echo "$line" | awk '{print $2}')

        echo ""
        log "Processing v${series}.x (latest: ${version})"

        local ver_dir="${OUTPUT_BASE}/v${series}.x"
        local source_dir="${ver_dir}/source"
        local rvf_file="${ver_dir}/claude-code-v${series}.rvf"
        local extract_dir="${TMP_DIR}/extract-${version}"

        mkdir -p "$ver_dir" "$source_dir"

        # Step 1: Download
        if ! download_version "$version" "$extract_dir"; then
            warn "  Skipping ${version} (download failed)"
            ((failed++)) || true
            continue
        fi

        local cli_path="${extract_dir}/cli.js"
        if [[ ! -f "$cli_path" ]]; then
            warn "  No CLI bundle found for ${version}"
            ((failed++)) || true
            continue
        fi

        # Step 2: Split into modules
        if ! split_modules "$cli_path" "$source_dir"; then
            warn "  Module splitting failed for ${version}"
        fi

        # Step 3: Build binary RVF container
        if build_rvf "$source_dir" "$rvf_file" "$version" "$series"; then
            log "  RVF container created: $(basename "$rvf_file")"
        else
            warn "  RVF creation failed for ${version}"
            # Create a fallback TODO note
            cat > "${ver_dir}/TODO-rvf.md" <<EOF
# TODO: Create RVF Container

Version: ${version}
Series: v${series}.x
Error: RVF binary creation failed

The source modules have been extracted to \`source/\` but the binary
RVF container could not be created. This typically means the
\`@ruvector/rvf-node\` native backend is not available.

To create the container manually:

\`\`\`bash
node scripts/lib/rvf-builder.mjs source/ claude-code-v${series}.rvf \\
  --meta version=${version} --meta series=${series}
\`\`\`
EOF
        fi

        # Step 4: Generate README
        generate_readme "$ver_dir" "$series" "$version" "$rvf_file"

        # Clean up extracted tarball content
        rm -rf "$extract_dir"

        ((processed++)) || true
        log "  Done (${processed}/${total_groups})"
    done <<< "$groups"

    # Generate index
    echo ""
    log "Generating corpus index..."

    # Rebuild index entries by scanning output dirs
    local final_entries=()
    for d in "${OUTPUT_BASE}"/v*.x; do
        [[ -d "$d" ]] || continue
        local dir_name
        dir_name=$(basename "$d")
        local series_name="${dir_name#v}"
        series_name="${series_name%.x}"

        local manifest="${d}/claude-code-v${series_name}.rvf.manifest.json"
        if [[ -f "$manifest" ]]; then
            local row
            row=$(node -e "
const m=JSON.parse(require('fs').readFileSync('$manifest','utf-8'));
const s=m.source||{};
const met=s.metrics||{};
const bundle=(met.bundleSizeBytes/1024/1024).toFixed(1)+'MB';
const rvfSize=(m.fileSizeBytes/1024).toFixed(1)+'KB';
console.log('| ${series_name} | '+s.version+' | '+bundle+' | '+rvfSize+' | '+m.totalVectors+' | \`'+m.fileId.slice(0,12)+'...\` |');
" 2>/dev/null || echo "| ${series_name} | ? | ? | ? | ? | ? |")
            final_entries+=("$row")
        else
            final_entries+=("| ${series_name} | ? | ? | N/A | N/A | N/A |")
        fi
    done

    generate_index "$OUTPUT_BASE" "${final_entries[@]}"

    echo ""
    echo -e "${BOLD}Corpus build complete.${NC}"
    log "Output: ${OUTPUT_BASE}/"
    log "Versions processed: ${processed:-0}"
    if [[ ${failed:-0} -gt 0 ]]; then
        warn "Versions failed: ${failed}"
    fi
}

main "$@"
