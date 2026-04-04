#!/usr/bin/env bash
# claude-code-decompile.sh - Extract and analyze Claude Code CLI source
#
# Extracts the bundled JavaScript from the Claude Code binary or npm package,
# applies basic beautification, and splits into logical modules.
#
# Usage: ./scripts/claude-code-decompile.sh [output-dir]
#
# Output directory defaults to ./claude-code-extracted/

set -euo pipefail

OUTPUT_DIR="${1:-./claude-code-extracted}"
BINARY=""
CLI_JS=""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[-]${NC} $*" >&2; }

# Find the Claude Code source
find_source() {
    # Method 1: NPM package (preferred - cleaner JS)
    local npm_paths=(
        "$(npm root -g 2>/dev/null)/claude-flow/node_modules/@anthropic-ai/claude-code/cli.js"
        "$(npm root -g 2>/dev/null)/@anthropic-ai/claude-code/cli.js"
        "./node_modules/@anthropic-ai/claude-code/cli.js"
    )
    for p in "${npm_paths[@]}"; do
        if [[ -f "$p" ]]; then
            CLI_JS="$p"
            log "Found NPM package: $CLI_JS"
            return 0
        fi
    done

    # Method 2: Bun SEA binary
    local bin_paths=(
        "$HOME/.local/bin/claude"
        "$HOME/.local/share/claude/versions/"
        "/usr/local/bin/claude"
    )
    for p in "${bin_paths[@]}"; do
        if [[ -f "$p" ]]; then
            BINARY="$(readlink -f "$p" 2>/dev/null || echo "$p")"
            log "Found binary: $BINARY"
            return 0
        elif [[ -d "$p" ]]; then
            BINARY="$(ls -t "$p"* 2>/dev/null | head -1)"
            if [[ -n "$BINARY" ]]; then
                log "Found binary: $BINARY"
                return 0
            fi
        fi
    done

    err "Could not find Claude Code binary or npm package"
    echo "Install via: npm install -g @anthropic-ai/claude-code"
    echo "Or ensure claude is installed: claude --version"
    return 1
}

# Extract JS from Bun SEA binary using strings
extract_from_binary() {
    local binary="$1"
    local output="$2"

    log "Extracting strings from binary ($(du -h "$binary" | cut -f1))..."
    strings "$binary" > "${output}/raw-strings.txt"

    local total_lines
    total_lines=$(wc -l < "${output}/raw-strings.txt")
    log "Extracted $total_lines string fragments"

    # Extract JS-like patterns
    log "Filtering JavaScript patterns..."
    grep -E '(function\s|class\s|=>\s*\{|export\s|import\s|require\(|async\s|await\s|const\s|let\s|var\s)' \
        "${output}/raw-strings.txt" > "${output}/js-fragments.txt" 2>/dev/null || true

    local js_lines
    js_lines=$(wc -l < "${output}/js-fragments.txt")
    log "Found $js_lines JS-like fragments"
}

# Process the cli.js bundle
process_bundle() {
    local source="$1"
    local output="$2"

    log "Processing bundle: $(du -h "$source" | cut -f1)"

    # Copy original
    cp "$source" "${output}/cli.js.original"

    # Basic beautification: add newlines at statement boundaries
    log "Beautifying (adding newlines at statement boundaries)..."
    sed 's/;/;\n/g' "$source" | \
    sed 's/{/{\n/g' | \
    sed 's/}/}\n/g' > "${output}/cli.beautified.js"

    local beautified_lines
    beautified_lines=$(wc -l < "${output}/cli.beautified.js")
    log "Beautified: $beautified_lines lines"

    # Extract metrics
    log "Computing code metrics..."
    {
        echo "=== Claude Code Source Metrics ==="
        echo "Date: $(date -Iseconds)"
        echo "Source: $source"
        echo "Original size: $(wc -c < "$source") bytes"
        echo "Original lines: $(wc -l < "$source")"
        echo "Beautified lines: $beautified_lines"
        echo ""
        echo "--- Counts ---"
        echo "Functions: $(grep -oP 'function\s*\w*\s*\(' "$source" | wc -l)"
        echo "Async functions: $(grep -oP 'async\s+function' "$source" | wc -l)"
        echo "Arrow functions: $(grep -oP '=>' "$source" | wc -l)"
        echo "Classes: $(grep -oP 'class \w+' "$source" | wc -l)"
        echo "Extends: $(grep -oP 'extends \w+' "$source" | wc -l)"
        echo "For-await loops: $(grep -c 'for await' "$source")"
        echo "Yield statements: $(grep -c 'yield' "$source")"
        echo ""
        echo "--- Node.js Imports ---"
        grep -oP 'from"[^"]*"' "$source" | sort -u | grep -P 'from"(node:|assert|child_process|crypto|events|fs|http|https|module|net|os|path|process|stream|tty|url|util|zlib)'
        echo ""
        echo "--- Class Definitions ---"
        grep -oP 'class \w+( extends \w+)?' "$source" | sort -u
    } > "${output}/metrics.txt"

    log "Metrics saved to ${output}/metrics.txt"
}

# Split into logical modules based on patterns
split_modules() {
    local source="$1"
    local output="$2"
    local modules_dir="${output}/modules"
    mkdir -p "$modules_dir"

    log "Splitting into logical modules..."

    # Extract tool-related code
    grep -oP '.{0,200}(BashTool|FileReadTool|FileEditTool|FileWriteTool|AgentOutputTool|WebFetch|WebSearch|TodoWrite|NotebookEdit|GlobTool|GrepTool).{0,200}' \
        "$source" > "${modules_dir}/tools.txt" 2>/dev/null || true

    # Extract permission-related code
    grep -oP '.{0,200}(permission|Permission|canUseTool|alwaysAllowRules|denyWrite|sandbox|Sandbox).{0,200}' \
        "$source" > "${modules_dir}/permissions.txt" 2>/dev/null || true

    # Extract MCP-related code
    grep -oP '.{0,200}(mcp__|McpClient|McpServer|McpError|callTool|listTools|initialize).{0,200}' \
        "$source" > "${modules_dir}/mcp.txt" 2>/dev/null || true

    # Extract streaming-related code
    grep -oP '.{0,200}(content_block_delta|message_start|message_stop|message_delta|content_block_start|content_block_stop|stream_event|text_delta|input_json_delta).{0,200}' \
        "$source" > "${modules_dir}/streaming.txt" 2>/dev/null || true

    # Extract context/compaction code
    grep -oP '.{0,200}(compact|compaction|tengu_compact|microcompact|auto_compact|compact_boundary|preCompactTokenCount|postCompactTokenCount).{0,200}' \
        "$source" > "${modules_dir}/compaction.txt" 2>/dev/null || true

    # Extract agent loop code
    grep -oP '.{0,200}(agentLoop|mainLoop|s\$\(|querySource|toolUseContext|systemPrompt).{0,200}' \
        "$source" > "${modules_dir}/agent-loop.txt" 2>/dev/null || true

    # Extract telemetry events
    grep -oP '"tengu_[^"]*"' "$source" | sort -u > "${modules_dir}/telemetry-events.txt" 2>/dev/null || true

    # Extract string constants (tool names, commands, etc.)
    grep -oP 'name:"[a-z][-a-z]*",description:"[^"]*"' "$source" | sort -u > "${modules_dir}/commands.txt" 2>/dev/null || true

    # Extract class hierarchy
    grep -oP 'class \w+ extends \w+' "$source" | sort -u > "${modules_dir}/class-hierarchy.txt" 2>/dev/null || true

    # Count extracted lines per module
    for f in "${modules_dir}"/*.txt; do
        local name
        name=$(basename "$f" .txt)
        local lines
        lines=$(wc -l < "$f")
        log "  Module '$name': $lines fragments"
    done
}

# Generate RVF files from extracted modules
generate_rvf() {
    local modules_dir="$1/modules"
    local rvf_dir="$1/rvf"
    mkdir -p "$rvf_dir"

    log "Generating RVF files..."

    local version
    version=$(grep -oP 'VERSION:"[^"]*"' "$modules_dir/../cli.js.original" 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")

    for f in "${modules_dir}"/*.txt; do
        local name
        name=$(basename "$f" .txt)
        local rvf_file="${rvf_dir}/${name}.rvf"
        {
            echo "---"
            echo "type: source-extraction"
            echo "module: ${name}"
            echo "binary: claude-code"
            echo "version: ${version}"
            echo "extraction-method: strings+pattern-match"
            echo "confidence: medium"
            echo "fragments: $(wc -l < "$f")"
            echo "---"
            echo ""
            echo "# ${name} - Extracted Fragments"
            echo ""
            echo '```javascript'
            cat "$f"
            echo '```'
        } > "$rvf_file"
        log "  Created ${rvf_file}"
    done
}

# Main
main() {
    log "Claude Code Decompiler"
    log "======================"

    mkdir -p "$OUTPUT_DIR"

    find_source

    if [[ -n "$CLI_JS" ]]; then
        process_bundle "$CLI_JS" "$OUTPUT_DIR"
        split_modules "$CLI_JS" "$OUTPUT_DIR"
        generate_rvf "$OUTPUT_DIR"
    elif [[ -n "$BINARY" ]]; then
        extract_from_binary "$BINARY" "$OUTPUT_DIR"
        # If we got enough JS, process it
        if [[ -f "${OUTPUT_DIR}/js-fragments.txt" ]]; then
            split_modules "${OUTPUT_DIR}/js-fragments.txt" "$OUTPUT_DIR"
            generate_rvf "$OUTPUT_DIR"
        fi
    fi

    log ""
    log "Extraction complete!"
    log "Output directory: $OUTPUT_DIR"
    log ""
    log "Key files:"
    log "  metrics.txt        - Code metrics and counts"
    log "  cli.beautified.js  - Beautified bundle (if from NPM)"
    log "  modules/           - Split by logical module"
    log "  rvf/               - RVF files with metadata headers"

    # Summary
    if [[ -f "${OUTPUT_DIR}/metrics.txt" ]]; then
        echo ""
        head -15 "${OUTPUT_DIR}/metrics.txt"
    fi
}

main "$@"
