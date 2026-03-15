#!/usr/bin/env bash
#
# train_brain.sh - Push discovery JSON files to pi.ruv.io brain API
#
# Reads all *_discoveries.json files from the discoveries directory,
# extracts each discovery entry, and POSTs it as a memory to the
# brain API using the challenge-nonce authentication flow.
#
# Usage: ./scripts/train_brain.sh [discoveries_dir]
#
set -euo pipefail

DISCOVERIES_DIR="${1:-/home/user/RuVector/examples/data/discoveries}"
BRAIN_API="https://pi.ruv.io"
RATE_LIMIT_SECONDS=1

# Counters
TOTAL=0
SUCCESS=0
FAIL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date '+%H:%M:%S') $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }

# -------------------------------------------------------------------
# Dependency check
# -------------------------------------------------------------------
check_deps() {
    if ! command -v curl &>/dev/null; then
        echo "curl is required but not installed. Aborting." >&2
        exit 1
    fi

    if ! command -v jq &>/dev/null; then
        log_warn "jq not found -- attempting to install..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq jq
        elif command -v yum &>/dev/null; then
            sudo yum install -y jq
        elif command -v brew &>/dev/null; then
            brew install jq
        else
            echo "Cannot install jq automatically. Please install it manually." >&2
            exit 1
        fi
    fi

    log_info "Dependencies satisfied (curl, jq)"
}

# -------------------------------------------------------------------
# Obtain a fresh challenge nonce from the brain API
# -------------------------------------------------------------------
get_nonce() {
    local response
    response=$(curl -sf --max-time 10 "${BRAIN_API}/v1/challenge" 2>/dev/null) || {
        log_fail "Failed to fetch challenge nonce from ${BRAIN_API}/v1/challenge"
        return 1
    }

    local nonce
    nonce=$(echo "$response" | jq -r '.nonce // empty')
    if [[ -z "$nonce" ]]; then
        log_fail "Challenge response did not contain a nonce"
        return 1
    fi

    echo "$nonce"
}

# -------------------------------------------------------------------
# Post a single discovery to the brain API
# -------------------------------------------------------------------
post_memory() {
    local title="$1"
    local content="$2"
    local tags_json="$3"

    # Get a fresh nonce for each request
    local nonce
    nonce=$(get_nonce) || return 1

    local payload
    payload=$(jq -n \
        --arg title "$title" \
        --arg content "$content" \
        --argjson tags "$tags_json" \
        '{
            title: $title,
            content: $content,
            category: "pattern",
            tags: $tags
        }')

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 15 \
        -X POST "${BRAIN_API}/v1/memories" \
        -H "Content-Type: application/json" \
        -H "X-Challenge-Nonce: ${nonce}" \
        -d "$payload" 2>/dev/null) || {
        log_fail "curl error posting memory: ${title}"
        return 1
    }

    if [[ "$http_code" =~ ^2 ]]; then
        return 0
    else
        log_fail "HTTP ${http_code} for memory: ${title}"
        return 1
    fi
}

# -------------------------------------------------------------------
# Process a single discovery JSON file
# -------------------------------------------------------------------
process_file() {
    local filepath="$1"
    local filename
    filename=$(basename "$filepath")

    log_info "Processing file: ${filename}"

    # Validate JSON
    if ! jq empty "$filepath" 2>/dev/null; then
        log_fail "Invalid JSON in ${filename} -- skipping"
        return
    fi

    # Determine structure: could be an array of discoveries or an
    # object with a "discoveries" key containing an array.
    local entries_json
    if jq -e 'type == "array"' "$filepath" &>/dev/null; then
        entries_json=$(jq -c '.[]' "$filepath")
    elif jq -e '.discoveries | type == "array"' "$filepath" &>/dev/null; then
        entries_json=$(jq -c '.discoveries[]' "$filepath")
    elif jq -e '.entries | type == "array"' "$filepath" &>/dev/null; then
        entries_json=$(jq -c '.entries[]' "$filepath")
    else
        # Treat the whole file as a single discovery object
        entries_json=$(jq -c '.' "$filepath")
    fi

    if [[ -z "$entries_json" ]]; then
        log_warn "No discovery entries found in ${filename}"
        return
    fi

    while IFS= read -r entry; do
        TOTAL=$((TOTAL + 1))

        # Extract title -- try common field names
        local title
        title=$(echo "$entry" | jq -r '
            .title //
            .name //
            .discovery //
            .summary //
            ("Discovery #" + (input_line_number | tostring))
        ' 2>/dev/null)
        [[ -z "$title" || "$title" == "null" ]] && title="Discovery ${TOTAL}"

        # Extract content -- try common field names
        local content
        content=$(echo "$entry" | jq -r '
            .content //
            .description //
            .details //
            .finding //
            .text //
            (. | tostring)
        ' 2>/dev/null)
        [[ -z "$content" || "$content" == "null" ]] && content=$(echo "$entry" | jq -c '.')

        # Extract tags -- merge with defaults
        local file_tags
        file_tags=$(echo "$entry" | jq -c '
            (if .tags then
                if (.tags | type) == "array" then .tags else [.tags] end
            else
                []
            end) + ["discovery"]
            | unique
        ' 2>/dev/null)
        [[ -z "$file_tags" || "$file_tags" == "null" ]] && file_tags='["discovery"]'

        # Post to brain API
        if post_memory "$title" "$content" "$file_tags"; then
            SUCCESS=$((SUCCESS + 1))
            log_ok "[${TOTAL}] ${title}"
        else
            FAIL=$((FAIL + 1))
            log_fail "[${TOTAL}] ${title}"
        fi

        # Rate limiting
        sleep "$RATE_LIMIT_SECONDS"

    done <<< "$entries_json"
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
main() {
    echo ""
    echo "=========================================="
    echo "  Brain Training - Discovery Upload"
    echo "=========================================="
    echo ""

    check_deps

    # Verify discoveries directory
    if [[ ! -d "$DISCOVERIES_DIR" ]]; then
        log_fail "Discoveries directory not found: ${DISCOVERIES_DIR}"
        exit 1
    fi

    # Collect discovery files
    shopt -s nullglob
    local files=("${DISCOVERIES_DIR}"/*_discoveries.json)
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        log_warn "No *_discoveries.json files found in ${DISCOVERIES_DIR}"
        log_info "Discovery agents may not have completed yet."
        exit 0
    fi

    log_info "Found ${#files[@]} discovery file(s) in ${DISCOVERIES_DIR}"
    echo ""

    # Process each file
    for filepath in "${files[@]}"; do
        process_file "$filepath"
        echo ""
    done

    # Summary
    echo "=========================================="
    echo "  Training Summary"
    echo "=========================================="
    echo ""
    log_info "Total discoveries processed: ${TOTAL}"
    log_ok   "Successful submissions:      ${SUCCESS}"
    if [[ $FAIL -gt 0 ]]; then
        log_fail "Failed submissions:          ${FAIL}"
    else
        log_info "Failed submissions:          0"
    fi
    echo ""

    if [[ $FAIL -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
