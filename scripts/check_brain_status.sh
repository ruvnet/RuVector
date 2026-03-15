#!/usr/bin/env bash
#
# check_brain_status.sh - Query pi.ruv.io brain API status and recent discoveries
#
# Usage: ./scripts/check_brain_status.sh [search_query] [limit]
#
set -euo pipefail

BRAIN_API="https://pi.ruv.io"
SEARCH_QUERY="${1:-discovery}"
LIMIT="${2:-5}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# -------------------------------------------------------------------
# Dependency check
# -------------------------------------------------------------------
if ! command -v jq &>/dev/null; then
    echo "jq is required. Install it with: sudo apt-get install jq" >&2
    exit 1
fi

# -------------------------------------------------------------------
# Brain status
# -------------------------------------------------------------------
echo ""
echo -e "${BOLD}=========================================="
echo "  Brain Status - pi.ruv.io"
echo -e "==========================================${NC}"
echo ""

status_response=$(curl -sf --max-time 10 "${BRAIN_API}/v1/status" 2>/dev/null) || {
    echo -e "${RED}Failed to reach ${BRAIN_API}/v1/status${NC}"
    echo "The brain API may be temporarily unavailable."
    echo ""
    exit 1
}

# Display status fields -- handle various response shapes
echo -e "${CYAN}API Status:${NC}"
echo "$status_response" | jq -r '
    to_entries[] |
    "  \(.key): \(.value)"
' 2>/dev/null || echo "  $status_response"

echo ""

# -------------------------------------------------------------------
# Recent discoveries search
# -------------------------------------------------------------------
echo -e "${BOLD}------------------------------------------"
echo "  Recent Discoveries (query: \"${SEARCH_QUERY}\", limit: ${LIMIT})"
echo -e "------------------------------------------${NC}"
echo ""

search_url="${BRAIN_API}/v1/memories/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${SEARCH_QUERY}'))" 2>/dev/null || echo "${SEARCH_QUERY}")&limit=${LIMIT}"

search_response=$(curl -sf --max-time 10 "$search_url" 2>/dev/null) || {
    echo -e "${YELLOW}Could not fetch discoveries from search endpoint.${NC}"
    echo "Endpoint: ${search_url}"
    echo ""
    exit 0
}

# Count results
result_count=$(echo "$search_response" | jq '
    if type == "array" then length
    elif .memories then (.memories | length)
    elif .results then (.results | length)
    elif .data then (.data | length)
    else 0
    end
' 2>/dev/null || echo "0")

echo -e "${CYAN}Results found: ${result_count}${NC}"
echo ""

# Extract and display results -- try common response shapes
display_results() {
    local data="$1"

    # Normalize to array
    local items
    items=$(echo "$data" | jq -c '
        if type == "array" then .
        elif .memories then .memories
        elif .results then .results
        elif .data then .data
        else [.]
        end
    ' 2>/dev/null) || return

    echo "$items" | jq -r '
        .[] |
        "  \u001b[1m\(.title // .name // "Untitled")\u001b[0m",
        "  \u001b[2mCategory: \(.category // "N/A") | Tags: \(.tags // [] | join(", "))\u001b[0m",
        "  \(.content // .description // "No content" | if length > 120 then .[:120] + "..." else . end)",
        ""
    ' 2>/dev/null || echo "  (Could not parse results)"
}

display_results "$search_response"

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
echo -e "${DIM}------------------------------------------${NC}"
echo -e "${DIM}  API: ${BRAIN_API}${NC}"
echo -e "${DIM}  Time: $(date '+%Y-%m-%d %H:%M:%S %Z')${NC}"
echo ""
