#!/bin/bash
# RVF Manifest Loader for rvAgent
# Parses manifest.rvf.json and initializes the agent environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$RUV_ROOT/manifest.rvf.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GOLD='\033[1;33m'
RESET='\033[0m'

log() { echo -e "${CYAN}[rvAgent] $1${RESET}"; }
success() { echo -e "${GREEN}[rvAgent] ✓ $1${RESET}"; }
warn() { echo -e "${YELLOW}[rvAgent] ⚠ $1${RESET}"; }
error() { echo -e "${RED}[rvAgent] ✗ $1${RESET}"; }

# =============================================================================
# Verify manifest exists
# =============================================================================
verify_manifest() {
    if [ ! -f "$MANIFEST" ]; then
        error "Manifest not found: $MANIFEST"
        exit 1
    fi
    success "Manifest found: $MANIFEST"
}

# =============================================================================
# Extract configuration values using jq or node
# =============================================================================
get_config() {
    local key="$1"
    if command -v jq &>/dev/null; then
        jq -r "$key" "$MANIFEST"
    else
        node -e "console.log(require('$MANIFEST')$key)"
    fi
}

# =============================================================================
# Initialize cognitive container segments
# =============================================================================
init_cognitive_container() {
    log "Initializing RVF cognitive container..."

    local segments=$(get_config '.cognitive_container.segments | length')
    log "Loading $segments segments..."

    # Initialize each segment type
    for i in $(seq 0 $((segments - 1))); do
        local seg_type=$(get_config ".cognitive_container.segments[$i].type")
        local purpose=$(get_config ".cognitive_container.segments[$i].purpose")
        log "  [$seg_type] $purpose"
    done

    success "Cognitive container initialized"
}

# =============================================================================
# Load agent definitions
# =============================================================================
load_agents() {
    log "Loading agent definitions..."

    local agents=("queen" "coder" "tester" "security")
    for agent in "${agents[@]}"; do
        local file=$(get_config ".agents.$agent.file")
        local model=$(get_config ".agents.$agent.model")
        local role=$(get_config ".agents.$agent.role")

        if [ -f "$RUV_ROOT/$file" ]; then
            success "  $agent ($role) → $model"
        else
            warn "  $agent: file not found: $file"
        fi
    done
}

# =============================================================================
# Configure swarm settings
# =============================================================================
configure_swarm() {
    log "Configuring swarm..."

    local topology=$(get_config '.swarm.topology')
    local max_agents=$(get_config '.swarm.max_agents')
    local consensus=$(get_config '.swarm.consensus')

    success "Topology: $topology | Max Agents: $max_agents | Consensus: $consensus"

    # Export for use by other scripts
    export RVAGENT_TOPOLOGY="$topology"
    export RVAGENT_MAX_AGENTS="$max_agents"
    export RVAGENT_CONSENSUS="$consensus"
}

# =============================================================================
# Enable SONA learning
# =============================================================================
enable_sona() {
    local sona_enabled=$(get_config '.learning.sona.enabled')
    local threshold=$(get_config '.learning.sona.adaptation_threshold_ms')

    if [ "$sona_enabled" = "true" ]; then
        success "SONA learning enabled (threshold: ${threshold}ms)"
        export RVAGENT_SONA_ENABLED=1
    else
        warn "SONA learning disabled"
    fi
}

# =============================================================================
# Initialize security controls
# =============================================================================
init_security() {
    log "Initializing security controls..."

    local virtual=$(get_config '.security.virtual_mode')
    local aimd=$(get_config '.security.aimd_enabled')
    local pii=$(get_config '.security.pii_detection')

    [ "$virtual" = "true" ] && success "  Virtual mode: ON"
    [ "$aimd" = "true" ] && success "  AIMD threat detection: ON"
    [ "$pii" = "true" ] && success "  PII detection: ON"

    export RVAGENT_VIRTUAL_MODE="$virtual"
    export RVAGENT_AIMD_ENABLED="$aimd"
}

# =============================================================================
# Main loader
# =============================================================================
main() {
    echo ""
    echo -e "${GOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${GOLD}║               🔷 rvAgent RVF Manifest Loader                 ║${RESET}"
    echo -e "${GOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
    echo ""

    verify_manifest
    init_cognitive_container
    load_agents
    configure_swarm
    enable_sona
    init_security

    echo ""
    success "rvAgent environment ready"
    echo ""
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
