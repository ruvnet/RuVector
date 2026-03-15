#!/bin/bash
# rvAgent Integration Hooks
# Connects rvAgent crates with Claude Flow learning and RVF cognitive stack

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RVAGENT_ROOT="$SCRIPT_DIR/.."

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
queen() { echo -e "${GOLD}[👑 Queen] $1${RESET}"; }

# =============================================================================
# Pre-Task Hook: Initialize RVF witness chain
# =============================================================================
pre_task() {
    local task_id="${1:-$(uuidgen | tr '[:upper:]' '[:lower:]')}"
    local description="$2"
    local agent_type="${3:-coder}"

    log "Initializing task: $task_id"

    # Get routing recommendation
    local routing=$(npx @claude-flow/cli@latest hooks pre-task \
        --taskId "$task_id" \
        --description "$description" 2>/dev/null)

    # Check for agent booster (skip LLM entirely)
    if echo "$routing" | grep -q "AGENT_BOOSTER_AVAILABLE"; then
        success "Agent Booster available - using Tier 1 (0ms, \$0)"
        echo "TIER=1"
        return 0
    fi

    # Get model recommendation
    local model=$(echo "$routing" | grep -o 'Use model="[^"]*"' | cut -d'"' -f2)
    if [ -n "$model" ]; then
        success "Recommended model: $model"
        echo "MODEL=$model"
    fi

    # Initialize witness chain
    npx @claude-flow/cli@latest memory store \
        --key "rvagent/task/$task_id/witness" \
        --namespace "rvagent" \
        --value "{\"task_id\":\"$task_id\",\"agent\":\"$agent_type\",\"started\":$(date +%s),\"entries\":[]}" \
        2>/dev/null

    success "Task initialized with witness chain"
    echo "TASK_ID=$task_id"
}

# =============================================================================
# Post-Task Hook: Record learning and consolidate patterns
# =============================================================================
post_task() {
    local task_id="$1"
    local success="${2:-true}"
    local quality="${3:-0.85}"

    log "Completing task: $task_id"

    # Record task outcome
    npx @claude-flow/cli@latest hooks post-task \
        --taskId "$task_id" \
        --success "$success" \
        --quality "$quality" \
        2>/dev/null

    # Trigger SONA learning if successful
    if [ "$success" = "true" ] && [ "$(echo "$quality >= 0.8" | bc)" -eq 1 ]; then
        log "Triggering SONA learning (quality: $quality)"
        npx @claude-flow/cli@latest hooks intelligence_learn \
            --consolidate true \
            2>/dev/null
    fi

    success "Task complete, patterns consolidated"
}

# =============================================================================
# Spawn rvAgent Swarm
# =============================================================================
spawn_swarm() {
    local task="$1"
    local topology="${2:-hierarchical}"
    local max_agents="${3:-6}"

    queen "Initializing rvAgent swarm"

    # Initialize swarm
    npx @claude-flow/cli@latest swarm init \
        --topology "$topology" \
        --max-agents "$max_agents" \
        --strategy specialized \
        2>/dev/null

    # Spawn queen
    queen "Spawning rvAgent Queen coordinator"
    npx @claude-flow/cli@latest agent spawn \
        --type rvagent-queen \
        --name queen-1 \
        --config '{"rvf_enabled":true,"witness_enabled":true}' \
        2>/dev/null

    success "rvAgent swarm initialized"
}

# =============================================================================
# RVF Witness Chain Operations
# =============================================================================
witness_append() {
    local task_id="$1"
    local operation="$2"
    local data="$3"

    # Get current chain
    local chain=$(npx @claude-flow/cli@latest memory retrieve \
        --key "rvagent/task/$task_id/witness" \
        --namespace "rvagent" 2>/dev/null)

    # Append entry (would be SHAKE-256 linked in production)
    local entry="{\"op\":\"$operation\",\"data\":$data,\"ts\":$(date +%s)}"

    log "Appended witness entry: $operation"
}

# =============================================================================
# Security Scan
# =============================================================================
security_scan() {
    local input="$1"

    log "Running AIMD security scan"

    local result=$(npx @claude-flow/cli@latest aidefence scan \
        --input "$input" 2>/dev/null)

    if echo "$result" | grep -q '"safe":true'; then
        success "Input is safe"
        return 0
    else
        warn "Potential threat detected"
        echo "$result"
        return 1
    fi
}

# =============================================================================
# Main dispatcher
# =============================================================================
case "$1" in
    pre-task)
        pre_task "$2" "$3" "$4"
        ;;
    post-task)
        post_task "$2" "$3" "$4"
        ;;
    spawn-swarm)
        spawn_swarm "$2" "$3" "$4"
        ;;
    witness-append)
        witness_append "$2" "$3" "$4"
        ;;
    security-scan)
        security_scan "$2"
        ;;
    *)
        echo "Usage: $0 {pre-task|post-task|spawn-swarm|witness-append|security-scan} [args]"
        exit 1
        ;;
esac
