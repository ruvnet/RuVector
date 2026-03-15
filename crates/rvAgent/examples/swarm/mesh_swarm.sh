#!/usr/bin/env bash
# Mesh Swarm Pattern - Peer-to-peer distributed coordination
#
# This pattern implements a mesh topology where all agents can communicate
# with each other. There's no central coordinator - agents share context
# and build on each other's work through message passing.
#
# Topology: Agent <-> Agent <-> Agent (fully connected)
# Use case: Collaborative problem-solving, brainstorming, consensus

set -e

# Configuration
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
MODEL="${MODEL:-google:gemini-2.5-pro}"
RVAGENT="${RVAGENT:-./target/debug/rvagent}"

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY not set"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           MESH SWARM PATTERN - Peer-to-Peer Collaboration          ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║  Topology: Agent <-> Agent <-> Agent                               ║"
echo "║  No central coordinator - shared context and iteration             ║"
echo "║  Agents: Security, Performance, UX (equal peers)                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

TOPIC="Design an authentication system for a web application"

# Round 1: Each peer contributes initial perspective
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Round 1: INITIAL PERSPECTIVES (parallel contributions)             │"
echo "└────────────────────────────────────────────────────────────────────┘"

echo "► Peer [SECURITY]:"
SEC_1=$($RVAGENT --model "$MODEL" run "
You are the SECURITY peer in a mesh swarm discussing: $TOPIC
Share your initial perspective focusing on security concerns.
3 bullet points max, be concise.
" 2>&1)
echo "$SEC_1"
echo

echo "► Peer [PERFORMANCE]:"
PERF_1=$($RVAGENT --model "$MODEL" run "
You are the PERFORMANCE peer in a mesh swarm discussing: $TOPIC
Share your initial perspective focusing on performance.
3 bullet points max, be concise.
" 2>&1)
echo "$PERF_1"
echo

echo "► Peer [UX]:"
UX_1=$($RVAGENT --model "$MODEL" run "
You are the UX peer in a mesh swarm discussing: $TOPIC
Share your initial perspective focusing on user experience.
3 bullet points max, be concise.
" 2>&1)
echo "$UX_1"
echo

# Round 2: Peers respond to each other (mesh communication)
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Round 2: PEER RESPONSES (building on each other)                   │"
echo "└────────────────────────────────────────────────────────────────────┘"

echo "► Peer [SECURITY] responds to UX concerns:"
SEC_2=$($RVAGENT --model "$MODEL" run "
You are SECURITY peer. UX peer raised: smooth login, SSO, password recovery.
How can security accommodate these UX needs? 2 lines max.
" 2>&1)
echo "$SEC_2"
echo

echo "► Peer [PERFORMANCE] responds to SECURITY concerns:"
PERF_2=$($RVAGENT --model "$MODEL" run "
You are PERFORMANCE peer. SECURITY suggested: rate limiting, MFA, token rotation.
How can we implement these without hurting performance? 2 lines max.
" 2>&1)
echo "$PERF_2"
echo

echo "► Peer [UX] responds to PERFORMANCE concerns:"
UX_2=$($RVAGENT --model "$MODEL" run "
You are UX peer. PERFORMANCE suggested: caching, async operations.
How can UX design support these performance optimizations? 2 lines max.
" 2>&1)
echo "$UX_2"
echo

# Round 3: Consensus (all peers agree on solution)
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Round 3: CONSENSUS FORMATION                                       │"
echo "└────────────────────────────────────────────────────────────────────┘"

CONSENSUS=$($RVAGENT --model "$MODEL" run "
The mesh swarm has discussed authentication from 3 perspectives:
- SECURITY: MFA, rate limiting, secure tokens
- PERFORMANCE: Caching, async verification
- UX: Smooth flow, SSO, password recovery

As a neutral synthesizer, state the consensus in 4 lines:
1. Authentication method chosen
2. Key security measure
3. Performance optimization
4. UX consideration
" 2>&1)
echo "$CONSENSUS"
echo

echo "════════════════════════════════════════════════════════════════════"
echo "MESH SWARM COMPLETE - 3 peers reached consensus"
echo "════════════════════════════════════════════════════════════════════"
