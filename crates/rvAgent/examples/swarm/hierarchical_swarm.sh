#!/usr/bin/env bash
# Hierarchical Swarm Pattern - Queen-led coordination with worker agents
#
# This pattern implements a hierarchical topology where a "Queen" agent
# coordinates multiple worker agents. The Queen breaks down tasks, assigns
# work, and synthesizes results.
#
# Topology: Queen (coordinator) -> Workers (specialists)
# Use case: Complex tasks requiring decomposition and synthesis

set -e

# Configuration
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
MODEL="${MODEL:-google:gemini-2.5-pro}"
RVAGENT="${RVAGENT:-./target/debug/rvagent}"

# Check API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY not set"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         HIERARCHICAL SWARM PATTERN - Queen-Led Coordination        ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║  Topology: Queen → Workers                                         ║"
echo "║  Queen: Task decomposition, coordination, synthesis                ║"
echo "║  Workers: Specialized execution (architect, coder, reviewer)       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# Phase 1: Queen analyzes task and creates work plan
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Phase 1: QUEEN - Task Analysis & Decomposition                     │"
echo "└────────────────────────────────────────────────────────────────────┘"

TASK="Design a simple key-value store with get, set, and delete operations"

QUEEN_PLAN=$($RVAGENT --model "$MODEL" run "
You are the QUEEN coordinator in a hierarchical swarm. Analyze this task and create a work plan:
TASK: $TASK

Decompose into 3 subtasks for worker agents:
1. ARCHITECT: Design decision (data structure, interface)
2. CODER: Implementation (Rust code)
3. REVIEWER: Quality check (edge cases, improvements)

Output format:
ARCHITECT_TASK: <one line>
CODER_TASK: <one line>
REVIEWER_TASK: <one line>
" 2>&1)

echo "Queen's Work Plan:"
echo "$QUEEN_PLAN"
echo

# Phase 2: Worker agents execute their subtasks
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Phase 2: WORKERS - Specialized Execution                           │"
echo "└────────────────────────────────────────────────────────────────────┘"

# Worker 1: Architect
echo "► Worker [ARCHITECT]:"
ARCHITECT_RESULT=$($RVAGENT --model "$MODEL" run "
You are the ARCHITECT worker in a hierarchical swarm.
Task: Design a key-value store with get, set, delete.
Provide: Data structure choice and public interface (function signatures).
Be concise, output in 5 lines or less.
" 2>&1)
echo "$ARCHITECT_RESULT"
echo

# Worker 2: Coder
echo "► Worker [CODER]:"
CODER_RESULT=$($RVAGENT --model "$MODEL" run "
You are the CODER worker in a hierarchical swarm.
Implement this design: HashMap-based store with get/set/delete.
Write minimal Rust code (struct + 3 methods). No tests, just implementation.
Output only the code, no explanations.
" 2>&1)
echo "$CODER_RESULT"
echo

# Worker 3: Reviewer
echo "► Worker [REVIEWER]:"
REVIEWER_RESULT=$($RVAGENT --model "$MODEL" run "
You are the REVIEWER worker in a hierarchical swarm.
Review this code and identify 2 potential improvements:
\`\`\`rust
use std::collections::HashMap;
struct KVStore { data: HashMap<String, String> }
impl KVStore {
    fn get(&self, k: &str) -> Option<&String> { self.data.get(k) }
    fn set(&mut self, k: String, v: String) { self.data.insert(k, v); }
    fn delete(&mut self, k: &str) -> Option<String> { self.data.remove(k) }
}
\`\`\`
Be concise, 3 lines max.
" 2>&1)
echo "$REVIEWER_RESULT"
echo

# Phase 3: Queen synthesizes results
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Phase 3: QUEEN - Result Synthesis                                  │"
echo "└────────────────────────────────────────────────────────────────────┘"

SYNTHESIS=$($RVAGENT --model "$MODEL" run "
You are the QUEEN synthesizing worker results:

ARCHITECT: HashMap-based KV store with get/set/delete interface
CODER: Implemented struct with 3 methods using HashMap
REVIEWER: Suggested improvements for edge cases

Synthesize into a 3-line summary of what was accomplished.
" 2>&1)
echo "$SYNTHESIS"
echo

echo "════════════════════════════════════════════════════════════════════"
echo "HIERARCHICAL SWARM COMPLETE - Queen coordinated 3 workers"
echo "════════════════════════════════════════════════════════════════════"
