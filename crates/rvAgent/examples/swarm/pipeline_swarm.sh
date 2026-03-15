#!/usr/bin/env bash
# Pipeline Swarm Pattern - Sequential task handoff
#
# This pattern implements a pipeline topology where each agent processes
# the output of the previous agent. Work flows linearly through stages.
#
# Topology: Stage1 -> Stage2 -> Stage3 -> Stage4
# Use case: Multi-stage processing, refinement chains, code reviews

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
echo "║          PIPELINE SWARM PATTERN - Sequential Processing            ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║  Topology: Stage1 -> Stage2 -> Stage3 -> Stage4                    ║"
echo "║  Each stage refines the output of the previous                     ║"
echo "║  Stages: Spec -> Design -> Implement -> Test                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

INPUT="Create a function to validate email addresses"

# Stage 1: Specification
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Stage 1: SPECIFICATION                                             │"
echo "└────────────────────────────────────────────────────────────────────┘"

SPEC=$($RVAGENT --model "$MODEL" run "
You are STAGE-1 (Specification) in a pipeline swarm.
Input: $INPUT

Output a precise specification:
- Function name and signature
- Input validation rules (format, length, allowed chars)
- Return type (bool or Result)
- Edge cases to handle

Be concise, 5 lines max.
" 2>&1)
echo "Input: $INPUT"
echo "Output:"
echo "$SPEC"
echo

# Stage 2: Design
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Stage 2: DESIGN                                                    │"
echo "└────────────────────────────────────────────────────────────────────┘"

DESIGN=$($RVAGENT --model "$MODEL" run "
You are STAGE-2 (Design) in a pipeline swarm.
Input from Stage 1:
$SPEC

Design the implementation approach:
- Algorithm (regex vs parser vs character scan)
- Performance considerations
- Error handling strategy

Be concise, 4 lines max.
" 2>&1)
echo "Input: (Specification from Stage 1)"
echo "Output:"
echo "$DESIGN"
echo

# Stage 3: Implementation
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Stage 3: IMPLEMENTATION                                            │"
echo "└────────────────────────────────────────────────────────────────────┘"

IMPL=$($RVAGENT --model "$MODEL" run "
You are STAGE-3 (Implementation) in a pipeline swarm.
Input from Stage 2:
$DESIGN

Write the Rust implementation. Use simple character scanning, no regex.
Output only the function code, no explanations.
Keep it under 15 lines.
" 2>&1)
echo "Input: (Design from Stage 2)"
echo "Output:"
echo "$IMPL"
echo

# Stage 4: Testing
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ Stage 4: TESTING                                                   │"
echo "└────────────────────────────────────────────────────────────────────┘"

TESTS=$($RVAGENT --model "$MODEL" run "
You are STAGE-4 (Testing) in a pipeline swarm.
Input from Stage 3:
$IMPL

Write 4 test cases in Rust #[test] format:
1. Valid email
2. Missing @
3. Missing domain
4. Invalid characters

Output only the test code, under 20 lines.
" 2>&1)
echo "Input: (Implementation from Stage 3)"
echo "Output:"
echo "$TESTS"
echo

# Final: Pipeline summary
echo "┌────────────────────────────────────────────────────────────────────┐"
echo "│ PIPELINE SUMMARY                                                   │"
echo "└────────────────────────────────────────────────────────────────────┘"

SUMMARY=$($RVAGENT --model "$MODEL" run "
Summarize this pipeline execution in 3 lines:
Stage 1: Created specification for email validation
Stage 2: Designed character-scanning approach
Stage 3: Implemented is_valid_email function
Stage 4: Generated 4 test cases

What was the transformation from input to output?
" 2>&1)
echo "$SUMMARY"
echo

echo "════════════════════════════════════════════════════════════════════"
echo "PIPELINE SWARM COMPLETE - 4 stages processed sequentially"
echo "════════════════════════════════════════════════════════════════════"
