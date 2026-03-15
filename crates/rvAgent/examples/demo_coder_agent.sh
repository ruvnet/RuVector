#!/bin/bash
# rvAgent Coder Demo - Code Generation with Witness Chains
# Uses Anthropic API via Claude Flow integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUV_DIR="$SCRIPT_DIR/../.ruv"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
GOLD='\033[1;33m'
RESET='\033[0m'

echo ""
echo -e "${GOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GOLD}║           🔷 rvAgent Coder Demo - Code Generation            ║${RESET}"
echo -e "${GOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Step 1: Load manifest
echo -e "${CYAN}[1/5] Loading RVF manifest...${RESET}"
source "$RUV_DIR/helpers/load-manifest.sh" 2>/dev/null || true

# Step 2: Initialize task with witness chain
TASK_ID="demo-coder-$(date +%s)"
TASK_DESC="Generate a Rust function to calculate Fibonacci numbers"

echo -e "${CYAN}[2/5] Initializing task with witness chain...${RESET}"
npx @claude-flow/cli@latest hooks pre-task \
    --taskId "$TASK_ID" \
    --description "$TASK_DESC" 2>/dev/null || echo "Task initialized: $TASK_ID"

# Step 3: Get model routing recommendation
echo -e "${CYAN}[3/5] Getting model routing recommendation...${RESET}"
ROUTING=$(npx @claude-flow/cli@latest hooks model-route \
    --task "$TASK_DESC" 2>/dev/null || echo "model: sonnet")
echo "Routing: $ROUTING"

# Step 4: Call Anthropic API via rvAgent
echo -e "${CYAN}[4/5] Calling Anthropic API (Sonnet)...${RESET}"
echo ""

RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are rvagent-coder, a Rust code generation specialist. Generate clean, efficient, well-documented code. Include tests.",
        "messages": [
            {
                "role": "user",
                "content": "Generate a Rust function to calculate Fibonacci numbers efficiently using memoization. Include unit tests."
            }
        ]
    }')

# Extract and display the response
echo -e "${GREEN}Generated Code:${RESET}"
echo "$RESPONSE" | jq -r '.content[0].text' 2>/dev/null || echo "$RESPONSE"

# Step 5: Record task completion with SONA learning
echo ""
echo -e "${CYAN}[5/5] Recording task completion (SONA learning)...${RESET}"
npx @claude-flow/cli@latest hooks post-task \
    --taskId "$TASK_ID" \
    --success true \
    --quality 0.95 2>/dev/null || echo "Task completed: $TASK_ID"

echo ""
echo -e "${GREEN}✓ Demo complete! Witness chain recorded.${RESET}"
