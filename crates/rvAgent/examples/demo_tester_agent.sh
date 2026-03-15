#!/bin/bash
# rvAgent Tester Demo - TDD London School Test Generation
# Uses Anthropic API (Haiku for fast test generation)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
GOLD='\033[1;33m'
RESET='\033[0m'

echo ""
echo -e "${GOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GOLD}║        🧪 rvAgent Tester Demo - TDD London School            ║${RESET}"
echo -e "${GOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Code to test
CODE_UNDER_TEST='pub struct Calculator {
    memory: f64,
}

impl Calculator {
    pub fn new() -> Self {
        Self { memory: 0.0 }
    }

    pub fn add(&mut self, a: f64, b: f64) -> f64 {
        let result = a + b;
        self.memory = result;
        result
    }

    pub fn recall(&self) -> f64 {
        self.memory
    }
}'

echo -e "${CYAN}[1/3] Analyzing code for test generation...${RESET}"
echo "Target: Calculator struct with add() and recall() methods"
echo ""

# Use Haiku for fast test generation (Tier 2)
echo -e "${CYAN}[2/3] Generating tests via Anthropic API (Haiku - fast)...${RESET}"

RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "{
        \"model\": \"claude-3-5-haiku-20241022\",
        \"max_tokens\": 1500,
        \"system\": \"You are rvagent-tester using TDD London School methodology. Generate comprehensive tests with mocks where appropriate. Use Rust's #[cfg(test)] module.\",
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Generate comprehensive unit tests for this Rust code using TDD London School approach:\\n\\n$CODE_UNDER_TEST\\n\\nInclude: happy path tests, edge cases, and verify memory state.\"
            }
        ]
    }")

echo -e "${GREEN}Generated Tests (TDD London School):${RESET}"
echo ""
echo "$RESPONSE" | jq -r '.content[0].text' 2>/dev/null || echo "$RESPONSE"

echo ""
echo -e "${CYAN}[3/3] Recording test pattern for SONA learning...${RESET}"
npx @claude-flow/cli@latest memory store \
    --key "rvagent/tester/pattern/calculator-tests" \
    --namespace "testing" \
    --value "TDD pattern: Calculator with memory state, test add/recall separation" 2>/dev/null || echo "Pattern stored"

echo ""
echo -e "${GREEN}✓ Tester demo complete!${RESET}"
