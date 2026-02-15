#!/bin/bash
set -e

# Testability Scoring Assessment Runner
# Usage: ./run-assessment.sh <URL> [browser]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TEST_FILE="$PROJECT_ROOT/tests/testability-scoring/testability-scoring.spec.js"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if URL provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: URL required${NC}"
    echo "Usage: $0 <URL> [browser]"
    echo "Example: $0 https://example.com chromium"
    exit 1
fi

TARGET_URL="$1"
BROWSER="${2:-chromium}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎯 Testability Assessment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${YELLOW}URL:${NC}     $TARGET_URL"
echo -e "  ${YELLOW}Browser:${NC} $BROWSER"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run Playwright tests with TEST_URL environment variable
echo -e "${YELLOW}⏳ Running assessment...${NC}\n"

TEST_URL="$TARGET_URL" npx playwright test "$TEST_FILE" --project="$BROWSER" --workers=1

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Assessment completed successfully!${NC}"
    
    # Find the latest JSON report
    LATEST_JSON=$(ls -t "$PROJECT_ROOT/tests/reports/testability-results-"*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_JSON" ]; then
        # Generate HTML report
        echo -e "\n${YELLOW}📊 Generating HTML report...${NC}\n"
        node "$SCRIPT_DIR/generate-html-report.js" "$LATEST_JSON"
        
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ Complete! Check your browser for the report.${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo -e "${YELLOW}⚠️  JSON report not found${NC}"
    fi
else
    echo -e "\n${RED}❌ Assessment failed${NC}"
    exit 1
fi
