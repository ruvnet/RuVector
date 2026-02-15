#!/bin/bash
# Brutal Honesty Test Assessment Script (Ramsay Mode)

set -e

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "👨‍🍳 BRUTAL HONESTY TEST ASSESSMENT (Ramsay Mode)"
echo "=================================================="
echo ""

# Check if test directory argument provided
if [ -z "$1" ]; then
    echo "Usage: $0 <test-directory>"
    exit 1
fi

TEST_DIR="$1"

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}🔴 FAILING: Test directory '$TEST_DIR' doesn't exist${NC}"
    echo "   → Where are the tests? Did you even write any?"
    exit 1
fi

# Function to assess coverage
assess_coverage() {
    echo "📊 COVERAGE CHECK"
    echo "----------------"

    # Run coverage if npm test with coverage exists
    if [ -f "package.json" ] && grep -q "test:coverage" package.json; then
        echo "Running coverage analysis..."
        npm run test:coverage 2>/dev/null || true

        # Extract coverage percentage
        coverage=$(npm run test:coverage 2>&1 | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "0")

        if (( $(echo "$coverage < 50" | bc -l) )); then
            echo -e "${RED}🔴 RAW: ${coverage}% coverage${NC}"
            echo "   → This is embarrassing. You're barely testing anything."
        elif (( $(echo "$coverage < 80" | bc -l) )); then
            echo -e "${YELLOW}🟡 ACCEPTABLE: ${coverage}% coverage${NC}"
            echo "   → Minimum is 80%. You're not there yet."
        else
            echo -e "${GREEN}🟢 MICHELIN STAR: ${coverage}% coverage${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No coverage command found${NC}"
        echo "   → Add 'test:coverage' script to package.json"
    fi
}

# Function to assess edge cases
assess_edge_cases() {
    echo ""
    echo "🎯 EDGE CASE CHECK"
    echo "-----------------"

    # Check for common edge case patterns
    edge_case_patterns=(
        "null"
        "undefined"
        "empty"
        "zero"
        "negative"
        "max"
        "min"
        "overflow"
        "boundary"
    )

    found_count=0
    for pattern in "${edge_case_patterns[@]}"; do
        if grep -ri "$pattern" "$TEST_DIR" > /dev/null 2>&1; then
            ((found_count++))
        fi
    done

    if [ "$found_count" -eq 0 ]; then
        echo -e "${RED}🔴 RAW: No edge cases tested${NC}"
        echo "   → You're only testing the happy path. That's not testing."
    elif [ "$found_count" -lt 3 ]; then
        echo -e "${YELLOW}🟡 ACCEPTABLE: Found $found_count edge case patterns${NC}"
        echo "   → Test more: null, empty, boundaries, overflow"
    else
        echo -e "${GREEN}🟢 MICHELIN STAR: Found $found_count edge case patterns${NC}"
    fi
}

# Function to assess test clarity
assess_clarity() {
    echo ""
    echo "📖 CLARITY CHECK"
    echo "---------------"

    # Check for descriptive test names
    unclear_tests=$(grep -r "test('test" "$TEST_DIR" 2>/dev/null | wc -l)
    if [ "$unclear_tests" -gt 0 ]; then
        echo -e "${RED}🔴 RAW: Found $unclear_tests unclear test names${NC}"
        echo "   → 'test1', 'test2' - What are you testing? Use descriptive names."
    fi

    # Check for describe/it blocks
    if grep -r "describe\|it\|test" "$TEST_DIR" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Tests have structure${NC}"
    else
        echo -e "${YELLOW}⚠️  No test framework patterns detected${NC}"
    fi
}

# Function to assess test speed
assess_speed() {
    echo ""
    echo "⚡ SPEED CHECK"
    echo "-------------"

    echo "Running tests..."
    start_time=$(date +%s)

    # Run tests (suppress output)
    if npm test > /dev/null 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        if [ "$duration" -gt 60 ]; then
            echo -e "${RED}🔴 RAW: Tests took ${duration}s${NC}"
            echo "   → Unit tests should run in seconds, not minutes."
            echo "   → Are you calling real databases/networks?"
        elif [ "$duration" -gt 10 ]; then
            echo -e "${YELLOW}🟡 ACCEPTABLE: Tests took ${duration}s${NC}"
            echo "   → Aim for <10s. Use mocks and in-memory operations."
        else
            echo -e "${GREEN}🟢 MICHELIN STAR: Tests took ${duration}s${NC}"
        fi
    else
        echo -e "${RED}🔴 FAILING: Tests don't even pass${NC}"
        echo "   → Fix your broken tests before worrying about speed."
    fi
}

# Function to assess stability
assess_stability() {
    echo ""
    echo "🎲 STABILITY CHECK"
    echo "-----------------"

    # Check for flaky patterns
    if grep -ri "setTimeout\|sleep\|wait" "$TEST_DIR" > /dev/null 2>&1; then
        echo -e "${RED}🔴 RAW: Timing-based tests detected${NC}"
        echo "   → You're creating flaky tests. Use proper async/await."
    fi

    # Run tests multiple times to detect flakes
    echo "Running tests 3x to detect flakes..."
    failures=0
    for i in {1..3}; do
        if ! npm test > /dev/null 2>&1; then
            ((failures++))
        fi
    done

    if [ "$failures" -gt 0 ]; then
        echo -e "${RED}🔴 RAW: Tests failed $failures/3 times${NC}"
        echo "   → FLAKY TESTS. These are worse than no tests."
        echo "   → Fix the non-determinism before merging."
    else
        echo -e "${GREEN}🟢 MICHELIN STAR: Tests are stable${NC}"
    fi
}

# Function to assess isolation
assess_isolation() {
    echo ""
    echo "🏝️  ISOLATION CHECK"
    echo "------------------"

    # Check for shared state patterns
    if grep -ri "global\|beforeAll\|shared" "$TEST_DIR" > /dev/null 2>&1; then
        echo -e "${YELLOW}🟡 WARNING: Shared state patterns detected${NC}"
        echo "   → Are your tests independent? Can they run in any order?"
    fi

    # Check for test order dependencies
    if grep -r "\.only\|\.skip" "$TEST_DIR" > /dev/null 2>&1; then
        echo -e "${YELLOW}🟡 WARNING: .only or .skip found${NC}"
        echo "   → Don't commit tests with .only or .skip"
    fi

    echo -e "${GREEN}✓ Review test isolation manually${NC}"
}

# Run all assessments
assess_coverage
assess_edge_cases
assess_clarity
assess_speed
assess_stability
assess_isolation

# Final verdict
echo ""
echo "=================================================="
echo "🎯 FINAL VERDICT"
echo "=================================================="
echo ""
echo "Look at the results above. If you see multiple 🔴 RAW marks,"
echo "these tests are NOT production-ready."
echo ""
echo "Expected standards:"
echo "  - 80%+ branch coverage"
echo "  - Edge cases tested (null, empty, boundaries)"
echo "  - Clear test names"
echo "  - <10s to run"
echo "  - 0% flaky"
echo "  - Independent tests"
echo ""
echo "You know what good tests look like. Why aren't you writing them?"
