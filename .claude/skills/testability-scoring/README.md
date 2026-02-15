# Testability Scoring Skill

Quick reference for AI-powered testability assessment using 10 principles of intrinsic testability.

## Quick Start

```bash
# Run assessment
.claude/skills/testability-scoring/scripts/run-assessment.sh https://example.com/

# Generate HTML report from JSON
AUTO_OPEN=false node .claude/skills/testability-scoring/scripts/generate-html-report.js tests/reports/testability-results-*.json
```

## 10 Principles

1. **Observability** (15%) - Can we see what's happening?
2. **Controllability** (15%) - Can we control the application?
3. **Algorithmic Simplicity** (10%) - Are behaviors simple?
4. **Algorithmic Transparency** (10%) - Can we understand it?
5. **Algorithmic Stability** (10%) - Does it stay consistent?
6. **Explainability** (10%) - Is the interface clear?
7. **Unbugginess** (10%) - How error-free is it?
8. **Smallness** (10%) - Are components appropriate size?
9. **Decomposability** (5%) - Can we test parts separately?
10. **Similarity** (5%) - How familiar is the tech?

## Scoring

- **90-100 (A)**: Excellent
- **80-89 (B)**: Good
- **70-79 (C)**: Adequate
- **60-69 (D)**: Below average
- **0-59 (F)**: Poor

## Files

```
.claude/skills/testability-scoring/
├── SKILL.md                    # Complete documentation
├── scripts/
│   ├── run-assessment.sh       # Main runner
│   └── generate-html-report.js # Report generator
└── resources/templates/        # Templates

tests/testability-scoring/
├── testability-scoring.spec.js # Test implementation
└── config.js                   # Configuration

tests/reports/
├── testability-results-*.json  # Results
└── testability-report-*.html   # Reports
```

## Common Commands

```bash
# Disable auto-open
AUTO_OPEN=false ./scripts/run-assessment.sh https://example.com/

# With timeout
timeout 180 ./scripts/run-assessment.sh https://example.com/

# Specific browser
./scripts/run-assessment.sh https://example.com/ firefox
```

## Credits

Based on James Bach and Michael Bolton's *Heuristics for Software Testability*  
Implementation: https://github.com/fndlalit/testability-scorer
