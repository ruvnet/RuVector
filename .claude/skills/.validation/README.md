# Skill Validation Infrastructure

**Version**: 1.0.0
**ADR**: [ADR-056: Deterministic Skill Validation System](../../v3/implementation/adrs/ADR-056-skill-validation-system.md)

This directory contains the validation infrastructure for AQE skills - ensuring skill outputs are deterministic and trustworthy.

## Directory Structure

```
.validation/
├── schemas/                  # JSON Schema definitions
│   ├── skill-frontmatter.schema.json    # SKILL.md frontmatter validation
│   ├── skill-output.template.json       # Base output schema template
│   ├── skill-output-meta.schema.json    # Meta-schema for skill schemas
│   └── skill-eval.schema.json           # Evaluation suite schema
├── templates/                # Templates for skill authors
│   ├── validate.template.sh             # Bash validator template
│   ├── validator-lib.cjs                 # Shared validation utilities
│   ├── eval.template.yaml               # Evaluation suite template
│   ├── security-testing-eval.template.yaml  # Domain-specific example
│   └── skill-frontmatter.example.yaml   # Frontmatter example
├── examples/                 # Example skill outputs
│   ├── security-testing-output.example.json
│   └── testability-scoring-output.example.json
├── test-data/               # Test data for validation
│   ├── minimal-output.json
│   ├── sample-output.json
│   └── invalid-output.json
├── skill-validation-mcp-integration.md  # MCP integration spec
└── README.md                # This file
```

## 4-Layer Validation Architecture

| Layer | Purpose | Files |
|-------|---------|-------|
| **L0** | Intent (SKILL.md) | Declarative instructions |
| **L1** | Schema | `schemas/*.schema.json` |
| **L2** | Validator | `templates/validate.template.sh` |
| **L3** | Eval Suite | `templates/eval.template.yaml` |

## Trust Tiers

| Tier | Name | Requirements |
|------|------|--------------|
| 0 | advisory | SKILL.md only |
| 1 | structured | SKILL.md + JSON Schema |
| 2 | validated | SKILL.md + Schema + Validator |
| 3 | verified | All above + Eval Suite |
| 4 | certified | All above + CI Green + MCP Learning |

## Usage

### Validate a Skill Output

```bash
# Using the template directly
.validation/templates/validate.template.sh output.json

# With options
.validation/templates/validate.template.sh --verbose output.json
.validation/templates/validate.template.sh --json output.json  # JSON output
.validation/templates/validate.template.sh --self-test         # Self-test mode
```

### Create a Skill Validator

1. Copy `templates/validate.template.sh` to your skill's directory
2. Customize the configuration section at the top
3. Add skill-specific validation in `run_skill_specific_validation()`
4. Update the output schema path

### Create an Eval Suite

1. Copy `templates/eval.template.yaml` to your skill's directory
2. Add test cases with inputs and expected outputs
3. Define passing criteria (pass rate, required patterns)

## MCP Integration

Skills MUST use AQE MCP tools for shared learning. See `skill-validation-mcp-integration.md` for details.

Key MCP calls:
- `memory_store` - Store validation patterns
- `memory_query` - Query learned patterns
- `test_outcome_track` - Track outcomes
- `memory_share` - Share learning with fleet
- `quality_assess` - Update quality scores

## For Skill Authors

When creating a new skill:

1. **Start with SKILL.md** (Trust Tier 0)
2. **Add output schema** in `your-skill/schemas/output.schema.json` (Trust Tier 1)
3. **Add validator script** in `your-skill/validate-skill.cjs` (Trust Tier 2)
4. **Add eval suite** in `your-skill/eval.yaml` (Trust Tier 3)
5. **Enable CI + MCP learning** (Trust Tier 4)

## Validation Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 1 | Validation failed |
| 2 | Validation skipped (missing tools) |

---

*Part of AQE v3 Skill Validation System*
