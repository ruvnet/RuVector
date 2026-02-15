# Six Thinking Hats for Testing

A comprehensive Claude Code skill that applies Edward de Bono's Six Thinking Hats methodology to software testing contexts.

## Quick Overview

**What It Does**: Enables structured exploration of quality concerns from six distinct perspectives (facts, emotions, risks, benefits, creativity, and process) to uncover blind spots and make better testing decisions.

**When to Use**:
- Designing test strategies for new features
- Conducting test retrospectives
- Analyzing test failures or production incidents
- Evaluating testing approaches
- Facilitating testing discussions with teams
- Overcoming analysis paralysis in testing decisions

## The Six Hats

| Hat | Focus | Testing Application | Time |
|-----|-------|---------------------|------|
| 🤍 **White** | Facts & Data | Test metrics, coverage, defect data | 5 min |
| ❤️ **Red** | Feelings & Intuition | Gut instincts about quality, confidence | 3 min |
| 🖤 **Black** | Risks & Problems | What could go wrong, coverage gaps | 7 min |
| 💛 **Yellow** | Benefits & Opportunities | Testing strengths, quick wins | 5 min |
| 💚 **Green** | Creativity & Alternatives | Innovative test approaches | 7 min |
| 🔵 **Blue** | Process & Organization | Test strategy, action plan | 5 min |

**Total**: 30 minutes for solo session, 60-90 minutes for team session

## Quick Start

### Solo Session (30 minutes)

```bash
# 1. Copy the template
cp .claude/skills/six-thinking-hats/resources/templates/solo-session-template.md my-analysis.md

# 2. Define your testing focus
# Example: "Test strategy for user authentication feature"

# 3. Work through each hat sequentially (use timer!)
# White Hat (5 min): List facts, metrics, data
# Red Hat (3 min): Capture gut feelings, no justification
# Black Hat (7 min): Identify risks, gaps, problems
# Yellow Hat (5 min): Find strengths, opportunities
# Green Hat (7 min): Generate creative testing ideas
# Blue Hat (5 min): Create action plan

# 4. Synthesize into actionable test plan
```

### Team Session (90 minutes)

```bash
# 1. Copy the team template
cp .claude/skills/six-thinking-hats/resources/templates/team-session-template.md team-session.md

# 2. Pre-session: Gather data (White Hat prep)

# 3. Run session with strict time boundaries
# - Blue Hat opening: 5 min (set context)
# - White Hat: 10 min (collect facts round-robin)
# - Red Hat: 5 min (silent reflection + sharing)
# - Black Hat: 12 min (brainstorm risks)
# - Yellow Hat: 8 min (identify opportunities)
# - Green Hat: 15 min (rapid ideation)
# - Blue Hat: 10 min (synthesize action plan)

# 4. Post-session: Document and share (20 min)
```

## File Structure

```
.claude/skills/six-thinking-hats/
├── SKILL.md                           # Main skill documentation
├── README.md                          # This file (human-readable overview)
├── resources/
│   ├── templates/
│   │   ├── solo-session-template.md  # Template for individual use
│   │   └── team-session-template.md  # Template for team sessions
│   └── examples/
│       └── api-testing-example.md    # Complete worked example
```

## When to Use Which Format

### Solo Session (30 min)
✅ **Use when**:
- Quick decision needed
- Individual test analysis
- Personal exploration
- Preparing for team discussion

### Team Session (90 min)
✅ **Use when**:
- Complex testing challenges
- Cross-functional alignment needed
- Multiple perspectives valuable
- Shared understanding required

### Async Session (2-3 days)
✅ **Use when**:
- Distributed team
- Deep thinking needed
- Time zone challenges
- Written documentation desired

## Integration with Other QE Skills

Works exceptionally well with:
- `context-driven-testing` - Choose practices based on context
- `risk-based-testing` - Prioritize using Black Hat
- `exploratory-testing-advanced` - Use Green Hat for creative charters
- `holistic-testing-pact` - Comprehensive quality model (all hats)

## Key Benefits

1. **Uncovers Blind Spots**: Each hat reveals different insights
2. **Balances Perspectives**: Black Hat pessimism balanced by Yellow Hat optimism
3. **Structures Discussion**: Prevents chaotic brainstorming
4. **Builds Consensus**: Team alignment through shared process
5. **Saves Time**: 30-90 min investment prevents days of rework
6. **Validates Intuition**: Red Hat legitimizes gut feelings
7. **Drives Creativity**: Green Hat forces innovative thinking

## Common Pitfalls to Avoid

❌ **Hat Mixing**: Combining perspectives (e.g., "Tests are passing (White) but I'm worried (Red)")
- ✅ **Solution**: Strict separation, use timer

❌ **Justifying Red Hat**: Rationalizing feelings instead of trusting intuition
- ✅ **Solution**: "I feel X" not "I feel X because..."

❌ **Skipping Hats**: "We don't need Green Hat"
- ✅ **Solution**: Every hat reveals insights, wear all six

❌ **Rushing**: 5 minutes for all hats
- ✅ **Solution**: Minimum 5 min per hat, use timer

❌ **Judging Contributions**: Criticizing ideas during Green Hat
- ✅ **Solution**: All ideas valid, evaluate later in Blue Hat

## Success Metrics

A successful Six Hats session should produce:
- ✅ 10+ facts documented (White Hat)
- ✅ Honest feelings captured (Red Hat)
- ✅ 5+ risks identified (Black Hat)
- ✅ 3+ opportunities found (Yellow Hat)
- ✅ 5+ creative ideas generated (Green Hat)
- ✅ Clear action plan with owners (Blue Hat)

## Resources

### Templates
- `resources/templates/solo-session-template.md` - Individual analysis
- `resources/templates/team-session-template.md` - Team facilitation

### Examples
- `resources/examples/api-testing-example.md` - Complete worked example with real test strategy

### Further Reading
- **SKILL.md** - Complete documentation with all use cases
- **"Six Thinking Hats" by Edward de Bono** - Original methodology
- **"Context-Driven Testing"** - Related testing philosophy

## Quick Tips

1. **Use Timer**: Strict time boundaries improve quality
2. **Start with Black Hat**: Teams love identifying risks
3. **Trust Red Hat**: Intuition catches what analysis misses
4. **Go Wild in Green Hat**: No idea too crazy
5. **Close with Blue Hat**: Always end with action plan
6. **Document Everything**: Especially Green Hat wild ideas
7. **Practice Solo First**: Get comfortable before facilitating teams

## Support

- Main documentation: `SKILL.md`
- Templates: `resources/templates/`
- Examples: `resources/examples/`
- Related skills: `context-driven-testing`, `risk-based-testing`, `exploratory-testing-advanced`

---

**Created**: 2025-11-13
**Category**: Testing Methodologies
**Difficulty**: Intermediate
**Best Used With**: context-driven-testing, risk-based-testing, exploratory-testing-advanced
