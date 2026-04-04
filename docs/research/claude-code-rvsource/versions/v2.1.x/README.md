# Claude Code v2.1.91 — Decompiled by ruDevolution

**Latest version** — 34,759 declarations, 981 Louvain-detected modules, 100% valid JS.

## Pipeline Stats

| Phase | Time | Result |
|-------|------|--------|
| Parse | 6.6s | 34,759 declarations |
| Graph | 0.7s | 599,034 reference edges |
| Partition | 1.2s | 981 modules (Louvain) |
| Infer | 13.8s | 32,091 names (HIGH=1,436) |
| Validate | auto | 100% parse rate |

## Key Discoveries (vs v2.0)

| Feature | Evidence |
|---------|----------|
| 🤖 Agent Teams | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, `CLAUDE_CODE_TEAMMATE_COMMAND` |
| 🌙 Auto Dream Mode | `tengu_auto_dream_completed`, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` |
| 🔮 Unreleased Models | `claude-opus-4-6`, `claude-sonnet-4-6` |
| 🔐 Amber Codenames | `tengu_amber_flint`, `amber_prism`, `amber_stoat`, `amber_wren` |
| 🧰 Advisor Tool | `tengu_advisor_tool_call` |
| 🧰 Agentic Search | `tengu_agentic_search_cancelled` |
| 📋 Tasks System | `CLAUDE_CODE_ENABLE_TASKS`, `CLAUDE_CODE_TASK_LIST_ID` |
| ⏰ Cron Scheduling | `CLAUDE_CODE_DISABLE_CRON` |
| 🧠 Plan V2 Interviews | `CLAUDE_CODE_PLAN_MODE_INTERVIEW_PHASE` |
| 🏪 Plugin Marketplace | 7 new `PLUGIN_*` env vars |
| 📡 MCP Streamable HTTP | `/v1/toolbox/shttp/mcp/` |
| 🔢 117 new env vars | Total since v2.0.62 |

## Growth

```
v0.2.126 ████░░░░░░░░░░░░░░░░  6.9 MB   13,869 funcs
v1.0.128 ██████░░░░░░░░░░░░░░  8.9 MB   16,593 funcs
v2.0.77  ████████░░░░░░░░░░░░ 10.5 MB   20,395 funcs
v2.1.91  ████████████████████ 13.2 MB   34,759 decls  ← THIS
```

## Files

- `source/` — decompiled modules by category
- `modules-manifest.json` — all 661 modules with sizes
- `witness.json` — SHA3-256 Merkle proof
- `metrics.json` — pipeline metrics

## Full 981-module decompile

Too large for git (638MB). Generate locally:

```bash
cargo run --release -p ruvector-decompiler --example run_on_cli -- \
  $(npm root -g)/@anthropic-ai/claude-code/cli.js --output-dir ./decompiled
```

Or download: [github.com/ruvnet/rudevolution/releases/tag/v0.1.0-claude-code-v2.1.91](https://github.com/ruvnet/rudevolution/releases/tag/v0.1.0-claude-code-v2.1.91)
