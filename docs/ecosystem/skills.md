# RuVector skill discovery

The `ruvector-constellation` skill selects npm packages and connects their working capabilities to an application or agent. It covers Node vector search, browser WASM, RVF, MCP discovery, and optional Ruflo and MetaHarness integration.

## Availability and installation

This skill is introduced on `feat/npm-constellation-20261011`. Until that branch is pushed or merged, use the local checkout; the GitHub default-branch command below will not discover this new skill.

From the repository root, install into the current project's Codex and Claude Code skill locations:

```bash
npx skills add . --skill ruvector-constellation --agent codex --agent claude-code
```

After this change is merged into the public repository's default branch:

```bash
npx skills add ruvnet/ruvector --skill ruvector-constellation --agent codex --agent claude-code
```

Inspect available skills before installing from a remote revision:

```bash
npx skills add ruvnet/ruvector --list
```

The source is [SKILL.md](../../skills/ruvector-constellation/SKILL.md). Installation uses the [upstream Skills CLI](https://github.com/vercel-labs/skills/blob/main/README.md). Project installation is the default; choose a global installation explicitly if desired. Installing the skill does not install RuVector, modify an MCP configuration, or activate a plugin. The skill guides those subsequent integrations within the user's task.

## Package and MCP entry points

Start with `npm view <package> version engines repository --json` and the project's existing lockfile. Install the required package at an exact compatible version. For the full facade, `npm install --save-exact ruvector` provides the `ruvector` binary. Its commands include `info`, `harness status --json`, and `mcp start`. Verify the installed release supports the command before configuring a long-lived agent.

The skill uses source-backed commands from [the CLI](../../npm/packages/ruvector/bin/cli.js), optional capability loading from [the MetaHarness bridge](../../npm/packages/ruvector/src/metaharness/index.ts), and tool restrictions from [the MCP policy](../../npm/packages/ruvector/bin/mcp-policy.js). The focused RVF server's executable and transport are declared in [its manifest](../../npm/packages/rvf-mcp-server/package.json). Ruflo's separate `mcp start` entry is documented in [its upstream README](https://github.com/ruvnet/ruflo#readme).

Use actual MCP `initialize` and `tools/list` responses for capability discovery. Keep CLI orchestration and learning adapters above native or WASM primitives. Record optional-package absence distinctly from core failure. This preserves independent package use while giving agents an explicit route to the broader ecosystem.

## Validation and coverage

Validate skill frontmatter and references, then verify local discovery with `npx skills add . --list`. A useful integration check installs into an isolated project, installs the selected runtime, exercises a vector insert/query, and initializes the configured MCP server. Test a missing optional dependency as well as the enabled integration. Do not claim remote availability, runtime compatibility, or marketplace visibility from a local discovery check alone.

Track discovery, installation, runtime smoke results, and MCP tool availability separately in recurring ecosystem reviews. Feed observed failures into narrow skill updates and reproducible regression cases; avoid publishing unsupported feature claims or adding dependencies merely to increase cross-links.

## Portable plugin artifact

The npm package includes a portable Agent Plugins 1.0 `plugin.json` and the same `skills/ruvector-constellation/SKILL.md`. This is a skills-only plugin surface; it does not silently install runtime dependencies or register an MCP process. A local export was built and inspected during validation. Public directory submission and host installation are not implied by npm publication. When creating an export, include only `plugin.json` and `skills/` inside a `ruvector-constellation` directory; exclude node_modules, credentials and unrelated repository files.
