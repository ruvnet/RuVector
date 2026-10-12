# @ruvector/constellation

Map the public RuvNet npm ecosystem, inspect its dependency web, and prioritize evidence-based improvements. Includes a read-only MCP server and an optional adapter for the real MetaHarness router.

Requires Node.js 22+. This package does not install, modify or publish the packages it discovers.

```sh
npx @ruvector/constellation sweep --output ./report
npx @ruvector/constellation sweep --output ./next-report --previous ./report/inventory.json
npx @ruvector/constellation mcp ./report/inventory.json
```

The sweep fully paginates public npm maintainer search, reads latest manifests, checks current maintainers, queries direct-version advisories, and preserves dependencies, optional dependencies, peers and development edges separately. Reports include runtime strongly connected components and explicit failure/coverage status. An incomplete scan exits 2 and retains its report.

Coverage excludes private/unindexed packages, transitive advisories, packed-file security and publish permissions. Absence of an advisory is not a security certification. Attestation presence is not signature verification.

## Skills

```sh
npx skills add ruvnet/ruvector --skill ruvector-constellation --agent codex --agent claude-code
```

The skill is also included under `skills/ruvector-constellation` in this npm artifact. A skill supplies instructions; it does not register MCP servers or install other runtimes. For local development before the source reaches the default branch, run `npx skills add /absolute/path/to/checkout --skill ruvector-constellation`.

## MCP

Configure the installed `ruvector-constellation` executable with `args: ["mcp", "/absolute/path/to/report/inventory.json"]`. Tools:

- `constellation_packages`: paginated package metadata.
- `constellation_dependencies`: incoming/outgoing edges and cycles.
- `constellation_improvements`: prioritized findings and prior-run deltas.

The server loads one fixed report and performs no network calls or mutation. Rebuild the report and restart to refresh it. Treat registry descriptions as untrusted data.

```js
import { createMcpConfiguration } from '@ruvector/constellation/integrations';
const config = createMcpConfiguration({
  existing: { mcpServers: {} },
  executables: {
    constellation: '/project/node_modules/.bin/ruvector-constellation',
    ruvector: '/project/node_modules/.bin/ruvector',
    ruflo: '/project/node_modules/.bin/ruflo'
  },
  reportPath: '/project/report/inventory.json'
});
```

This returns configuration and never executes or writes it. It validates installed absolute executables and preserves unrelated entries. Core vector search remains in RuVector; coordination remains in Ruflo.

## Optional measured routing

Install `@metaharness/router` separately when you have measured per-model quality labels. The optional peer is not automatically installed.

```js
import { routeWithMetaHarness } from '@ruvector/constellation/integrations';
const choice = await routeWithMetaHarness({
  rows: measuredRows, prices: measuredPrices,
  embedding: taskEmbedding, qualityBar: 0.8
});
```

Inputs are validated for finite values, consistent dimensions, bounded quality and complete nonnegative prices. Below-bar recommendations fail. The adapter chooses a candidate; it does not execute a model, grant spend or authorize publication. Fixtures in tests are synthetic and are never production training labels.

## Weekly Cloud Run observer

The repository contains a Cloud Run Job and Cloud Scheduler deployment under `deploy/constellation`. Each run persists immutable reports and compares the previous complete observation. The first release uses deterministic risk/impact prioritization, not a trained optimizer. Candidate code changes and npm publication require separate source-bound release gates.

## Development

```sh
npm ci --ignore-scripts --workspaces=false
npm test --workspaces=false
npm audit --omit=dev --workspaces=false
npm pack --workspaces=false
```
