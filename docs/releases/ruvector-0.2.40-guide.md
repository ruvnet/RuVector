# RuVector 0.2.40: MetaHarness, Darwin, and Flywheel

RuVector 0.2.40 makes its research and optimization control plane available
through the npm SDK, CLI, and MCP server. The integration uses pinned, direct
dependencies so an installed package has the same capability set that was
tested for the release.

## Install

```bash
npm install ruvector@0.2.40
```

Node.js 20 or newer is required.

## Check the installation

```bash
npx ruvector harness doctor --json
```

The doctor loads and checks all nine pinned capabilities:

- MetaHarness core
- Darwin
- Flywheel
- algorithmic harness kernel
- cost-aware router
- Red/Blue safety controls
- Weight-EFT reward-hack detection
- workspace lens
- workspace probe

## Use the SDK

All new APIs are exported from the main `ruvector` package:

```js
const {
  getMetaHarnessCapabilities,
  routeWithMetaHarness,
  evaluateMetaHarnessPromotion,
  verifyMetaHarnessReplay,
} = require('ruvector');

const capabilities = await getMetaHarnessCapabilities();

const route = await routeWithMetaHarness({
  rows: [
    {
      embedding: [1, 0],
      scores: { economical: 0.91, frontier: 0.99 },
    },
  ],
  prices: { economical: 1, frontier: 20 },
  queryEmbedding: [1, 0],
  qualityBar: 0.9,
});

console.log(route);
// { id: "economical", predictedQuality: 0.91, costPerMTok: 1, metBar: true }
```

Additional SDK entry points support:

- Flywheel generation runs and signed replay verification
- explicit Darwin evolution runs
- frozen promotion-gate evaluation
- algorithmic harness construction
- workspace receipt scoring
- reward-hack trajectory scanning
- live-credential guards

Darwin will not execute unless the caller explicitly supplies
`{ execute: true }`.

## Use the CLI

### Route by quality and cost

Prepare three JSON files:

```json
[
  {
    "embedding": [1, 0],
    "scores": {
      "economical": 0.91,
      "frontier": 0.99
    }
  }
]
```

```json
{
  "economical": 1,
  "frontier": 20
}
```

```json
[1, 0]
```

Then route:

```bash
npx ruvector harness route \
  --examples examples.json \
  --prices prices.json \
  --query query.json \
  --quality-bar 0.9
```

### Verify research evidence

```bash
npx ruvector harness flywheel verify replay-bundle.json
npx ruvector harness flywheel gate promotion-evidence.json
```

### Run Darwin explicitly

```bash
npx ruvector harness darwin evolution-config.json --execute
```

The `--execute` requirement is deliberate because Darwin candidates can run
code. Review the configuration and sandbox policy before authorizing a run.

## Use MCP safely

Start the MCP server with the curated read-only profile:

```bash
RUVECTOR_MCP_PROFILE=readonly npx ruvector mcp start
```

The profile exposes six non-executing MetaHarness tools:

- `metaharness_status`
- `metaharness_route`
- `metaharness_replay_verify`
- `metaharness_flywheel_gate`
- `metaharness_workspace_probe`
- `metaharness_reward_hack_scan`

Darwin execution and Flywheel mutation loops are intentionally not exposed over
MCP. They remain explicit SDK/CLI authority boundaries.

## Upgrade notes

- Node.js 20+ is now required.
- MetaHarness packages are direct dependencies and are installed with RuVector.
- Dependencies are loaded lazily, keeping ordinary RuVector startup fast.
- Existing vector database APIs remain available from the same package entry
  point.

## Verification

The published tarball was installed into a clean project before release. The
release passed the full npm suite, SDK/CLI/MCP integration checks, MCP handshake
and policy checks, distribution verification, startup-budget checks, and a
production dependency audit with zero reported vulnerabilities.

- npm: https://www.npmjs.com/package/ruvector/v/0.2.40
- repository: https://github.com/ruvnet/RuVector
