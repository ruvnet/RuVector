# ruvector package guidance

Source: `npm/packages/ruvector` in `ruvnet/RuVector`. Maintainer metadata is the
`author` and issue tracker in `package.json`; do not infer release authority.

Follow the checked-in [shared improvement policy](../../../docs/development/package-improvement-policy.md)
and repository `CLAUDE.md` / contribution policy. In an npm artifact, consult the
same policy in the package's source repository at the release commit; the relative
link above is repository-relative and does not imply that policy is bundled.

## Package contract

- JavaScript/declaration entries are `dist/index.js` and `dist/index.d.ts`; the
  `ruvector` executable is `bin/cli.js`. The inspected source requires Node >=20.
- Preserve the documented vector database API, CLI behavior, native backend
  loading and optional-backend fallbacks. Keep required, optional and peer
  dependency relationships explicit; do not add edges merely to connect a graph.
- Source 0.3.3 and an installed 0.2.41 consumer are different versions. Passing
  tests against one does not validate the other.

## Validation and rollback

From this package directory, run `npm run build`, `npm run verify-dist` and
`npm test` with installed, recorded dependencies. The test script covers native
integration, backend fallback, CLI/database/MCP workflows and cleanup/security
boundaries; do not weaken tests or omit failing applicable checks.

`npm pack --ignore-scripts --json` is only an artifact-inclusion / size inspection.
It deliberately skips prepack: a source checkout without generated `dist/` is not
a validated release. Preserve the existing prepack and prepublish verification.

Record source hashes, exact dependency versions, platform, controlled-cache
consumer measurements and reverse-patch / revert rollback. Persistence changes
require write, exit, restart and retrieval without sacrificing existing data.
Do not auto-publish, merge or bypass existing signing and release gates.
