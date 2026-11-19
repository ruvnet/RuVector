# ADR-362: npm provenance for RuVector packages

- **Status**: Accepted; workflow change implemented, first provenance-bearing publish pending
- **Date**: 2026-11-19
- **Related**: [ADR-356](./ADR-356-portfolio-distribution-seo.md) (canonical `repository` fields)

## Context

Consumers want to verify that a RuVector npm tarball was built from this
repository by GitHub Actions, rather than uploaded from a developer machine.
npm supports this with SLSA provenance. When `npm publish` runs on a
GitHub-hosted runner with an OIDC token and `--provenance`, npm signs a
provenance statement through Sigstore. The statement records the repository,
commit, workflow file and run. The registry attaches it to the version.

Before this change only `protected-release.yml` (`@ruvector/core` and its
`ruvector-core-*` platform binaries) passed `--provenance`. Every other
publishing workflow uploaded tarballs without it. As of 2026-11-19 no `latest`
version of any `@ruvector/*` package or of `ruvector` carries an attestation;
`@ruvector/core@0.1.38` and `ruvector-core-*@0.1.35` are the only exceptions.

## Decision

Every CI job that publishes to npm declares, at job level:

```yaml
permissions:
  contents: read
  id-token: write
env:
  NPM_CONFIG_PROVENANCE: 'true'
```

The environment variable is used instead of a `--provenance` flag. It reaches
every `npm publish` the job runs, including those inside
`scripts/ci/npm-publish-if-missing.sh`,
`npm/packages/graph-node/scripts/publish-platforms.js` and `napi prepublish`.
Platform binary packages that the workflows generate on the fly are therefore
covered as well as the wrapper packages. `--dry-run` publishes skip the
signing step, so dry runs are unaffected.

The jobs changed are the `publish` job in each of these workflows:
`build-attention`, `build-diskann`, `build-gnn`, `build-graph-node`,
`build-graph-transformer`, `build-kge`, `build-router`, `build-tiny-dancer`,
`build-typesafe`, `publish-rvagent-wasm`, `ruvllm-build`, `ruvllm-native` and
`sona-napi`. The `publish-npm` job in `publish-all` is also changed. For
`ruvector-publish`, only the env var was added, because its workflow-level
permissions already grant `id-token: write` for GCP federation.
`protected-release` was already compliant.

The repository's default workflow token is read-only, so `contents: read` takes
no access away from these jobs. All of them already run on GitHub-hosted
runners (`ubuntu-24.04` or `ubuntu-latest`), which provenance requires.
Self-hosted runners cannot produce provenance.

### `repository` field

npm rejects a provenance publish (E422) if `repository.url` in `package.json`
does not match the repository named in the OIDC token. Every package published
by these workflows has a field pointing at `github.com/ruvnet/ruvector` or
`github.com/ruvnet/RuVector`. The registry compares the two without regard to
case. Both spellings are already accepted from this repository:
`@ruvector/core@0.1.38` uses `ruvector.git` and `ruvector-core-*@0.1.35` use
`RuVector.git`, and both carry attestations. No casing was changed, so the
ADR-356 manifest check (`scripts/ci/check-platform-manifests.sh`, which expects
`RuVector.git` on platform manifests) still holds. A new package must have a
`repository.url` that points at this repository, or its first CI publish fails.

## Consequences

- **Laptop publishes get no provenance.** A version published from a laptop
  has no attestation, and nothing can add one later. Provenance exists only on
  versions published by these workflows after this change. Most of the
  repository's npm packages have no CI publish path today (the `rvf*` family,
  the `*-wasm` packages built outside `publish-all`, `@ruvector/cli`,
  `@ruvector/wasm`, `@ruvector/node`, `rvlite`, `ruvbot`, `agentic-synth`, and
  others). Those need a workflow before they can carry provenance.
- Publishing a CI-built package by hand after this change produces a
  provenance gap. Downstream verification will treat that version as
  unattested.
- If a platform package was published by hand at version `X`, the
  `npm-publish-if-missing.sh` idempotency check skips it at `X`. That version
  stays unattested until the next version bump.

## Verifying a package

From a project that depends on RuVector packages:

```bash
npm audit signatures
```

This reports `verified registry signatures` for every package and
`verified attestations` for those that have provenance.

For a single version, the registry attestations endpoint returns the Sigstore
bundles:

```bash
curl -s https://registry.npmjs.org/-/npm/v1/attestations/@ruvector%2fcore@0.1.38
# or: https://registry.npmjs.org/-/npm/v1/attestations/<pkg>@<ver>
```

A 404 means that version has no provenance. The SLSA predicate
(`https://slsa.dev/provenance/v1`) names the repository
(`https://github.com/ruvnet/RuVector`), the workflow path and the commit. The
npm website shows the same data as a "Provenance" section on the version page.
`npm view <pkg>@<ver> dist.attestations` prints the attestation URL when one
exists.
