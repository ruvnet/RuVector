# ADR-065: npm Publishing Strategy

**Status**: Accepted
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-064 (Pi Brain Infrastructure), ADR-066 (SSE MCP Transport), ADR-063 (WASM Executable Nodes)

## 1. Context

The RuVector project produces 48+ npm packages under the `@ruvector/` scope. These range from core TypeScript libraries and WASM bindings to MCP server wrappers and CLI tools. A new package, `@ruvector/pi-brain`, bundles CLI, SDK, and MCP stdio access to the Shared Brain into a single installable unit.

Without a defined publishing strategy, interdependent packages can break consumers if published in the wrong order, pre-release versions can leak into production, and TypeScript compilation errors can ship as broken packages. This ADR documents the publishing order, semver strategy, and authentication setup that govern all npm releases.

## 2. Decision

Adopt a structured publishing pipeline: categorize packages, enforce dependency-ordered publishing, use semver with pre-release tags for unstable packages, and require TypeScript compilation to succeed before any publish. All packages are published under the `ruvnet` npm account.

## 3. Architecture

### 3.1 Package Categories

| Category | Description | Examples |
|----------|-------------|---------|
| **Core** | Foundational libraries with no RuVector dependencies | `@ruvector/types`, `@ruvector/utils` |
| **WASM** | WebAssembly bindings compiled from Rust crates | `@ruvector/solver-wasm`, `@ruvector/rvf-wasm` |
| **MCP** | Model Context Protocol server packages | `@ruvector/mcp-brain`, `@ruvector/mcp-gate` |
| **CLI** | Command-line tools | `@ruvector/pi-brain`, `@ruvector/cli` |
| **Infrastructure** | Build tools, codegen, testing utilities | `@ruvector/build-tools`, `@ruvector/test-utils` |

### 3.2 The `@ruvector/pi-brain` Package

`@ruvector/pi-brain` is the primary user-facing npm package for the Shared Brain. It provides three interfaces in one package:

**CLI**: `npx @ruvector/pi-brain search "tokio deadlock"` — search the brain from the command line. `npx @ruvector/pi-brain share --category debug --title "..." --content "..."` — share knowledge.

**SDK**: Programmatic TypeScript API for integrating brain capabilities into applications. Handles authentication, embedding, and the full REST API surface.

**MCP stdio**: `@ruvector/pi-brain mcp` — starts a JSON-RPC stdio server implementing the MCP protocol. Claude Code connects via `claude mcp add pi-brain -- npx @ruvector/pi-brain mcp`.

### 3.3 Publish Order

Packages must be published in dependency order. A package can only be published after all of its `@ruvector/` dependencies have been published at the required version.

**Tier 1 — No `@ruvector/` dependencies**:
- `@ruvector/types`
- `@ruvector/utils`
- `@ruvector/build-tools`

**Tier 2 — Depends on Tier 1 only**:
- `@ruvector/solver-wasm` (depends on `@ruvector/types`)
- `@ruvector/rvf-wasm` (depends on `@ruvector/types`)
- `@ruvector/test-utils` (depends on `@ruvector/types`, `@ruvector/utils`)

**Tier 3 — Depends on Tier 1 and/or Tier 2**:
- `@ruvector/mcp-brain` (depends on `@ruvector/types`)
- `@ruvector/mcp-gate` (depends on `@ruvector/types`)
- `@ruvector/pi-brain` (depends on `@ruvector/types`, `@ruvector/mcp-brain`)

**Tier 4 — Meta-packages and aggregators**:
- `@ruvector/cli` (depends on multiple Tier 2-3 packages)

Within a tier, packages can be published in any order.

### 3.4 Semver Strategy

| Version Range | Meaning | Tag |
|---------------|---------|-----|
| `0.x.y` | Pre-1.0, breaking changes expected between minors | `latest` |
| `x.y.z-alpha.N` | Active development, not for production | `alpha` |
| `x.y.z-beta.N` | Feature-complete, testing in progress | `beta` |
| `x.y.z-rc.N` | Release candidate, final validation | `rc` |
| `x.y.z` | Stable release | `latest` |

Pre-release versions are published with explicit dist-tags:
```bash
npm publish --tag alpha
npm publish --tag beta
```

The `latest` tag is only set on stable releases. This prevents `npm install @ruvector/pi-brain` from pulling pre-release versions.

### 3.5 TypeScript Compilation Requirements

Every package with TypeScript source must pass these checks before publish:

1. `tsc --noEmit` succeeds with zero errors
2. `tsc --declaration` generates `.d.ts` files
3. `package.json` includes `types` or `typings` field pointing to the declaration entry point
4. `exports` map includes `types` condition for each entry point

Packages that fail TypeScript compilation are blocked from publishing. This is enforced by running `npm run build` (which includes `tsc`) as the first step of the publish pipeline.

## 4. Implementation

### 4.1 Authentication

npm authentication uses the `ruvnet` account. Credentials are stored in the project `.env` file and loaded into `~/.npmrc` before publishing. Verify with `npm whoami`.

The npm token is scoped to the `@ruvector/` scope with publish permissions. Read-only tokens are used in CI for install-only workflows.

### 4.2 Pre-Publish Checklist

For each package:

1. Verify `npm whoami` returns `ruvnet`
2. Run `npm run build` (TypeScript compilation + any bundling)
3. Run `npm test` (all tests must pass)
4. Verify `package.json` version matches the intended release
5. Check that `files` field in `package.json` includes only intended artifacts
6. Run `npm pack --dry-run` to inspect the tarball contents
7. Publish: `npm publish --access public`

### 4.3 WASM Package Build

WASM packages require a Rust compilation step before the npm publish:

1. `cargo build --release --target wasm32-unknown-unknown -p <crate>`
2. `wasm-bindgen` or `wasm-pack` generates the JS/TS bindings
3. Copy generated files into the npm package directory
4. Run TypeScript compilation on the wrapper code
5. Publish as a standard npm package

### 4.4 Solver Crate Publish Order (Cargo)

For the Rust solver crates published to crates.io (not npm), the order is:

1. `ruvector-solver` first (no dependencies)
2. `ruvector-solver-wasm` second (depends on `ruvector-solver`)
3. `ruvector-solver-node` third (depends on `ruvector-solver`)

Always run `cargo publish --dry-run --allow-dirty` before real publish. `ruvector-profiler` has `publish = false` and is intentionally not publishable.

### 4.5 Version Coordination

When a breaking change occurs in a Tier 1 package, all dependent packages must be updated and republished. The procedure is:

1. Publish the updated Tier 1 package with the new major/minor version
2. Update `package.json` in all dependent packages to reference the new version
3. Run `npm install` in each dependent package to verify resolution
4. Run tests in each dependent package
5. Publish dependent packages in tier order

For non-breaking changes (patch versions), only the changed package needs republishing. Dependents using caret ranges (`^x.y.z`) automatically resolve the new patch.

### 4.6 Package.json Standards

All `@ruvector/` packages must include:

```json
{
  "name": "@ruvector/<package>",
  "version": "x.y.z",
  "license": "MIT OR Apache-2.0",
  "repository": {
    "type": "git",
    "url": "https://github.com/ruvnet/ruvector"
  },
  "engines": { "node": ">=18" },
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js"
    }
  },
  "files": ["dist/", "README.md", "LICENSE"]
}
```

The `files` field is explicitly set to prevent accidental inclusion of source maps, test fixtures, `.env` files, or other development artifacts in the published tarball.

### 4.7 CI Integration

The publish pipeline runs in GitHub Actions:

1. **Trigger**: Manual workflow dispatch with package name and version as inputs
2. **Auth**: npm token from GitHub Secrets, loaded into `~/.npmrc`
3. **Build**: `npm run build` in the package directory
4. **Test**: `npm test` in the package directory
5. **Dry run**: `npm pack --dry-run` to verify tarball contents
6. **Publish**: `npm publish --access public` (or `--tag alpha/beta` for pre-releases)
7. **Verify**: `npm view @ruvector/<package> version` confirms the published version

The workflow rejects publishes if the version already exists on the registry (npm returns 403 for duplicate versions).

## 5. Consequences

### Positive

- **No broken installs**: Dependency-ordered publishing ensures consumers never pull a package whose dependencies are not yet available
- **Safe defaults**: Pre-release tags prevent accidental production use of unstable versions
- **Type safety**: Mandatory TypeScript compilation catches type errors before they reach consumers
- **Single account**: All packages under `ruvnet` with `@ruvector/` scope provides consistent ownership and discoverability
- **Explicit file lists**: The `files` field prevents credential leaks and keeps tarballs small

### Negative

- **Manual coordination**: Publishing 48+ packages in order requires discipline. Automation (a publish script that resolves the dependency graph and publishes in topological order) is deferred but recommended.
- **WASM build complexity**: WASM packages require both Rust and Node.js toolchains. Build failures in either chain block the publish.
- **ESM-only**: All packages use `"type": "module"`. CommonJS consumers must use dynamic `import()` or a bundler. This is a deliberate choice — the ecosystem is moving to ESM and dual-packaging adds complexity.

### Neutral

- The `@ruvector/pi-brain` package combining CLI + SDK + MCP in one package increases the install size but simplifies the getting-started experience. Users who only need the SDK can tree-shake unused CLI code.
- Node.js >= 18 is required. This matches the current LTS baseline and enables native `fetch`, `structuredClone`, and other modern APIs.
