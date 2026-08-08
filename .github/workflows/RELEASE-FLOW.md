# RuVector Release Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRIGGER RELEASE PIPELINE                     │
│                                                                  │
│  Method 1: git tag v0.1.3 && git push origin v0.1.3            │
│  Method 2: Manual workflow_dispatch with version input          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: VALIDATION                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • cargo fmt --check                                       │ │
│  │  • cargo clippy (all warnings as errors)                  │ │
│  │  • cargo test --workspace                                 │ │
│  │  • npm run test:unit                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    Runner: ubuntu-22.04                          │
│                    Time: 3-12 minutes                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│  STAGE 2: BUILD CRATES    │    │  STAGE 3: BUILD WASM      │
│  ┌─────────────────────┐  │    │  ┌─────────────────────┐  │
│  │ • Build 26 crates   │  │    │  │ • ruvector-wasm     │  │
│  │ • Dependency order  │  │    │  │ • ruvector-gnn-wasm │  │
│  │ • Release mode      │  │    │  │ • ruvector-graph-   │  │
│  │ • Run tests         │  │    │  │   wasm              │  │
│  └─────────────────────┘  │    │  │ • tiny-dancer-wasm  │  │
│  ubuntu-22.04             │    │  └─────────────────────┘  │
│  5-20 minutes             │    │  ubuntu-22.04             │
└───────────────────────────┘    │  4-15 minutes             │
            │                    └───────────────────────────┘
            │                                   │
            └──────────┬────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 4: BUILD NATIVE (Parallel Matrix)             │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Platform 1        Platform 2        Platform 3             ││
│  │  linux-x64-gnu     linux-arm64-gnu   darwin-x64             ││
│  │  ubuntu-22.04      ubuntu-22.04      macos-13               ││
│  │  ┌──────────┐      ┌──────────┐      ┌──────────┐          ││
│  │  │ napi-rs  │      │ napi-rs  │      │ napi-rs  │          ││
│  │  │ build    │      │ + cross  │      │ build    │          ││
│  │  │          │      │ compile  │      │          │          ││
│  │  └──────────┘      └──────────┘      └──────────┘          ││
│  │                                                              ││
│  │  Platform 4        Platform 5                               ││
│  │  darwin-arm64      win32-x64-msvc                           ││
│  │  macos-14          windows-2022                             ││
│  │  ┌──────────┐      ┌──────────┐                            ││
│  │  │ napi-rs  │      │ napi-rs  │                            ││
│  │  │ build    │      │ build    │                            ││
│  │  │          │      │          │                            ││
│  │  └──────────┘      └──────────┘                            ││
│  └──────────────────────────────────────────────────────────────┘│
│  Time: 3-12 minutes per platform (runs in parallel)             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌────────────────────────┐           ┌────────────────────────┐
│ STAGE 5: PUBLISH       │           │ STAGE 6: PUBLISH       │
│ RUST CRATES            │           │ npm PACKAGES           │
│                        │           │                        │
│ Publishing Order:      │           │ Publishing Order:      │
│ 1. ruvector-           │           │ 1. Platform packages   │
│    turboquant          │           │    (@ruvector/core-*)  │
│ 2. ruvector-core       │           │                        │
│ 3. metrics             │           │ 2. @ruvector/wasm      │
│ 4. filter              │           │ 3. @ruvector/cli       │
│ 5. snapshot            │           │ 4. @ruvector/          │
│ ... (27 total)         │           │    extensions          │
│                        │           │ 5. @ruvector/core      │
│                        │           │                        │
│ Target: crates.io      │           │ Target: npmjs.com      │
│ Auth: CARGO_REGISTRY_  │           │ Auth: NPM_TOKEN        │
│       TOKEN            │           │                        │
│ Time: 5-10 minutes     │           │ Time: 2-5 minutes      │
└────────────────────────┘           └────────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 7: CREATE GITHUB RELEASE                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Download all artifacts (native + WASM)                 │ │
│  │  2. Package as .tar.gz files:                              │ │
│  │     - ruvector-native-linux-x64-gnu.tar.gz                 │ │
│  │     - ruvector-native-linux-arm64-gnu.tar.gz               │ │
│  │     - ruvector-native-darwin-x64.tar.gz                    │ │
│  │     - ruvector-native-darwin-arm64.tar.gz                  │ │
│  │     - ruvector-native-win32-x64-msvc.tar.gz                │ │
│  │     - ruvector-wasm.tar.gz                                 │ │
│  │  3. Generate comprehensive release notes                   │ │
│  │  4. Create GitHub release with artifacts                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  Time: 2-3 minutes                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 8: RELEASE SUMMARY                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Generate final summary with:                              │ │
│  │  • Status of all jobs (success/failure)                    │ │
│  │  • Links to published packages                             │ │
│  │  • Verification steps                                      │ │
│  │  • Next steps for maintainers                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│  Always runs (even on failure)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RELEASE COMPLETE! 🎉                          │
│                                                                  │
│  Published to:                                                   │
│  ✅ crates.io: https://crates.io/crates/ruvector-core           │
│  ✅ npmjs.com: https://www.npmjs.com/package/@ruvector/core     │
│  ✅ GitHub: https://github.com/ruvnet/ruvector/releases         │
│                                                                  │
│  Total Time: 15-30 minutes (with caching)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🚀 Parallel Execution
- Stages 2, 3, and 4 run simultaneously
- 5 native platform builds run in parallel
- Total time: ~60% faster than sequential

### 💾 Smart Caching
- Rust dependencies cached via `Swatinem/rust-cache`
- npm dependencies cached via `actions/setup-node`
- wasm-pack binary cached
- Cache hit rate: 70-95%

### 🔒 Security
- Secrets never exposed in logs
- Environment protection for production
- Optional reviewer approval gates
- Conditional publishing (tag or manual only)

### 🛡️ Error Handling
- Continue on already-published packages
- Graceful failure handling
- Rate limiting protection (10s between publishes)
- Comprehensive error logging

### 📊 Monitoring
- Job summaries at each stage
- Final comprehensive summary
- Artifact upload/download tracking
- GitHub release with all binaries

## Workflow Dependencies

```
┌──────────┐
│ validate │──┐
└──────────┘  │
              ├──> build-crates ──┐
              │                    │
              ├──> build-wasm ─────┤
              │                    ├──> publish-crates ──┐
              └──> build-native ───┤                     │
                                   ├──> publish-npm ─────┤
                                   │                     │
                                   └─────────────────────┴──> create-release
                                                              │
                                                              └──> release-summary
```

## Critical Paths

### Path 1: Rust Publishing
```
validate → build-crates → publish-crates → create-release
```
**Time**: 15-25 minutes

### Path 2: npm Publishing
```
validate → build-native → publish-npm → create-release
         → build-wasm ─┘
```
**Time**: 12-20 minutes

### Path 3: Release Creation
```
All paths → create-release → release-summary
```
**Time**: 2-3 minutes

## Artifact Flow

```
┌──────────────┐
│ build-native │──> bindings-linux-x64-gnu.artifact
│              │──> bindings-linux-arm64-gnu.artifact
│              │──> bindings-darwin-x64.artifact
│              │──> bindings-darwin-arm64.artifact
│              │──> bindings-win32-x64-msvc.artifact
└──────────────┘
                     │
                     ├──> publish-npm (downloads & publishes)
                     │
                     └──> create-release (downloads & packages)

┌──────────────┐
│  build-wasm  │──> wasm-packages.artifact
└──────────────┘
                     │
                     ├──> publish-npm (downloads & publishes)
                     │
                     └──> create-release (downloads & packages)
```

## Environment Variables

| Variable | Scope | Purpose |
|----------|-------|---------|
| `CARGO_TERM_COLOR` | Global | Colored Cargo output |
| `RUST_BACKTRACE` | Global | Detailed error traces |
| `CARGO_REGISTRY_TOKEN` | publish-crates | crates.io auth |
| `NODE_AUTH_TOKEN` | publish-npm | npmjs.com auth |
| `GITHUB_TOKEN` | create-release | GitHub API auth |

## Job Conditions

| Job | Runs When |
|-----|-----------|
| `validate` | Always (unless skip_tests=true) |
| `build-crates` | After validation passes |
| `build-wasm` | After validation passes |
| `build-native` | After validation passes |
| `publish-crates` | Tag push OR manual + not dry_run |
| `publish-npm` | Tag push OR manual + not dry_run |
| `create-release` | All builds succeed + tag OR manual |
| `release-summary` | Always (even on failure) |

## Quick Start Commands

```bash
# Test the workflow locally (dry run)
gh workflow run release.yml \
  -f version=0.1.3-test \
  -f dry_run=true

# Trigger production release
git tag v0.1.3
git push origin v0.1.3

# Emergency release (skip tests)
gh workflow run release.yml \
  -f version=0.1.3 \
  -f skip_tests=true

# View workflow status
gh run list --workflow=release.yml
```

## Support Matrix

| Component | Platforms | Total |
|-----------|-----------|-------|
| Native Binaries | linux-x64, linux-arm64, darwin-x64, darwin-arm64, win32-x64 | 5 |
| WASM Packages | Universal (wasm32-unknown-unknown) | 4 |
| Rust Crates | Platform-independent source | 26 |
| npm Packages | 5 platform + 4 core | 9 |

**Total Release Artifacts**: 44 packages across 3 registries
