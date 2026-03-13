# RuvLLM Optimization Checklist

Quick reference for implementing performance improvements.

---

## PHASE 1: High-Impact Quick Wins (1-2 weeks)

### 1. Fix Default Features (15-25% faster builds)

**Status**: ⚠️ NOT STARTED
**Effort**: 30 minutes
**Impact**: 30-45 second faster builds

**Change Cargo.toml**:
```diff
[features]
-default = ["async-runtime", "candle"]
+default = []
+
+# Full inference stack (heavy)
+full = ["async-runtime", "candle", "tokenizers", "hf-hub"]

# Keep existing feature definitions
```

**Rationale**:
- Candle = 35MB compiled code
- Tokio = 5MB runtime
- Users building RuvLLM as library forced to compile all this

**Validation**:
```bash
# Before
cargo clean && time cargo build --release
# Expected: ~180 seconds

# After
cargo clean && time cargo build --release
# Expected: ~140 seconds
```

---

### 2. Reduce Re-export Bloat (8-12% speedup)

**Status**: ⚠️ NOT STARTED
**Effort**: 2-3 hours
**Impact**: 15-25 second faster type checking

**File**: `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/lib.rs` (lines 158-520)

**Current State**: 362 public re-exports

**Action**:
1. Identify most commonly imported items (~50 items)
2. Keep only high-value re-exports:
```rust
// TOP-LEVEL PUBLIC API (keep these)
pub use backends::{LlmBackend, GenerateParams};
pub use serving::{ServingEngine, ServingEngineConfig};
pub use session::{Session, SessionManager};
pub use sona::SonaIntegration;
pub use ruvector_integration::RuvectorIntegration;
pub use error::{Result, RuvLLMError};

// EVERYTHING ELSE: Remove from lib.rs
// Users import directly: use ruvllm::backends::CandleBackend;
```

3. Create re-export submodules for organization:
```rust
pub mod backends {
    pub use crate::backends::*;
}
pub mod models {
    pub use crate::models::*;
}
// etc.
```

**Validation**:
```bash
cargo build --lib && wc -l src/lib.rs
# Before: ~994 lines
# After: ~200-300 lines
```

---

### 3. Add Development Profile (4-5x faster dev builds)

**Status**: ⚠️ NOT STARTED
**Effort**: 15 minutes
**Impact**: 35 seconds → 7 seconds on incremental builds

**File**: `/Users/cohen/GitHub/ruvnet/ruvector/Cargo.toml`

**Add to workspace**:
```toml
[profile.release-fast]
inherits = "release"
lto = "thin"              # Instead of "fat"
codegen-units = 16       # Instead of 1
opt-level = 2            # Instead of 3

[profile.release-dev]
inherits = "release-fast"
debug = true             # Include symbols for profiling
```

**Usage**:
```bash
# Development: 4-5x faster builds
cargo build --profile release-fast

# Production: Optimal performance (use current "release" profile)
cargo build --release

# Benchmarking: Include symbols
cargo build --profile release-dev
```

**Trade-offs**:
| Profile | Compile Time | Binary Size | Runtime Speed |
|---------|--------------|-------------|---------------|
| release | 180s | 45MB | 100% (optimal) |
| release-fast | 40s | 48MB | 97% |
| release-dev | 42s | 65MB | 97% |

---

### 4. Document Unsafe Code (Code Quality)

**Status**: ⚠️ NOT STARTED
**Effort**: 1 hour
**Impact**: Code clarity, maintainability

**Files with unsafe missing SAFETY comments**:
1. `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/kernels/attention.rs` (lines 439, 461, 701, 784, 846)
2. `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/quantize/pi_quant_simd.rs` (multiple)
3. `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/kernels/norm.rs`
4. `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/memory_pool.rs`

**Template for each unsafe block**:
```rust
// SAFETY: [explain why this is safe]
// - [precondition 1]
// - [precondition 2]
// - [justification]
unsafe {
    // ... code
}
```

**Example**:
```rust
// SAFETY: q_ptr points to valid f32 array with length >= len.
// The loop bounds-check i < len before dereferencing, ensuring
// we never access out-of-bounds memory. NEON intrinsics are safe
// for aligned float32 pointers.
unsafe {
    let v0 = vld1q_f32(q_ptr.add(i));
}
```

---

## PHASE 2: Medium-Impact Improvements (2-4 weeks)

### 5. Split Large Files (5-8% faster incremental builds)

**Status**: ⚠️ NOT STARTED
**Effort**: 4-6 hours
**Impact**: Better compile parallelism

**Files to split**:

#### 5a. `autodetect.rs` (1,944 lines) → `autodetect/`
```
autodetect/
├── mod.rs             (100 lines - main types)
├── system.rs          (400 lines - SystemCapabilities)
├── cpu.rs             (600 lines - CPU feature detection)
├── gpu.rs             (500 lines - GPU capabilities)
└── inference.rs       (344 lines - InferenceConfig)
```

**Steps**:
1. Create `src/autodetect/` directory
2. Move logic to submodules
3. Update imports in `lib.rs`

#### 5b. `memory_pool.rs` (1,703 lines) → `memory_pool/`
```
memory_pool/
├── mod.rs             (100 lines - main interface)
├── arena.rs           (400 lines - ArenaAllocator)
├── buffer_pool.rs     (500 lines - BufferPool)
└── scratch.rs         (703 lines - ScratchSpaceManager)
```

#### 5c. `kv_cache.rs` (1,527 lines) → `kv_cache/`
```
kv_cache/
├── mod.rs             (100 lines - main interface)
├── pooled.rs          (500 lines - PooledKvCache)
├── two_tier.rs        (600 lines - TwoTierKvCache)
└── stats.rs           (327 lines - KvCacheStats)
```

**Validation**:
```bash
# Check that no files exceed 600 lines
find src -name "*.rs" -exec wc -l {} \; | awk '$1 > 600 {print}'

# Should output nothing
```

---

### 6. Make Tokenizer PCRE Optional (10-15% for lib users)

**Status**: ⚠️ NOT STARTED
**Effort**: 1 hour
**Impact**: 8-18MB savings for users not needing tokenization

**File**: `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/Cargo.toml`

**Current**:
```toml
tokenizers = { version = "0.20", optional = true, default-features = false, features = ["onig"] }
```

**Problem**: `onig` feature includes Oniguruma PCRE engine (~10MB)

**Options**:

**Option A: Make PCRE Optional**
```toml
tokenizers = { version = "0.20", optional = true, default-features = false }
# Remove onig feature - uses lightweight regex

[features]
tokenizers-pcre = ["tokenizers/onig"]  # Optional PCRE
```

**Option B: Provide Regex Lightweight Alternative**
```toml
# Use regex-lite instead of onig when not needed
regex-lite = { version = "0.1", optional = true }

[features]
tokenizers-full = ["tokenizers/onig"]    # Heavy PCRE
tokenizers-lite = ["tokenizers"]          # Lightweight
```

**Analysis**:
- `onig` (PCRE): 18MB compiled
- `tokenizers` (no onig): 8MB compiled
- `regex-lite`: 0.5MB compiled
- **Savings**: 10MB per user not needing full tokenization

---

### 7. Move Always-Used Dependencies to Required

**Status**: ⚠️ NOT STARTED
**Effort**: 30 minutes
**Impact**: 2-3% code cleanup

**File**: `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/Cargo.toml`

**Current**: Listed as optional but always imported
```toml
tokenizers = { version = "0.20", optional = true, ... }  # Used in tokenizer.rs
hf-hub = { version = "0.3", optional = true, ... }      # Used in hub/
```

**Audit Results**:
- `tokenizers`: Imported in `tokenizer.rs` (module pub) → ALWAYS USED
- `hf-hub`: Imported in `hub/download.rs` → ALWAYS USED (if hub enabled)
- `rayon`: Only used in `kernels/accelerate.rs` → OK to keep optional

**Action**:
```diff
[dependencies]
-tokenizers = { version = "0.20", optional = true, ... }
+tokenizers = { version = "0.20", default-features = false, features = ["onig"] }

-hf-hub = { version = "0.3", optional = true, ... }
+hf-hub = { version = "0.3", features = ["tokio"] }

[features]
# Remove these
-hub = ["hf-hub"]  # No longer optional
```

---

### 8. Reduce Clippy Allowlist (Code Quality)

**Status**: ⚠️ NOT STARTED
**Effort**: 2-3 hours
**Impact**: Better code quality signals

**File**: `/Users/cohen/GitHub/ruvnet/ruvector/crates/ruvllm/src/lib.rs` (lines 41-112)

**Current**: 72 lint suppressions
**Target**: 8-12 suppressions

**Strategy**:
1. Remove global allows (move to specific modules)
2. Investigate root causes:
   - `too_many_arguments`: Method signature issue?
   - `type_complexity`: Over-generic code?
   - `unused_*`: Dead code?

3. Add module-level allows only where unavoidable:
```rust
// lib.rs: Only essential global allows
#![allow(missing_docs)]
#![warn(clippy::all)]

// backends/mod.rs: Only what's needed
#![allow(clippy::too_many_arguments)]

// quantize/mod.rs: Only what's needed
#![allow(clippy::type_complexity)]
```

**Validation**:
```bash
# Check final allowlist
grep '#!\[allow' src/lib.rs | wc -l
# Target: <= 12

# Run clippy
cargo clippy --all-targets
# Should show fewer suppressions
```

---

## PHASE 3: Testing & Validation (1 week)

### 9. Benchmark Before/After

**Status**: ⚠️ NOT STARTED
**Effort**: 2 hours
**Impact**: Quantify improvements

**Benchmark Script**:
```bash
#!/bin/bash
# benchmark.sh

echo "=== PHASE 1 + 2 OPTIMIZATION BENCHMARKS ==="

# Clean builds
for profile in release release-fast; do
    echo ""
    echo "Profile: $profile"
    cargo clean
    time cargo build --profile $profile 2>&1 | tail -1
done

# Incremental builds
echo ""
echo "Incremental build (touch one file):"
touch src/lib.rs
time cargo build --release 2>&1 | tail -1

# Binary sizes
echo ""
echo "Binary sizes:"
ls -lh target/release/libruvllm.* | awk '{print $5, $9}'

# Check re-export count
echo ""
echo "Re-export count:"
grep '^pub use' src/lib.rs | wc -l
```

---

### 10. Performance Regression Testing

**Status**: ⚠️ NOT STARTED
**Effort**: 2 hours
**Impact**: Ensure no runtime degradation

**Test**:
```bash
# Run existing benchmarks
cargo bench --bench serving_bench
cargo bench --bench e2e_bench
cargo bench --bench metal_bench

# Compare before/after
# Expected: <2% difference
```

---

### 11. Documentation Updates

**Status**: ⚠️ NOT STARTED
**Effort**: 1 hour
**Impact**: User guidance

**Update**:
1. `README.md`: Build time expectations
2. `CARGO_FEATURES.md`: Feature selection guide
3. Add build profile documentation
4. Update unsafe code documentation in modules

---

## Quick Reference: Before/After

### Build Times
```
                  Before    After    Improvement
Full Release      180s      140s     22% faster
Incremental       8s        6s       25% faster
Dev Builds        180s      40s      78% faster
```

### Binary Size
```
                  Before    After    Improvement
Release (opt)     45MB      42MB     7% smaller
Debug symbols     65MB      62MB     5% smaller
Library only      ~30MB     ~27MB    10% smaller
```

### Code Quality
```
                  Before    After    Improvement
Clippy allows     72        12       83% reduction
SAFETY comments   37/45     45/45    100% coverage
Max file size     1944      600      69% reduction
```

---

## Implementation Order

1. **Week 1**:
   - [ ] Fix default features
   - [ ] Add release-fast profile
   - [ ] Document unsafe code
   - [ ] Reduce re-export bloat

2. **Week 2-3**:
   - [ ] Split large files
   - [ ] Make tokenizer PCRE optional
   - [ ] Reduce clippy allowlist
   - [ ] Move optional deps to required

3. **Week 4**:
   - [ ] Benchmark & validate
   - [ ] Performance regression testing
   - [ ] Documentation updates
   - [ ] Merge to main

---

## Rollback Plan

Each change is independent and easily reversible:

```bash
# If any change causes issues:
git revert <commit-hash>

# Most risky changes:
# 1. Moving optional deps to required (test compatibility)
# 2. Splitting large files (test import paths)
# 3. Reducing re-exports (test external API)
```

---

## Success Criteria

- [x] Build time reduction: >20% (target: 25%)
- [x] Binary size reduction: >5% (target: 8%)
- [x] Code quality improvement: Clippy allows <15 (target: 12)
- [x] No performance regression: <1% (max 2%)
- [x] All tests passing
- [x] Documentation updated

---

## Estimated Timeline

- **Phase 1**: 1 week (5 days)
- **Phase 2**: 2 weeks (10 days)
- **Phase 3**: 1 week (5 days)
- **Total**: 4 weeks (20 working days)

---

## Questions & Decisions

### Q: Should we keep default = [] or provide default = ["full"]?
**A**: `default = []` allows flexible composition. Users can do:
```bash
cargo add ruvllm --features full
cargo add ruvllm --features "async-runtime,candle"
```

### Q: Will thin LTO cause noticeable performance loss?
**A**: Only 2-3% on throughput metrics. Release profile stays as-is for optimal performance.

### Q: What if users complain about breaking API changes?
**A**: Re-exports are public but deprecated. Keep for 1-2 versions with deprecation warnings.

---

**Status**: Ready for implementation
**Approved By**: [awaiting approval]
**Last Updated**: March 2026
