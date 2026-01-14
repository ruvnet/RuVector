# Root Cause Analysis: HNSW Index Scan Crash on PostgreSQL 18

**Date:** 2026-01-14
**Status:** RESOLVED
**Project:** ruvector-postgres (PostgreSQL 18 extension with pgrx 0.16)

---

## Executive Summary

The HNSW index access method was crashing on PostgreSQL 18 during `ORDER BY embedding <-> '...'::ruvector` queries. The root cause was **callback pointer misalignment** in the `IndexAmRoutine` structure due to pg18 adding a new callback field (`amgettreeheight`) in the middle of the callback list.

**Fix:** Moved `amgettreeheight` callback from the end of the struct template to its correct position (between `amcostestimate` and `amoptions`) for pg18 builds.

---

## Problem Description

### Symptoms
- Extension loaded successfully
- Index builds succeeded
- Queries with `ORDER BY embedding <-> '...'::ruvector` caused PostgreSQL to crash with:
  ```
  server closed the connection unexpectedly
  ```
- Warning logged: `HNSW: Could not extract query vector, using zeros`

### Affected Versions
- PostgreSQL 18.1
- pgrx 0.16
- ruvector-postgres 2.0.0

---

## Root Cause Analysis

### Discovery Process

1. **Initial Investigation** - Examined `hnsw_rescan` and `hnsw_gettuple` functions
   - Query vector extraction appeared correct
   - Added pg18-specific logging and extraction attempts

2. **Structure Comparison** - Compared pg17 vs pg18 `IndexScanDescData`
   - Found pg18 added `instrument` field (memory layout change)
   - Added defensive null checks for `orderByData`

3. **Deep Dive on IndexAmRoutine** - Examined callback structure definition
   - **CRITICAL FINDING:** pg18 inserts `amgettreeheight` between `amcostestimate` and `amoptions`
   - The Rust template had `amgettreeheight` at the END with `#[cfg(feature = "pg18")]`
   - This caused ALL subsequent callbacks to be offset by one pointer slot

### Root Cause

**Callback Pointer Misalignment**

In PostgreSQL's C struct, the field order is:

**pg17:**
```c
amcostestimate_function amcostestimate;
amoptions_function amoptions;
amproperty_function amproperty;
```

**pg18:**
```c
amcostestimate_function amcostestimate;
amgettreeheight_function amgettreeheight;  // NEW!
amoptions_function amoptions;
amproperty_function amproperty;
```

But the Rust template was:
```rust
amcostestimate: None,
amoptions: None,       // WRONG! This is where amgettreeheight should be
amproperty: None,
// ... at the end ...
#[cfg(feature = "pg18")]
amgettreeheight: None,  // WRONG POSITION!
```

**Result:** PostgreSQL was calling wrong function pointers, causing segmentation faults.

---

## Resolution

### Code Changes

**File:** `/Users/devops/Projects/active/ruvector/crates/ruvector-postgres/src/index/hnsw_am.rs`

**Change:** Moved `amgettreeheight` to correct position:

```rust
// Callbacks - set to None, will be filled in at runtime
ambuild: None,
ambuildempty: None,
aminsert: None,
ambulkdelete: None,
amvacuumcleanup: None,
amcanreturn: None,
amcostestimate: None,
// PG18: amgettreeheight MUST be between amcostestimate and amoptions
#[cfg(feature = "pg18")]
amgettreeheight: None,  // <-- MOVED HERE (correct position)
amoptions: None,
amproperty: None,
// ... rest of callbacks ...
// PG18 additions - new boolean flags only at end
#[cfg(feature = "pg18")]
amcanhash: false,
#[cfg(feature = "pg18")]
amconsistentequality: false,
#[cfg(feature = "pg18")]
amconsistentordering: false,
```

### Build Fix

Also discovered and worked around a pgrx 0.16 linking issue:
```bash
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo pgrx package
```

---

## Validation

### Test Results

**Before Fix:**
```sql
SELECT * FROM test_scan ORDER BY embedding <-> '[1,0,0]'::ruvector LIMIT 1;
-- server closed the connection unexpectedly
```

**After Fix:**
```sql
SELECT * FROM test_scan ORDER BY embedding <-> '[1,0,0]'::ruvector LIMIT 3;
-- Results:
--  1 | A    | [1,0,0]
--  4 | D    | [1,1,0]
--  2 | B    | [0,1,0]
```

All similarity search queries now return correct results.

### Test Coverage
- Single-point k-NN search
- Multiple result LIMIT queries
- Different query vectors
- Correct distance-based ranking

---

## Lessons Learned

1. **Struct Layout Matters:** Rust `#[cfg]` attributes don't reorder fields. When C structs change field order between versions, the Rust struct must match exactly.

2. **Version-Specific Callbacks:** PostgreSQL 18 added `amgettreeheight` callback in the MIDDLE of the callback list, not at the end. This breaks any code that assumes new fields are always appended.

3. **pgrx 0.16 Migration:** Upgrading from pgrx 0.12 to 0.16 requires careful verification of:
   - New callback positions
   - New structure fields (like `instrument` in IndexScanDescData)
   - Linker flags for dynamic symbol resolution

4. **Debugging Technique:** For PostgreSQL extension crashes:
   - Check callback pointer alignment first
   - Compare struct definitions between pg versions
   - Verify field order matches C headers exactly

---

## References

**Files Modified:**
- `/Users/devops/Projects/active/ruvector/crates/ruvector-postgres/src/index/hnsw_am.rs` (lines 1655-1696)

**pgrx Source References:**
- `~/.cargo/registry/src/index.crates.io-*/pgrx-pg-sys-0.16.1/src/include/pg17.rs` - pg17 IndexAmRoutine
- `~/.cargo/registry/src/index.crates.io-*/pgrx-pg-sys-0.16.1/src/include/pg18.rs` - pg18 IndexAmRoutine

**PostgreSQL 18 Changes:**
- Added `amgettreeheight` callback for B-tree height reporting
- Added `instrument` field to IndexScanDescData
- Added `amconsistentordering`, `amconsistentequality` boolean flags
