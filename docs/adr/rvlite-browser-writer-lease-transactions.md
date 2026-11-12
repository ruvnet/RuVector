# Commit browser lease ownership changes atomically

## Context
BrowserWriterLease promises one writer per store. Its read/write/confirm sequence uses separate IndexedDB transactions, allowing two contenders to each confirm ownership. A timed-out holder also renews or deletes another holder's replacement lease. This can permit competing RVF writers despite the coordination contract.

## Decision
Compare the current record and conditionally mutate it in one readwrite transaction. Resolve success only after commit. Acquisition respects the existing stale threshold; heartbeat and release require the stored holder to match the caller. Close database handles on success and failure and stop renewal after ownership is lost. Serialize acquire/release operations within each instance, and require release before changing stores, so an overlapping release cannot delete a newly acquired lease or strand the previous one.

## Evidence
The unchanged public class fails three producer schedules: simultaneous acquisition, stale-holder release and stale heartbeat; ordinary lifecycle control passes. Spec-compatible IndexedDB unit tests cover these and a deliberately aborted transaction. Independent real isolated Chromium replay reproduces all three baseline failures and verifies the corrected schedules. The exported class is transpiled directly from source to isolate this lease contract from generated WASM. Its strict standalone TypeScript check passes.

## Limits
These are class-level transaction and browser protocol tests, not a full WASM package build or RVF persistence stress test. Heartbeats remain best effort, stale leases use wall-clock time, and callers must respect lease results. No application-wide single-writer or physical durability guarantee is inferred.
