# Reject incomplete Salesforce query pagination

## Context
fetch_records promises to follow query pagination until done. An incomplete page without its nextRecordsUrl currently returns partial records as a successful result; an empty or repeated cursor can loop. This can produce a silently incomplete grounding collection or prevent a sync from completing.

## Decision
Require a nonempty string cursor on each page explicitly marked incomplete. Track fetched paths and reject repeats before another request. Raise RuntimeError instead of returning incomplete records. Preserve single-page completion and valid multistep cursor traversal.

## Evidence
Actual httpx client calls use its documented MockTransport seam and fixture query responses; no remote Salesforce org is involved. Three regressions fail against upstream, with 27 controls passing and one optional test skipped. The bounded regression fixture stops malformed loops after a small number of requests; the first unbounded empty-cursor baseline was interrupted and retained as invalidated evidence. Fixed full Python tests using the current-source native07933730 artifact pass 298 tests with 2 optional-integration skips.

## Limits
This validates response completeness and finite traversal for repeated paths. It does not impose a maximum number of distinct valid pages, recover a remote expired cursor, or prove production Salesforce access.
