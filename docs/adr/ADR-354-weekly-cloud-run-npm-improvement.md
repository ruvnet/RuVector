# ADR-354: Weekly Cloud Run npm improvement loop

- Status: Proposed architecture; observer deployed and verified, adaptive promotion not activated
- Date: 2026-10-11
- Updated: 2026-10-11
- Related: ADR-353; existing research-gate and Ruflo ADR-322A promotion boundary

## Decision and scope

Use a Cloud Run **Job**, invoked by Cloud Scheduler each Monday at 09:00 in `America/Toronto` (`0 9 * * 1`). Region `us-central1`, project `ruv-dev`. The user requested Cloud Run and the npm token. Use the existing Secret Manager `NPM_TOKEN` rather than embedding credentials in source, container layers, logs, task arguments or reports.

The job is an evidence collector and prioritizer. The initial implementation is not a trained optimizer and does not execute arbitrary source scripts or publish packages. This distinction is deliberate: a cron schedule alone is not a self-improvement system. The implementation maintains historical observations and deltas; candidate generation, measured evaluation and governed publication are separate stages below.

## Weekly stages and durable artifacts

| Stage | Work | Artifact / gate |
|---|---|---|
| Discover | Enumerate all public maintainer packages and read latest manifests; deduplicate | inventory.json with observed timestamp, coverage, errors and source digest |
| Inspect | Typed dependency graph, runtime cycles, metadata/lifecycle/semver findings, npm direct-version advisories | graph and explicit audit scope |
| Recall | Read previous complete report from GCS | version changes, additions/removals and resolved findings |
| Prioritize | Security severity plus downstream impact plus evidence gaps | deterministic improvement-plan.json; never claims model training |
| Propose | Select small independent change batches and write/update ADRs | source-bound candidate branch and package allowlist; future execution lane |
| Evaluate | Run real consumer, absent-optional, MCP handshake, clean skills install and native matrix tests | nonzero test results, baseline and candidate metrics, held-out workloads |
| Learn | Record actual outcome labels and failed experiments; optionally use MetaHarness router/flywheel | durable labels, replay bundle and pinned signer trust; future lane |
| Promote | Enforce package-specific protected release checks | sealed source SHA, tarball hash, authorization, immutable release evidence |
| Verify | Download public bytes, compare seal, clean consumer smoke | publication receipt; failed gates retain prior known-good release |

One scheduled observer refreshes evidence weekly; a follow-up executor can consume its queue. The release executor must not infer authorization from registry metadata, model predictions or a signed receipt whose key has not been independently trusted.

## Cloud isolation

- Scheduler service account has `roles/run.invoker` only on the specific job.
- Runtime service account has access only to the npm secret and the dedicated report bucket.
- No public HTTP endpoint; no compute/service-account key in the container.
- Secret version pinned when deployed; rotation requires redeploying with a new version.
- One task, bounded CPU/memory, 30-minute timeout, zero automatic retries.
- GCS create-if-absent controller lock prevents overlap. A crash leaves a visible lock for investigation; remove it only after confirming no execution is active.
- Immutable run paths plus `latest.json`, advanced only after a complete scan. Failed scans remain diagnostic artifacts and fail the execution.
- Container is deployed by digest. Logs report execution identity and artifact paths, never tokens.

The observer checks the npm token's username and uses public registry reads; it does not gain release authority by holding that token. A later protected publisher should receive publish credentials separately from test execution. GitHub's established protected publisher can use its existing npm secret; do not enable a second independent publisher for the same package/version.

## Learning and optimization contract

Baseline metrics: pack size, cold install success/time, SDK correctness, MCP initialize/list/call success, skill discoverability, absent optional dependency behavior, dependent contract pass rate, advisory count and performance latency/quality. Store raw observations with package/version/platform/source SHA, command, test count, elapsed time and artifact digest. Unknown and skipped tests are not successful labels.

Use training/validation/holdout separation, frozen anchors and task-family stratification. Repeated attempts against a holdout require refreshed holdouts. Reject regressions in correctness, security and capability coverage even if a cost metric improves. Cap evaluator time, tokens and candidate count outside the model. Failed or budget-exhausted evaluations cannot promote.

MetaHarness `Router.fromExamples` may rank execution choices only from real labels and complete price information. `runFlywheelGenerations` needs a real proposer, evaluator, holdout and signer; its defaults do not supply persistence or trusted promotion. Ruflo records coordinated tasks and memory; an executor must actually run changes. No new provider spend is silently introduced by the observer.

## Release security and rollback

Each package batch needs its own verified owner/source mapping, lockfile audit, license and secret scan, packed-artifact tests, immutable source, dependency order and zero open release blockers. Use the protected workflow and npm token requested by the user. Token permissions/expiry and npm 2FA remain registry-enforced; never weaken them to make a job succeed.

Publish exactly the tested tarball, not a rebuilt checkout. Npm versions are immutable. On failure, stop the train, preserve receipts and prior channel state; recovery uses a new fixed version or an explicitly approved dist-tag change. Never unpublish historical packages as automatic rollback.

## Acceptance

Demonstrate one real Cloud Run execution, persisted complete report, Scheduler timezone/next run, least-privilege IAM bindings, secret reference and deployed digest. Unit tests cover incomplete scans, audit failure, graph cycles, prior-run comparison and routing rejection. Report the live observer separately from unimplemented candidate/evaluation/publication automation.

## References

- [Schedule Cloud Run jobs](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule)
- [Cloud Run job secrets](https://docs.cloud.google.com/run/docs/configuring/jobs/secrets)
- [npm immutable versions](https://docs.npmjs.com/cli/v10/commands/npm-publish/)

## Deployment evidence — 2026-10-11

Two Cloud Run executions succeeded with 398/398 packages; the second read the prior baseline and stored JSON, Markdown, the interactive map and improvement queue. Scheduler is enabled, next run October 12 at 09:00 Toronto time. See [deployment receipt](../ecosystem/deployment.json). The optimizer and publisher remain disabled in the observer.
