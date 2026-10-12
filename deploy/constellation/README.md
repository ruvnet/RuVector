# Weekly npm constellation observer

Project: `ruv-dev`; region: `us-central1`; schedule: Monday 09:00 `America/Toronto`.

The runtime uses `npm-constellation@ruv-dev.iam.gserviceaccount.com`, access to the single `NPM_TOKEN` secret and object access in `gs://ruv-dev-npm-constellation`. The scheduler identity has invoker access only to this job. The image contains the dependency-free observer sources, no npm configuration or provider credentials.

Read ADR-353 and ADR-354 before changing the control loop. Initial runtime is observation, persistent deltas and deterministic prioritization; adaptive evaluation and automatic release are not enabled.

## Operations

```sh
gcloud run jobs execute npm-constellation-weekly --project=ruv-dev --region=us-central1 --wait
gcloud scheduler jobs describe npm-constellation-weekly --project=ruv-dev --location=us-central1
gcloud storage cp gs://ruv-dev-npm-constellation/latest.json ./latest.json
```

A failed run retains diagnostics and never replaces the previous complete baseline. Inspect failed executions and Cloud Logging. A crashed process can leave `controller.lock`; before removing it, verify no execution is running. The generation-checked lock prevents competing executions from deleting another controller's lock.

Pause without deleting evidence:

```sh
gcloud scheduler jobs pause npm-constellation-weekly --project=ruv-dev --location=us-central1
```

Rotate the pinned npm secret version by updating the job's secret reference. Never print or export its value into a file. Token scope and expiry remain enforced by npm. Scanner credentials are never forwarded to arbitrary registry metadata URLs.

Build from `npm/packages/constellation` after a clean source commit, deploy the resulting image digest and set `SOURCE_SHA` to that commit. Keep infrastructure receipt in `docs/ecosystem/deployment.json`. A successful observer run does not certify releases.
