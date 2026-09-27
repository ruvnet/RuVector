# ruvector-gpu-runner

Runs one training job on a rented vast.ai GPU. It enforces a budget cap, destroys
the instance when the job ends, and uploads artifacts to GCS. Written in Rust,
with blocking HTTP via `ureq` and no async runtime. It does not use the Python
`vastai` CLI.

## Usage

```sh
cargo build -p ruvector-gpu-runner
export VAST_API_KEY="$(gcloud secrets versions access latest --secret=VAST_API_KEY --project=cognitum-20260110)"

# Plan only. This runs every check against the live offers API but skips the paid create call.
./target/debug/ruvector-gpu-runner launch --dry-run \
  --max-usd 5 --max-hours 3 \
  --git-ref feat/openjev --git-sha "$(git rev-parse HEAD)" \
  --cmd 'cargo run --release -p <crate> -- --out "$RVGR_ARTIFACT_DIR"'

# CPU mode: rank by $/effective-core-hour, any GPU (or none), plain ubuntu image.
./target/debug/ruvector-gpu-runner launch --dry-run --cpu-mode \
  --min-cpu-cores 32 --min-ram-gb 64 --max-usd 5 --max-hours 3 \
  --git-ref feat/kge-bench --git-sha "$(git rev-parse HEAD)" \
  --cmd 'cargo run --release -p <crate> -- --out "$RVGR_ARTIFACT_DIR"'

# Continue a previous run from its newest checkpoint (sha256-verified on the instance).
./target/debug/ruvector-gpu-runner launch ... --resume-from rvgr-20260927t050547-d3e29a23

# The same command without --dry-run launches the job and supervises it until DONE, FAILED or a cap.
# Recover after the launcher was killed with SIGKILL: list instances, then destroy them.
./target/debug/ruvector-gpu-runner reap          # list only
./target/debug/ruvector-gpu-runner reap --yes    # destroy + verify
```

The exit code is 0 when the job succeeds or the dry run would launch. It is 2 when
the launch is refused (the reasons are printed) and 1 when the job fails or a
watchdog destroys the instance.

## Safety controls (enforced in code)

| Control | Where |
|---|---|
| `--max-usd`, `--max-hours` are required. The worst case is `planning_rate x max_hours + transfer`. The planning rate is `max(dph_total, dph_base + disk x storage_cost/720)`. The launch is refused if the worst case exceeds `--max-usd` or the account credit. | `budget.rs`, `offer.rs` |
| The watchdog destroys the instance at 95% of `--max-usd` or 95% of `--max-hours`. It measures from the create call, because billing starts at creation. The job's own `timeout` inside the instance fires earlier by 10 minutes plus a transfer window sized at 160 s/GB for `--expected-upload-gb` (plus `--expected-checkpoint-gb` when a separate final checkpoint is uploaded), at least 2 minutes. Artifacts and the final checkpoint are cut when that window ends, so the log and the DONE/FAILED marker still land before the watchdog. A `--max-hours` too short for that reserve is refused. | `budget.rs`, `runner.rs`, `job.rs` |
| The instance is destroyed on DONE, FAILED, timeout, budget cap, startup timeout, instance exit, a price mismatch (actual price more than 5% above plan), Ctrl-C or SIGTERM, a panic or early return (`Drop` guard), or an orphan left after a create error. Destroy retries with backoff for up to 10 minutes, treats 404 as success, and confirms the id is gone from `/api/v1/instances/`. | `runner.rs`, `vast.rs` |
| SIGKILL recovery: the instance id is written to `~/.local/state/ruvector-gpu-runner/live/` as soon as create returns. `reap` destroys any instance labelled `rvgr-*`. By default, a launch is refused while such an instance exists. | `audit.rs`, `main.rs` |
| Offer filters: the host must be verified, reliability must be at least 0.98 (the CLI refuses a lower value), and GPU model allow-list, GPU count, minimum GPU RAM, minimum CUDA version, disk and maximum $/h are all checked. Every filter is re-checked on the client, because the server ignores some filter keys. Datacenter hosts (`hosting_type=1`) rank first, or are required with `--require-datacenter`. | `offer.rs` |
| CPU mode (`--cpu-mode`): `--min-cpu-cores` and `--min-ram-gb` are checked against `cpu_cores_effective` (threads allocated to the offer, not the host total) and `cpu_ram`. The GPU allow-list, count and RAM default to "any" (including GPU-less offers). Ranking is by planning rate / effective threads. The default image is `ubuntu:24.04` pinned by digest; `--min-cuda` defaults to 12.6 for every GPU-mode launch whatever the image; only a CPU-mode launch of a non-`nvidia/cuda` image defaults to 0 (unchecked). Verification, the reliability floor, `--max-dph`, the budget and credit gates, auto-destroy and the audit are unchanged. | `offer.rs`, `main.rs` |
| Job contract: the image must be pinned by `@sha256:` digest. The SHA must be a full 40-character hash that is reachable from `--git-ref` on the remote. The instance checks out that SHA and asserts `git rev-parse HEAD` matches it. The onstart script uploads artifacts, then the log, then the DONE or FAILED marker, in that order. | `job.rs` |
| Audit: every dry run, refusal, launch and destroy is written to a JSONL file at `~/.local/state/ruvector-gpu-runner/audit.jsonl`. Records hold the offer, rate, estimate, instance id, start and stop times, and estimated cost. They never contain secrets or signed URLs. | `audit.rs` |

## Artifact upload design

The instance receives no credentials of any kind: no vast.ai API key, no GCP key
and no token. Before the create call, the launcher mints four **V4 signed PUT
URLs** locally:

```
gcloud storage sign-url gs://ruvector-openjev-artifacts/runs/<run_id>/<obj> \
  --http-verb=PUT --headers=content-type=application/octet-stream \
  --impersonate-service-account=rvgr-uploader@ruv-dev.iam.gserviceaccount.com
```

The four objects are `artifacts.tar.gz`, `job.log`, `DONE` and `FAILED`. The URLs
reach the instance as environment variables. Each URL is valid for one object, is
PUT-only and expires at `max_hours + 30m`.

The signer service account holds only `roles/storage.objectCreator` on this one
bucket and has **no keys**. Impersonation signs through IAM signBlob, so no
credential file exists anywhere. gcloud caps impersonated URLs at 12 hours, so
`--max-hours` is capped at 11.5.

The launcher polls for the marker with the operator's own gcloud credentials.

Bucket: `gs://ruvector-openjev-artifacts` in project `ruv-dev`, location
us-central1. Uniform bucket-level access and public-access prevention are enforced.

One-time setup of the service account (the dry run prints these commands and refuses to launch until it is done):

```sh
gcloud iam service-accounts create rvgr-uploader --project=ruv-dev
gcloud storage buckets add-iam-policy-binding gs://ruvector-openjev-artifacts \
  --member=serviceAccount:rvgr-uploader@ruv-dev.iam.gserviceaccount.com --role=roles/storage.objectCreator
gcloud iam service-accounts add-iam-policy-binding rvgr-uploader@ruv-dev.iam.gserviceaccount.com \
  --project=ruv-dev --member=user:$(gcloud config get-value account) --role=roles/iam.serviceAccountTokenCreator
```

## Checkpoints and resume

Every `--checkpoint-secs` (default 600, `0` disables; minimum 60) the onstart
script tars `--checkpoint-dir` (default `--artifact-dir`) and PUTs it to a
rolling slot `ckpt-<n mod ring>.tar.gz` (`--checkpoint-ring`, default 3), then
PUTs `ckpt-latest.json` = `{run_id, seq, slot, object, sha256, at}`. Uploads start
only after clone and resume finish, and the pointer is written only after its slot
upload succeeds. When the job ends (or its in-instance timeout fires) the loop is stopped, and a
periodic upload still in flight is cut rather than awaited (its slot is not the one
the pointer names). Artifacts are uploaded, then one final checkpoint, then the log
and marker. If `--checkpoint-dir` is the artifact dir, the final checkpoint is
skipped, because `artifacts.tar.gz` already holds the same state. Resume from it with
`--resume-from gs://<bucket>/runs/<run_id>/artifacts.tar.gz`. A watchdog destroy
therefore loses at most one interval. Jobs should write checkpoint files
atomically (tmp file + rename), because the directory is tarred while the job runs.
Every checkpoint URL is one more signed PUT URL of the same kind as above. The
estimated transfer (`--expected-checkpoint-gb` x uploads) is part of the budget gate.

`--resume-from <run id | gs://<bucket>/...tar.gz>` resolves the object on the
launcher (a run id is resolved through its `ckpt-latest.json`, using the operator's
own gcloud credentials). The launcher then signs one **GET** URL for that object,
valid for `--startup-timeout-mins` + 30 minutes. The uploader SA cannot read, so a
second keyless SA signs this URL: `rvgr-reader`, which holds only
`roles/storage.objectViewer` (`--reader-sa`; the dry run prints its setup commands).
The instance downloads the object, checks its sha256 when known, and extracts it
into the checkpoint directory before the command starts. It then sets
`RVGR_RESUMED=1`. A failed download or checksum marks the job FAILED instead of
starting from scratch. The instance still holds no credentials: the script
unsets every `RVGR_URL_*` variable before running the job command, and it starts
with `set +x`.

## Known limits

- If the launcher is SIGKILLed, the instance keeps billing after its job finishes
  until `reap --yes` runs. The instance cannot destroy itself, because it holds no
  API key. Run the launcher under `tmux` or `systemd-run`.
- The host operator can see the environment variables, including the signed URLs.
  Each URL only grants writes to one object in this run's prefix, until it expires.
