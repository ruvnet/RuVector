#!/usr/bin/env bash
# KGE bench job for a ruvector-gpu-runner CPU-mode instance (ADR-007 plan M6b).
#
# Runs INSIDE the instance (repo checked out at the pinned SHA, cwd = repo
# root; the runner exports RVGR_ARTIFACT_DIR, RVGR_CHECKPOINT_DIR and, after a
# --resume-from restore, RVGR_RESUMED=1). Also runs locally for a $0 pass:
#   RVGR_ARTIFACT_DIR=/tmp/art RVGR_CHECKPOINT_DIR=/tmp/ck KGE_WORK=/tmp/kw \
#     KGE_BENCH_BIN=target/release/ruvector-kge-bench \
#     scripts/kge/vast-kge.sh --dataset wn18rr --config-id C1 --max-epochs 2 --stop-after-epoch 1
#   RVGR_RESUMED=1 ... same command without --stop-after-epoch   # resumes
#
# Launch (CPU mode; checkpoint dir separate from artifact dir so the runner
# writes ckpt-latest.json and `--resume-from <run id>` works):
#   ruvector-gpu-runner launch --cpu-mode --min-cpu-cores 32 --min-ram-gb 32 \
#     --max-usd 1.5 --max-hours 1.0 --git-ref feat/kge-bench --git-sha "$(git rev-parse HEAD)" \
#     --artifact-dir /workspace/out --checkpoint-dir /workspace/ckpt --checkpoint-secs 120 \
#     --expected-checkpoint-gb 2.0 --expected-upload-gb 0.05 \
#     --cmd 'bash scripts/kge/vast-kge.sh --dataset wn18rr --config-id C1 --max-epochs 2 --stop-after-epoch 1'
#   WN18RR C1 (k=1000): one checkpoint snapshot is ~0.94 GB (checkpoint.bin +
#   best/), ~0.91 GB gzipped; one epoch took 552 s on 32 EPYC 7B13 threads
#   (2026-09-27 smoke). $RVGR_CHECKPOINT_DIR holds exactly ONE snapshot, so
#   every checkpoint tarball is ~0.91 GB. --expected-checkpoint-gb sizes the
#   post-job transfer window at 160 s/GB (gzip time included): 2.0 gives
#   ~320 s for ~0.91 GB, about 2x margin. (Before this layout the dir held two
#   snapshots, ~1.83 GB per tarball, and the resumed smoke's final checkpoint
#   was cut.) Pass --threads to match the offer's allocation if the
#   container exposes no CPU quota (nproc shows the whole host).
#
# Stages: toolchain (rustup, pinned) -> build ruvector-kge-bench (--features
# parallel, -C target-cpu=native) -> datasets (the bench's pinned, sha256-
# verified loaders) -> train + score VALID only -> artifacts.
#
# Checkpoints: the bench writes OUT/checkpoint.bin (and best/) atomically after
# every epoch. Every --checkpoint-secs this script snapshots them into
# $RVGR_CHECKPOINT_DIR/snap-<n>/ (copied to a temp dir on the same filesystem,
# hashed into SHA256SUMS, then renamed into place) and moves the LATEST pointer
# (tmp + rename). The runner tars that directory while the job runs, so it only
# ever sees complete snapshots. Only the newest snapshot stays in the tarred
# dir; the previous one moves to $KGE_WORK/prev-snap/ (a local fallback, never
# uploaded), and older history is the runner's GCS ring (--checkpoint-ring).
# A tar that races the move exits 1 and the runner counts that tick as failed.
# Resume restores the newest snapshot whose SHA256SUMS verifies ($CK, then
# prev-snap), reuses its exact run config (threads included: the bench refuses
# a thread-count change) and passes --resume.
#
# TEST is never scored: this script has no --final path. No credentials are
# needed or used. Nothing is published.
set -euo pipefail

DATASET=""; CONFIG_ID=""; SEED=100; MAX_EPOCHS=""; STOP_AFTER=""; THREADS=""
EVAL_EVERY=1; PATIENCE=""; CKPT_SECS=120
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --config-id) CONFIG_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --stop-after-epoch) STOP_AFTER="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --eval-every) EVAL_EVERY="$2"; shift 2 ;;
    --early-stop-patience) PATIENCE="$2"; shift 2 ;;
    --checkpoint-secs) CKPT_SECS="$2"; shift 2 ;;
    -h|--help) sed -n 2,42p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
# Input validation: every value is interpolated into JSON or a command line.
[[ "$DATASET" =~ ^(wn18rr|fb15k237|codexm)$ ]] || { echo "--dataset must be wn18rr|fb15k237|codexm" >&2; exit 2; }
[[ "$CONFIG_ID" =~ ^C[1-8]$ ]] || { echo "--config-id must be C1..C8" >&2; exit 2; }
for v in SEED MAX_EPOCHS EVAL_EVERY CKPT_SECS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "--${v,,} must be a non-negative integer (got '${!v}')" >&2; exit 2; }
done
for v in STOP_AFTER THREADS PATIENCE; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "${v,,} must be a positive integer" >&2; exit 2; }
done
[ "$MAX_EPOCHS" -ge 1 ] && [ "$EVAL_EVERY" -ge 1 ] && [ "$CKPT_SECS" -ge 10 ] || { echo "bad --max-epochs/--eval-every/--checkpoint-secs" >&2; exit 2; }

ART="${RVGR_ARTIFACT_DIR:?RVGR_ARTIFACT_DIR must be set}"
CK="${RVGR_CHECKPOINT_DIR:-$ART/ckpt}"
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
WORK="${KGE_WORK:-/workspace/kge}"
RUN="$WORK/run"            # bench --out (not tarred by the runner)
TMP="$WORK/.snap-tmp"      # same filesystem as $CK only when both are under /workspace
PREV="$WORK/prev-snap"      # previous snapshot, outside the tarred dir
export RVKGE_CACHE_DIR="${RVKGE_CACHE_DIR:-$WORK/cache}"
mkdir -p "$ART/logs" "$CK" "$RUN" "$RVKGE_CACHE_DIR"
log() { echo "[kge] $(date -u +%FT%TZ) $*"; }
t0=$(date +%s)
stamp() { echo "{\"stage\":\"$1\",\"seconds\":$(( $(date +%s) - t0 ))}" >> "$ART/logs/stages.jsonl"; }

# ---- 1. toolchain + build ----------------------------------------------------
if [ -z "${KGE_BENCH_BIN:-}" ]; then
  if ! command -v cc >/dev/null || ! command -v pkg-config >/dev/null; then
    log "installing build-essential pkg-config"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      build-essential pkg-config libssl-dev ca-certificates curl git >/dev/null
  fi
  if ! command -v cargo >/dev/null; then
    log "installing rust $RUST_TOOLCHAIN"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
  fi
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  stamp toolchain
  log "cargo build ruvector-kge-bench (parallel, target-cpu=native) on $(nproc) threads"
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/target}"
  build() { cargo build --release --locked -p ruvector-kge-bench --features parallel 2>&1 | tee -a "$ART/logs/build.log" | tail -3; }
  build || { log "build failed; retrying once"; build; }
  BIN="$CARGO_TARGET_DIR/release/ruvector-kge-bench"
  stamp build
else
  BIN="$KGE_BENCH_BIN"
fi
[ -x "$BIN" ] || { echo "ruvector-kge-bench binary missing: $BIN" >&2; exit 1; }

# ---- 2. restore (resume) ----------------------------------------------------
# Snapshot dir names in $CK (snap-<n>), unsorted.
snaps() { local d; for d in "$CK"/snap-*; do [ -d "$d" ] && [[ "${d##*/}" =~ ^snap-[0-9]+$ ]] && echo "${d##*/}"; done; }
# A snapshot verifies when its SHA256SUMS checks out. Prints its path: the
# LATEST one, other snapshots in $CK newest first, then the local prev-snap.
verified_snap() {
  local s d
  for d in $( { s=$(cat "$CK/LATEST" 2>/dev/null) && echo "$CK/$s"; snaps | sort -t- -k2 -nr | sed "s|^|$CK/|"; ls -d "$PREV"/snap-* 2>/dev/null; } ); do
    [[ "${d##*/}" =~ ^snap-[0-9]+$ ]] || continue
    [ -f "$d/SHA256SUMS" ] && (cd "$d" && sha256sum -c --quiet SHA256SUMS) >/dev/null 2>&1 && { echo "$d"; return 0; }
  done
  return 1
}
SEQ=0; RESUME=0
if [ "${RVGR_RESUMED:-0}" = 1 ] || [ -f "$CK/LATEST" ]; then
  SNAPDIR=$(verified_snap) || { echo "resume requested but no verified snapshot in $CK or $PREV" >&2; exit 1; }
  SNAP=${SNAPDIR##*/}; SEQ=${SNAP#snap-}
  log "resume: restoring $SNAP (sha256 verified)"
  rm -rf "$RUN"; mkdir -p "$RUN"
  cp "$SNAPDIR/checkpoint.bin" "$RUN/"
  [ -d "$SNAPDIR/best" ] && cp -r "$SNAPDIR/best" "$RUN/"
  THREADS_SNAP=$(sed -n 's/.*"threads":\([0-9]*\).*/\1/p' "$SNAPDIR/run.json")
  [ -z "$THREADS" ] || [ "$THREADS" = "$THREADS_SNAP" ] || { echo "--threads $THREADS != checkpoint's $THREADS_SNAP (bench refuses)" >&2; exit 2; }
  THREADS=$THREADS_SNAP; RESUME=1
elif [ -f "$RUN/checkpoint.bin" ]; then
  RESUME=1   # same instance, restarted job
fi
# Default threads: the container's CPU quota, not nproc. A vast.ai offer gets
# `cpu_cores_effective` threads of a larger host, but nproc shows the host
# (the 2026-09-27 smoke ran 256 rayon threads on a 32-thread allocation).
cpu_quota() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  else return 1; fi
  [[ "$q" =~ ^[0-9]+$ ]] && [[ "$p" =~ ^[1-9][0-9]*$ ]] || return 1
  echo $(( (q + p - 1) / p ))
}
if [ -z "$THREADS" ]; then
  THREADS=$(nproc)
  if Q=$(cpu_quota) && [ "$Q" -ge 1 ] && [ "$Q" -lt "$THREADS" ]; then THREADS=$Q; fi
fi

# Deterministic config (the bench's resume compares the whole RunConfig).
CFG="{\"dataset\":\"$DATASET\",\"config_id\":\"$CONFIG_ID\",\"seed\":$SEED,\"max_epochs\":$MAX_EPOCHS,\"eval_every\":$EVAL_EVERY,\"threads\":$THREADS"
[ -z "$PATIENCE" ] || CFG="$CFG,\"early_stop_patience\":$PATIENCE"
CFG="$CFG}"
if [ -n "${SNAP:-}" ] && [ "$CFG" != "$(cat "$SNAPDIR/run.json")" ]; then
  echo "resume: checkpoint config $(cat "$SNAPDIR/run.json") != requested $CFG" >&2; exit 2
fi
printf '%s\n' "$CFG" > "$WORK/run.json"; printf '%s' "$CFG" > "$WORK/run.json.raw"
stamp restore

{
  echo "{"
  echo "  \"git_sha\": \"$(git rev-parse HEAD)\","
  echo "  \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//;s/"//g')\","
  echo "  \"nproc\": $(nproc), \"cgroup_cpu_quota\": \"$(cpu_quota || echo none)\", \"threads\": $THREADS,"
  echo "  \"mem_kb\": $(grep -m1 MemTotal /proc/meminfo | awk '{print $2}'),"
  echo "  \"avx512f\": $(grep -qm1 avx512f /proc/cpuinfo && echo true || echo false),"
  echo "  \"rustc\": \"$( (rustc --version 2>/dev/null || echo n/a) | tr -d '"')\","
  echo "  \"resumed\": $RESUME, \"resumed_from_snapshot\": \"${SNAP:-}\","
  echo "  \"config\": $CFG"
  echo "}"
} > "$ART/env.json"

# ---- 3. snapshots -----------------------------------------------------------
LAST_ID=""
snapshot() {
  [ -f "$RUN/checkpoint.bin" ] || return 0
  local id0 id1 n dst want got
  id0=$(stat -c '%i:%s:%Y' "$RUN/checkpoint.bin")
  [ "$id0" != "$LAST_ID" ] || return 0
  n=$(( SEQ + 1 )); dst="$TMP/snap-$n"
  rm -rf "$TMP"; mkdir -p "$dst"
  cp "$RUN/checkpoint.bin" "$dst/"      # an open fd keeps the old inode on rename
  [ -d "$RUN/best" ] && cp -r "$RUN/best" "$dst/"
  id1=$(stat -c '%i:%s:%Y' "$RUN/checkpoint.bin")
  if [ "$id0" != "$id1" ]; then log "snapshot: checkpoint moved during copy; next tick"; rm -rf "$TMP"; return 0; fi
  if [ -f "$dst/best/manifest.json" ]; then
    want=$(sed -n 's/.*"weights_sha256": *"\([0-9a-f]*\)".*/\1/p' "$dst/best/manifest.json")
    got=$(sha256sum "$dst/best/weights.bin" | cut -d' ' -f1)
    if [ "$want" != "$got" ]; then log "snapshot: best/ mid-export; next tick"; rm -rf "$TMP"; return 0; fi
  fi
  cp "$WORK/run.json.raw" "$dst/run.json"
  (cd "$dst" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  rm -rf "$CK/snap-$n"                 # a stale same-numbered dir (resumed from prev-snap)
  mv "$dst" "$CK/snap-$n"               # atomic rename into the tarred dir
  printf 'snap-%s\n' "$n" > "$CK/LATEST.tmp" && mv -f "$CK/LATEST.tmp" "$CK/LATEST"
  SEQ=$n; LAST_ID=$id0
  # Only the newest snapshot stays in the tarred dir (one snapshot per
  # tarball); the previous one moves to $PREV as a local fallback.
  local old
  for old in $(snaps | sort -t- -k2 -n); do   # ascending: the newest old one survives
    [ "$old" = "snap-$n" ] && continue
    rm -rf "${PREV:?}"; mkdir -p "$PREV"; mv "${CK:?}/$old" "$PREV/"
  done
  log "snapshot snap-$n ($(du -sh "$CK/snap-$n" | cut -f1))"
}
if [ "$RESUME" = 1 ] && [ -n "${SNAP:-}" ]; then LAST_ID=$(stat -c '%i:%s:%Y' "$RUN/checkpoint.bin"); fi

# ---- 4. train + score VALID only ---------------------------------------------
ARGS=(--config "$WORK/run.json" --out "$RUN" --repo "$REPO")
[ "$RESUME" = 1 ] && ARGS+=(--resume)
[ -z "$STOP_AFTER" ] || ARGS+=(--stop-after-epoch "$STOP_AFTER")
log "bench: ${ARGS[*]}"
"$BIN" "${ARGS[@]}" > >(tee -a "$ART/logs/bench.log") 2>&1 &
BPID=$!
finish() {
  local rc=$1
  snapshot || true
  [ -f "$RUN/receipt.json" ] && cp "$RUN/receipt.json" "$ART/"
  for d in best last; do [ -f "$RUN/$d/manifest.json" ] && { mkdir -p "$ART/$d"; cp "$RUN/$d/manifest.json" "$ART/$d/"; }; done
  [ -f "$CK/LATEST" ] && cp "$CK/LATEST" "$ART/checkpoint-LATEST"
  stamp "train rc=$rc"
  (cd "$ART" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  log "done rc=$rc"
  exit "$rc"
}
# `timeout` signals the whole process group, so the bench may already be gone:
# neither the kill nor the wait may trip errexit before the final snapshot.
trap 'set +e; log "TERM: stopping bench, final snapshot"; kill -TERM "$BPID" 2>/dev/null; wait "$BPID" 2>/dev/null; finish 124' TERM INT
stamp train-start
i=0
while kill -0 "$BPID" 2>/dev/null; do
  sleep 1; i=$(( i + 1 ))
  if [ "$i" -ge "$CKPT_SECS" ]; then i=0; snapshot || log "snapshot failed (continuing)"; fi
done
set +e; wait "$BPID"; RC=$?; set -e
finish "$RC"
