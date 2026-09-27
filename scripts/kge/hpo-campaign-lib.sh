# shellcheck shell=bash
# Helpers for scripts/kge/hpo-campaign.sh (sourced, not executed).
#
# Everything reads the committed campaign manifest ($MANIFEST); nothing here
# recomputes caps. State that only exists after a launch (segment records,
# runner logs) lives outside git in $STATE. The vast.ai key is fetched into
# this process's environment only (never printed, never put on an argv or in
# a systemd unit file).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${KGE_HPO_MANIFEST:-$REPO/npm/packages/kge/bench/results/hpo/campaign-2026-09-27.json}"
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }
CAMPAIGN_ID="$(jq -r .campaign_id "$MANIFEST")"
STATE="${KGE_HPO_STATE:-$HOME/.local/state/kge-hpo/$CAMPAIGN_ID}"
AUDIT="${RVGR_AUDIT_LOG:-$HOME/.local/state/ruvector-gpu-runner/audit.jsonl}"
RUNNER="${RVGR_BIN:-$REPO/target/release/ruvector-gpu-runner}"
export BENCH="${KGE_BENCH_BIN:-$REPO/target/release/ruvector-kge-bench}"
BUCKET="$(jq -r .runner.bucket "$MANIFEST")"
GIT_REF="$(jq -r .git.ref "$MANIFEST")"
mkdir -p "$STATE/jobs"

log() { echo "[hpo] $*" >&2; }
die() { echo "[hpo] error: $*" >&2; exit 1; }
mf() { jq -r "$@" "$MANIFEST"; }
# Money arithmetic without Python: awk with fixed precision.
calc() { awk "BEGIN{printf \"%.4f\", $*}"; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }

valid_job() { [[ "$1" =~ ^(fb15k237|wn18rr|codexm)-C[1-8]$ ]] && [ "$(mf --arg j "$1" '[.jobs[]|select(.id==$j)]|length')" = 1 ]; }
job_field() { jq -r --arg j "$1" ".jobs[]|select(.id==\$j)|$2" "$MANIFEST"; }
unit_of() { echo "kge-hpo-$1"; }
jobs_all() { mf '.jobs[].id'; }
jobs_wave() { mf --argjson w "$1" '.jobs[]|select(.wave==$w)|.id'; }
jobs_dataset() { mf --arg d "$1" '.jobs[]|select(.dataset==$d)|.id'; }

need_key() {
  [ -n "${VAST_API_KEY:-}" ] && return 0
  VAST_API_KEY="$(gcloud secrets versions access latest --secret=VAST_API_KEY --project=cognitum-20260110 2>/dev/null)" \
    || die "cannot read VAST_API_KEY from Secret Manager"
  [ -n "$VAST_API_KEY" ] || die "VAST_API_KEY is empty"
  export VAST_API_KEY
}

need_runner() {
  [ -x "$RUNNER" ] && return 0
  log "building ruvector-gpu-runner (release)"
  (cd "$REPO" && cargo build --release -q -p ruvector-gpu-runner) || die "runner build failed"
}

# Live vast.ai credit in USD. The key reaches curl through process
# substitution (printf is a builtin), never argv.
credit_now() {
  need_key
  curl -fsS --max-time 30 https://console.vast.ai/api/v0/users/current/ \
    -H @<(printf 'Authorization: Bearer %s\n' "$VAST_API_KEY") | jq -r '.credit // empty'
}

# ---- segment state ----------------------------------------------------------
# $STATE/jobs/<job>/segments.jsonl: one line per launched segment
#   {seg, unit, sha, threads, max_hours, max_usd, launched_at, log, resume_from}
seg_file() { echo "$STATE/jobs/$1/segments.jsonl"; }
seg_count() { local f; f=$(seg_file "$1"); [ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0; }
seg_get() { sed -n "${2}p" "$(seg_file "$1")"; }
# run_id of a segment: parsed from the runner's first stdout line.
seg_run_id() { awk '/^run_id:/{print $2; exit}' "$STATE/jobs/$1/seg-$2.log" 2>/dev/null; }
unit_active() { systemctl --user is-active --quiet "$(unit_of "$1")" 2>/dev/null; }
audit_event() { [ -f "$AUDIT" ] && jq -c --arg r "$1" --arg e "$2" 'select(.run_id==$r and .event==$e)' "$AUDIT" | tail -1; }

# GCS markers for a run: prints DONE, FAILED or nothing; CKPT=1 when the run
# wrote a ckpt-latest.json pointer.
gcs_objects() { gcloud storage ls "gs://$BUCKET/runs/$1/" 2>/dev/null; }

# Classify one segment. Prints: <state> <run_id|-> <detail>
#   running | done | resumable | failed | refused | orphan
seg_state() {
  local job=$1 n=$2 rid objs launch destroy
  rid=$(seg_run_id "$job" "$n")
  if unit_active "$job" && [ "$n" = "$(seg_count "$job")" ]; then echo "running ${rid:--} unit-active"; return; fi
  [ -n "$rid" ] || { echo "refused - no-run_id(see seg-$n.log)"; return; }
  launch=$(audit_event "$rid" launch)
  [ -n "$launch" ] || { echo "refused $rid no-launch-record"; return; }
  destroy=$(audit_event "$rid" destroy)
  objs=$(gcs_objects "$rid")
  if [ -z "$destroy" ]; then echo "orphan $rid launched-not-destroyed(run: hpo-campaign.sh spend; runner reap)"; return; fi
  if grep -q '/DONE$' <<<"$objs"; then echo "done $rid $(jq -r .reason <<<"$destroy")"; return; fi
  if grep -q '/ckpt-latest.json$' <<<"$objs"; then echo "resumable $rid $(jq -r .reason <<<"$destroy")"; return; fi
  echo "failed $rid $(jq -r .reason <<<"$destroy")"
}

# Job state = its last segment's state (or not-launched).
job_state() {
  local n; n=$(seg_count "$1")
  [ "$n" -gt 0 ] || { echo "not-launched - -"; return; }
  seg_state "$1" "$n"
}

# Newest segment of a job whose run left a ckpt-latest.json (resume source).
latest_ckpt_run() {
  local job=$1 n rid
  for (( n=$(seg_count "$job"); n>=1; n-- )); do
    rid=$(seg_run_id "$job" "$n"); [ -n "$rid" ] || continue
    gcs_objects "$rid" | grep -q '/ckpt-latest.json$' && { echo "$rid"; return 0; }
  done
  return 1
}

# ---- money ------------------------------------------------------------------
# Worst case still exposed by running segments = their --max-usd (the runner
# destroys at 95% of it). An orphan (launched, unit gone, no destroy record)
# is counted at its full --max-usd, both here and in spent.
running_worst() {
  local total=0 job n st m
  for job in $(jobs_all); do
    n=$(seg_count "$job"); [ "$n" -gt 0 ] || continue
    st=$(seg_state "$job" "$n" | cut -d' ' -f1)
    if [ "$st" = running ] || [ "$st" = orphan ]; then
      m=$(seg_get "$job" "$n" | jq -r .max_usd); total=$(calc "$total + $m")
    fi
  done
  echo "$total"
}

# Spend recorded by the runner's audit log for this campaign's runs.
audit_spent() {
  local total=0 job n rid d c
  for job in $(jobs_all); do
    for (( n=1; n<=$(seg_count "$job"); n++ )); do
      rid=$(seg_run_id "$job" "$n"); [ -n "$rid" ] || continue
      d=$(audit_event "$rid" destroy)
      if [ -n "$d" ]; then c=$(jq -r '.est_cost_usd // 0' <<<"$d"); total=$(calc "$total + $c"); fi
    done
  done
  echo "$total"
}

# spent = max(audit sum, credit delta since the manifest's baseline). The
# credit delta also catches spend the audit log cannot see (and other lanes'
# spend, which only makes the gate more conservative).
spent_usd() {
  local a base now delta
  a=$(audit_spent); base=$(mf .budget.credit_at_creation_usd); now=${1:-$(credit_now)}
  delta=$(calc "$base - ${now:-$base}")
  if gt "$delta" "$a"; then echo "$delta"; else echo "$a"; fi
}

# The campaign gate. $1 = this launch's --max-usd. Refuses (return 1, reason
# on stderr) when spent + worst(running) + this > campaign cap, or > the
# baseline credit minus the reserve, or when worst(running) + this exceeds the
# live credit minus the reserve.
budget_gate() {
  local this=$1 cap reserve base now spent run total
  cap=$(mf .budget.campaign_cap_usd); reserve=$(mf .budget.credit_reserve_usd)
  base=$(mf .budget.credit_at_creation_usd)
  now=$(credit_now); [ -n "$now" ] || { log "gate: live credit unavailable; refusing"; return 1; }
  spent=$(spent_usd "$now"); run=$(running_worst)
  total=$(calc "$spent + $run + $this")
  log "gate: spent \$$spent + running worst \$$run + this \$$this = \$$total (cap \$$cap; baseline credit \$$base - \$$reserve; live credit \$$now)"
  if gt "$total" "$cap"; then log "gate: REFUSE (campaign cap \$$cap)"; return 1; fi
  if gt "$total" "$(calc "$base - $reserve")"; then log "gate: REFUSE (baseline credit \$$base - reserve \$$reserve)"; return 1; fi
  if gt "$(calc "$run + $this")" "$(calc "$now - $reserve")"; then log "gate: REFUSE (live credit \$$now - reserve \$$reserve)"; return 1; fi
  return 0
}

# ---- runner invocation ----------------------------------------------------------
# Fills the global array RARGS for `ruvector-gpu-runner ... launch`.
#   $1 job  $2 seg  $3 sha  $4 threads  $5 min-cpu-cores  $6 resume-from|-  $7 dry(0/1)  $8 allow-concurrent(0/1)
build_args() {
  local job=$1 seg=$2 sha=$3 thr=$4 mincores=$5 resume=$6 dry=$7 conc=$8 ds cfg hours usd ckgb cmd
  ds=$(job_field "$job" .dataset); cfg=$(job_field "$job" .config)
  hours=$(job_field "$job" ".segments[$((seg-1))].max_hours // .extra_segment.max_hours")
  usd=$(job_field "$job" ".segments[$((seg-1))].max_usd // .extra_segment.max_usd")
  ckgb=$(job_field "$job" .expected_checkpoint_gb)
  cmd="bash scripts/kge/vast-kge.sh --dataset $ds --config-id $cfg --seed $(mf .protocol.hpo_seed)"
  cmd+=" --max-epochs $(mf .protocol.max_epochs) --early-stop-patience $(mf .protocol.early_stop.patience)"
  cmd+=" --eval-every $(mf .protocol.early_stop.eval_every) --threads $thr"
  RARGS=(--audit-log "$AUDIT" launch --cpu-mode
    --min-cpu-cores "$mincores" --min-ram-gb "$(mf .runner.min_ram_gb)"
    --min-cpu-speed "$(mf .runner.min_cpu_speed)" --max-dph "$(mf .runner.max_dph)"
    --disk-gb "$(mf .runner.disk_gb)" --max-usd "$usd" --max-hours "$hours"
    --git-ref "$GIT_REF" --git-sha "$sha" --local-repo "$REPO" --bucket "$BUCKET"
    --artifact-dir /workspace/out --checkpoint-dir /workspace/ckpt
    --checkpoint-secs "$(mf .runner.checkpoint_secs)" --checkpoint-ring "$(mf .runner.checkpoint_ring)"
    --expected-checkpoint-gb "$ckgb" --expected-upload-gb "$(mf .runner.expected_upload_gb)"
    --expected-download-gb "$(mf .runner.expected_download_gb)" --cmd "$cmd")
  [ "$resume" = - ] || RARGS+=(--resume-from "$resume")
  [ "$dry" = 1 ] && RARGS+=(--dry-run)
  [ "$conc" = 1 ] && RARGS+=(--allow-concurrent)
  return 0
}

# Pinned SHA for new jobs: HEAD, clean, equal to origin/<ref>, and the
# manifest is committed at it. Resume never calls this (it reuses the job's SHA).
pinned_head() {
  local head remote
  (cd "$REPO" && git diff --quiet && git diff --cached --quiet) || die "worktree has uncommitted changes; commit + push first"
  (cd "$REPO" && git ls-files --error-unmatch "$MANIFEST" >/dev/null 2>&1) || die "manifest is not committed"
  head=$(cd "$REPO" && git rev-parse HEAD)
  remote=$(cd "$REPO" && git ls-remote origin "refs/heads/$GIT_REF" | cut -f1)
  [ "$head" = "$remote" ] || die "HEAD $head != origin/$GIT_REF ${remote:-<missing>}; push first"
  echo "$head"
}

# The units need a live systemd --user manager (user@UID.service). If it is
# down, systemctl --user cannot see units, so launching would leave instances
# the status/gate logic could not track: refuse instead.
user_bus_ok() { systemctl --user show-environment >/dev/null 2>&1; }
need_user_bus() {
  user_bus_ok || die "systemd --user manager unreachable (user@$(id -u).service down?). Start it first, e.g. 'sudo systemctl start user@$(id -u).service'; nothing launched"
}

# A job whose segments never produced a checkpoint (runner refused, or the
# instance died before the first checkpoint) cannot be resumed: move its
# segment records and logs to attempts/<n>/ so `launch` can start it afresh
# (and a stale run_id cannot shadow the new one).
archive_attempt() {
  local d="$STATE/jobs/$1" n=1
  while [ -e "$d/attempts/$n" ]; do n=$((n + 1)); done
  mkdir -p "$d/attempts/$n"
  mv "$d/segments.jsonl" "$d"/seg-*.log "$d/attempts/$n/" 2>/dev/null || true
  log "$1: no checkpoint from previous attempt; archived to $d/attempts/$n"
}
