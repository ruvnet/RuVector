#!/usr/bin/env bash
# ADR-007 M7 HPO campaign launcher (plan M4/M6/M7): every dataset x C1..C8 at
# HPO seed 100, <= 100 epochs, early stop on valid MRR (Bottom), on vast.ai CPU
# instances through ruvector-gpu-runner. VALIDATION ONLY: nothing here, in the
# job script (scripts/kge/vast-kge.sh) or in `collect` can score test; there is
# no --final path.
#
# The manifest (npm/packages/kge/bench/results/hpo/campaign-2026-09-27.json)
# fixes every job, wave, per-segment --max-hours / --max-usd and the budget
# rules; this script only executes it. A "job" is one (dataset, config) HPO
# run. A job longer than the runner's 11.5 h signed-URL cap runs as a chain of
# segments: segment 1 via `launch`, segments 2.. via `resume` (from the newest
# GCS checkpoint, same SHA, same --threads).
#
# Subcommands:
#   plan [--wave N|--dataset D|--job J]   dry-run segment 1 of every job via
#                                         the runner's --dry-run; prints offers,
#                                         $/h and worst-case $ (no launch)
#   launch --wave N --max-concurrent M    start not-yet-launched jobs of wave N
#                                         as systemd --user units
#                                         kge-hpo-<dataset>-<cfg> (--collect)
#   status                                per job: not-launched / running /
#                                         done / resumable / failed / orphan
#   resume <job> [--allow-extra]          next segment of <job> from its
#                                         latest checkpoint (--resume-from)
#   collect [dataset|all]                 fetch DONE receipts from GCS and,
#                                         once all 8 configs of a dataset are
#                                         done, run the bench's valid-only
#                                         selection -> selection.json
#   spend                                 real spend: credit delta + audit log
#
# Budget gate (before every paid launch or resume): refuse if
#   spent + worst(running) + this > campaign cap ($80)
#   or   > credit at manifest creation - $10 reserve
#   or   worst(running) + this > live credit - $10.
# worst(x) is the segment's --max-usd (the runner destroys at 95% of it);
# spent = max(audit-log est_cost of this campaign's runs, credit delta).
#
# Secrets: VAST_API_KEY is read from Secret Manager into this process (and,
# inside each unit, by `_unit`), never printed, never on an argv or in a unit
# file. Never run `ruvector-gpu-runner reap --yes` while any kge-hpo unit is
# active: it destroys every rvgr-* instance.
set -euo pipefail
# shellcheck source=scripts/kge/hpo-campaign-lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/hpo-campaign-lib.sh"
SELF="$REPO/scripts/kge/hpo-campaign.sh"

usage() { sed -n 2,40p "$SELF"; exit "${1:-0}"; }

# Dry-run one segment; prints the runner's run_id (the audit key) on stdout,
# the runner's full output to fd 3 (or /dev/null), returns the runner's code.
dry_run() {
  local out rc=0
  out=$("$RUNNER" "${RARGS[@]}" 2>&1) || rc=$?
  printf '%s\n' "$out" >&3
  awk '/^run_id:/{print $2; exit}' <<<"$out"
  return "$rc"
}

any_running() { local w; w=$(running_worst); gt "$w" 0; }

cmd_plan() {
  local sel=() job rid rec seg nseg sha
  case "${1:-}" in
    --wave) mapfile -t sel < <(jobs_wave "$2") ;;
    --dataset) mapfile -t sel < <(jobs_dataset "$2") ;;
    --job) valid_job "$2" || die "unknown job $2"; sel=("$2") ;;
    "") mapfile -t sel < <(jobs_all) ;;
    *) die "plan: unknown option $1" ;;
  esac
  need_key; need_runner
  sha=$(cd "$REPO" && git rev-parse HEAD)
  local out; out="$STATE/plan-$(date -u +%Y%m%dt%H%M%S).tsv"
  printf 'job\twave\tsegs\toffer\tcpu\tthreads\tusd_per_h\tseg1_runner_worst_usd\tjob_worst_usd_at_offer\tjob_cap_usd\texpected_usd_100ep\tverdict\n' > "$out"
  local conc=0; any_running && conc=1
  for job in "${sel[@]}"; do
    build_args "$job" 1 "$sha" 32 "$(mf .runner.min_cpu_cores)" - 1 "$conc"
    rid=$(dry_run 3>"$STATE/jobs/.plan-$job.log") || true
    rec=$(audit_event "${rid:-none}" dry_run)
    if [ -z "$rec" ]; then
      printf '%s\t%s\t-\t-\t-\t-\t-\t-\t-\t-\t-\tERROR(see %s)\n' "$job" "$(job_field "$job" .wave)" "$STATE/jobs/.plan-$job.log" >> "$out"; continue
    fi
    # Extrapolate later segments from segment 1's offer: rate x hours plus
    # the offer's bandwidth $/GB (seg-1 bandwidth / seg-1 GB) x each segment's GB.
    nseg=$(job_field "$job" '.segments|length')
    local wjob
    wjob=$(jq -r --argjson rec "$rec" --arg j "$job" '
      .jobs[]|select(.id==$j) as $J
      | ($rec.hourly_usd) as $h
      | (($rec.worst_case_usd - $h*$J.segments[0].max_hours) / $J.segments[0].transfer_gb) as $pgb
      | [$J.segments[] | $h*.max_hours + $pgb*.transfer_gb] | add | .*100|ceil/100' "$MANIFEST")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%.2f\t%s\t%s\t%s\t%s\n' "$job" "$(job_field "$job" .wave)" "$nseg" \
      "$(jq -r .offer_id <<<"$rec")" "$(jq -r '.cpu_name|gsub("\\s+$";"")' <<<"$rec")" \
      "$(jq -r '.cpu_cores_effective|floor' <<<"$rec")" "$(jq -r .hourly_usd <<<"$rec")" \
      "$(jq -r .worst_case_usd <<<"$rec")" "$wjob" "$(job_field "$job" .worst_usd)" \
      "$(job_field "$job" .expected_usd_100ep)" \
      "$(jq -r 'if (.blockers|length)==0 then "WOULD_LAUNCH" else "REFUSE:"+(.blockers|join("; ")) end' <<<"$rec")" >> "$out"
  done
  column -t -s $'\t' "$out"
  awk -F'\t' 'NR>1 && $9!="-" {w+=$9; c+=$10; e+=$11} END{printf "TOTAL worst-case at chosen offers $%.2f | manifest cap sum $%.2f | expected (100 epochs, no margin) $%.2f\n", w, c, e}' "$out"
  echo "effective campaign cap: \$$(mf .budget.effective_cap_usd) (min(\$$(mf .budget.campaign_cap_usd), credit \$$(mf .budget.credit_at_creation_usd) - \$$(mf .budget.credit_reserve_usd)))"
  echo "plan table: $out"
}

# Start one segment as a transient user unit. $1 job $2 seg $3 sha $4 threads $5 resume|-
start_unit() {
  local job=$1 seg=$2 sha=$3 thr=$4 resume=$5 conc=0 dir usd hours
  dir="$STATE/jobs/$job"; mkdir -p "$dir"
  any_running && conc=1
  build_args "$job" "$seg" "$sha" "$thr" "$thr" "$resume" 0 "$conc"
  usd=$(job_field "$job" ".segments[$((seg-1))].max_usd // .extra_segment.max_usd")
  hours=$(job_field "$job" ".segments[$((seg-1))].max_hours // .extra_segment.max_hours")
  budget_gate "$usd" || return 1
  systemd-run --user --quiet --collect --unit="$(unit_of "$job")" \
    -p TimeoutStopSec=900 --working-directory="$REPO" \
    --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=KGE_HPO_STATE="$STATE" \
    -- bash "$SELF" _unit "$job" "$seg" -- "${RARGS[@]}"
  jq -nc --argjson seg "$seg" --arg unit "$(unit_of "$job")" --arg sha "$sha" --argjson thr "$thr" \
    --argjson h "$hours" --argjson u "$usd" --arg at "$(date -u +%FT%TZ)" --arg r "$resume" \
    --arg log "$dir/seg-$seg.log" \
    '{seg:$seg,unit:$unit,sha:$sha,threads:$thr,max_hours:$h,max_usd:$u,launched_at:$at,resume_from:$r,log:$log}' \
    >> "$(seg_file "$job")"
  log "$job: segment $seg started as $(unit_of "$job") (threads $thr, --max-hours $hours, --max-usd $usd, sha ${sha:0:12})"
}

cmd_launch() {
  local wave="" maxc=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --wave) wave=$2; shift 2 ;; --max-concurrent) maxc=$2; shift 2 ;;
      *) die "launch: unknown option $1" ;;
    esac
  done
  [[ "$wave" =~ ^[1-9]$ ]] && [[ "$maxc" =~ ^[1-9][0-9]*$ ]] || die "launch needs --wave N --max-concurrent M"
  need_key; need_runner
  local sha job active rid rec cores thr
  sha=$(pinned_head)
  for job in $(jobs_wave "$wave"); do
    active=$(for j in $(jobs_all); do unit_active "$j" && echo "$j"; done | wc -l)
    [ "$active" -lt "$maxc" ] || { log "max-concurrent $maxc reached; stopping"; break; }
    [ "$(seg_count "$job")" = 0 ] || { log "$job: already launched ($(job_state "$job" | cut -d' ' -f1)); skip (use resume)"; continue; }
    # Dry-run first to learn the offer's allocated threads; launch then pins
    # --threads and --min-cpu-cores to it (a later resume must reuse it).
    build_args "$job" 1 "$sha" 32 "$(mf .runner.min_cpu_cores)" - 1 "$(any_running && echo 1 || echo 0)"
    rid=$(dry_run 3>"$STATE/jobs/.dry-$job.log") || { log "$job: dry run refused (see $STATE/jobs/.dry-$job.log)"; continue; }
    rec=$(audit_event "$rid" dry_run)
    cores=$(jq -r '.cpu_cores_effective // 0 | floor' <<<"$rec")
    [ "$cores" -ge "$(mf .runner.min_cpu_cores)" ] || { log "$job: dry run offer has $cores threads; skip"; continue; }
    thr=$cores
    start_unit "$job" 1 "$sha" "$thr" - || { log "$job: not launched (budget gate)"; break; }
    sleep 5
  done
}

cmd_resume() {
  local job=${1:-} extra=0 n last st sha thr rid planned
  valid_job "$job" || die "resume: unknown job '$job'"
  [ "${2:-}" = --allow-extra ] && extra=1
  need_key; need_runner
  n=$(seg_count "$job"); [ "$n" -gt 0 ] || die "$job was never launched (use launch)"
  st=$(seg_state "$job" "$n")
  case "${st%% *}" in
    running) die "$job: segment $n still running" ;;
    done) die "$job: already done" ;;
    orphan) die "$job: segment $n is an orphan instance; resolve it first ($st)" ;;
  esac
  planned=$(job_field "$job" '.segments|length')
  [ $((n + 1)) -le "$planned" ] || [ "$extra" = 1 ] || die "$job: $planned planned segments used; --allow-extra uses the manifest's extra_segment caps"
  last=$(seg_get "$job" "$n"); sha=$(jq -r .sha <<<"$last"); thr=$(jq -r .threads <<<"$last")
  rid=$(latest_ckpt_run "$job") || die "$job: no run with a ckpt-latest.json to resume from"
  build_args "$job" $((n + 1)) "$sha" "$thr" "$thr" "$rid" 1 "$(any_running && echo 1 || echo 0)"
  dry_run 3>"$STATE/jobs/.dry-$job.log" >/dev/null || die "$job: resume dry run refused (see $STATE/jobs/.dry-$job.log)"
  start_unit "$job" $((n + 1)) "$sha" "$thr" "$rid"
}

cmd_status() {
  local job st
  printf '%-16s %-5s %-4s %-13s %-34s %s\n' JOB WAVE SEGS STATE RUN_ID DETAIL
  for job in $(jobs_all); do
    st=$(job_state "$job")
    printf '%-16s %-5s %-4s %-13s %-34s %s\n' "$job" "$(job_field "$job" .wave)" "$(seg_count "$job")/$(job_field "$job" '.segments|length')" \
      "$(cut -d' ' -f1 <<<"$st")" "$(cut -d' ' -f2 <<<"$st")" "$(cut -d' ' -f3- <<<"$st")"
  done
}

cmd_collect() {
  local want=${1:-all} ds job cfg rid tmp runs res n ok
  [[ "$want" =~ ^(all|fb15k237|wn18rr|codexm)$ ]] || die "collect: dataset must be fb15k237|wn18rr|codexm|all"
  for ds in fb15k237 wn18rr codexm; do
    [ "$want" = all ] || [ "$want" = "$ds" ] || continue
    res="$REPO/$(mf --arg d "$ds" '.datasets[$d].results_dir')"
    runs="$STATE/collect/$ds"; mkdir -p "$runs" "$res/raw-receipts"
    n=0
    for job in $(jobs_dataset "$ds"); do
      cfg=$(job_field "$job" .config)
      read -r ok rid _ <<<"$(job_state "$job")"
      [ "$ok" = "done" ] || { log "$job: $ok"; continue; }
      tmp=$(mktemp -d)
      gcloud storage cp -q "gs://$BUCKET/runs/$rid/artifacts.tar.gz" "$tmp/a.tgz" >/dev/null || { rm -rf "$tmp"; log "$job: download failed"; continue; }
      tar -xzf "$tmp/a.tgz" -C "$tmp" ./receipt.json 2>/dev/null || tar -xzf "$tmp/a.tgz" -C "$tmp" receipt.json
      # Guard: valid-only, pinned SHA, clean tree, the manifest's stopping rule.
      jq -e --arg sha "$(seg_get "$job" 1 | jq -r .sha)" --arg cfg "$cfg" --arg ds "$ds" \
        --argjson p "$(mf .protocol.early_stop.patience)" --argjson e "$(mf .protocol.max_epochs)" '
        .mode=="valid" and .eval.test_scored==false and (has("test")|not)
        and .provenance.git.sha==$sha and .provenance.git.dirty==false
        and .config.run.config_id==$cfg and .dataset.name==$ds and .seed==100
        and .config.run.max_epochs==$e and .config.run.early_stop_patience==$p
        and .stopped!="interrupted"' "$tmp/receipt.json" >/dev/null \
        || { rm -rf "$tmp"; die "$job: receipt fails the valid-only/SHA/protocol guard"; }
      mkdir -p "$runs/$cfg"; cp "$tmp/receipt.json" "$runs/$cfg/receipt.json"
      cp "$tmp/receipt.json" "$res/raw-receipts/$cfg.json"
      rm -rf "$tmp"; n=$((n + 1))
    done
    log "$ds: $n/8 receipts collected into $res/raw-receipts"
    [ "$n" = 8 ] || { log "$ds: selection deferred until all 8 configs are done"; continue; }
    [ -x "$BENCH" ] || (cd "$REPO" && cargo build --release -q -p ruvector-kge-bench) || die "bench build failed"
    # Selection only: every config already has a finished receipt, and a zero
    # wall cap means a missing one is recorded as not started, never trained.
    (cd "$REPO" && "$BENCH" --hpo --dataset "$ds" --configs "$res/configs.json" --runs-dir "$runs" \
      --out "$res" --wall-cap-mins 0 --parallel-jobs 1 --threads-per-job 1 --quiet)
    log "$ds: selection -> $res/selection.json (valid only)"
  done
}

cmd_spend() {
  local now base a job n rid d
  need_key
  now=$(credit_now); base=$(mf .budget.credit_at_creation_usd); a=$(audit_spent)
  printf 'credit at manifest creation  $%s\ncredit now                   $%s\ncredit delta                 $%s\n' "$base" "${now:-unknown}" "$(calc "$base - ${now:-$base}")"
  printf 'audit-log est. cost (campaign runs) $%s\n' "$a"
  printf 'spent (max of the two)       $%s\nworst case still running     $%s\n' "$(spent_usd "$now")" "$(running_worst)"
  for job in $(jobs_all); do
    for (( n=1; n<=$(seg_count "$job"); n++ )); do
      rid=$(seg_run_id "$job" "$n"); [ -n "$rid" ] || continue
      d=$(audit_event "$rid" destroy)
      printf '  %-16s seg %s %s %s\n' "$job" "$n" "$rid" "$( [ -n "$d" ] && jq -r '"$\(.est_cost_usd|.*100|round/100) \(.elapsed_h|.*100|round/100)h \(.reason)"' <<<"$d" || echo "(no destroy record)")"
    done
  done
}

# Inside the unit: fetch the key into this process only, then exec the runner.
cmd_unit() {
  local job=$1 seg=$2; shift 2; [ "${1:-}" = -- ] && shift
  valid_job "$job" || die "_unit: bad job"
  need_key
  exec "$RUNNER" "$@" >> "$STATE/jobs/$job/seg-$seg.log" 2>&1
}

case "${1:-}" in
  plan) shift; cmd_plan "$@" ;;
  launch) shift; cmd_launch "$@" ;;
  status) cmd_status ;;
  resume) shift; cmd_resume "$@" ;;
  collect) shift; cmd_collect "$@" ;;
  spend) cmd_spend ;;
  _unit) shift; cmd_unit "$@" ;;
  -h|--help|"") usage 0 ;;
  *) usage 2 ;;
esac
