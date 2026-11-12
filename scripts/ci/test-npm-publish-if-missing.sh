#!/usr/bin/env bash
# Tests for npm-publish-if-missing.sh using a stub `npm` on PATH (no network).
#   bash scripts/ci/test-npm-publish-if-missing.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
helper="$here/npm-publish-if-missing.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/bin" "$work/pkg"
printf '{"name":"@scope/demo","version":"1.2.3"}\n' > "$work/pkg/package.json"

# Stub npm. STUB_VIEW: "exists" | "missing" | "empty" (exit 0, no output).
# STUB_PUBLISH: "ok" | "fail". Every publish call is logged to $STUB_LOG.
cat > "$work/bin/npm" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  view)
    case "$STUB_VIEW" in
      exists) echo "1.2.3" ;;
      missing) echo "npm error code E404" >&2; exit 1 ;;
      empty) ;;
    esac ;;
  publish)
    echo "publish ${*:2}" >> "$STUB_LOG"
    [ "$STUB_PUBLISH" = ok ] || { echo "npm error code E403" >&2; exit 1; } ;;
esac
STUB
chmod +x "$work/bin/npm"
export PATH="$work/bin:$PATH" STUB_LOG="$work/log"

fails=0
run_case() { # name view publish expect_rc expect_publishes [args...]
  local name=$1 view=$2 pub=$3 want_rc=$4 want_pubs=$5; shift 5
  : > "$STUB_LOG"
  local rc=0
  (cd "$work/pkg" && STUB_VIEW=$view STUB_PUBLISH=$pub bash "$helper" "$@") >/dev/null 2>&1 || rc=$?
  local pubs; pubs=$(wc -l < "$STUB_LOG")
  if { [ "$want_rc" = 0 ] && [ "$rc" -ne 0 ]; } || { [ "$want_rc" = nz ] && [ "$rc" -eq 0 ]; } || [ "$pubs" -ne "$want_pubs" ]; then
    echo "FAIL: $name (rc=$rc, publishes=$pubs)"; fails=$((fails + 1))
  else
    echo "ok:   $name"
  fi
}

run_case "existing version is skipped"          exists  fail 0  0
run_case "missing version is published"         missing ok   0  1
run_case "failing publish exits non-zero"       missing fail nz 1
run_case "empty npm view output still publishes" empty  ok   0  1
run_case "extra args forwarded"                 missing ok   0  1 --ignore-scripts
if grep -q -- '--ignore-scripts --access public' "$STUB_LOG"; then
  echo "ok:   default --access public appended"
else
  echo "FAIL: args not forwarded: $(cat "$STUB_LOG")"; fails=$((fails + 1))
fi

# A workflow-style loop (a fresh `bash -e` process, like an Actions `run:`
# step) must stop at the first failing publish. A separate process is needed:
# errexit is suspended inside a subshell that sits in an `||` context.
: > "$STUB_LOG"
rc=0
# shellcheck disable=SC2016  # $1/$2 expand in the child shell
STUB_VIEW=missing STUB_PUBLISH=fail bash -e -c '
  for _ in 1 2 3; do
    cd "$1"
    bash "$2"
  done' _ "$work/pkg" "$helper" >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ] && [ "$(wc -l < "$STUB_LOG")" -eq 1 ]; then
  echo "ok:   loop aborts on first failed publish"
else
  echo "FAIL: loop did not abort (rc=$rc)"; fails=$((fails + 1))
fi

if [ "$fails" -ne 0 ]; then echo "$fails failed"; exit 1; fi
echo "all passed"
