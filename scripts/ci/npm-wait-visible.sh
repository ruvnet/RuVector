#!/usr/bin/env bash
# Wait until each name@version is publicly installable from the npm registry.
#
#   bash scripts/ci/npm-wait-visible.sh [--print-url] <name@version>...
#
# "Visible" means `npm view <spec> dist.tarball` returns an https URL and that
# tarball answers a HEAD request. npm can take several minutes to expose a
# fresh version (4.5 min for ruvector-core-linux-x64-gnu@0.1.34 on
# 2026-11-06), so a wrapper package must not be published until its platform
# packages pass this check, or `npm install` of the wrapper resolves no
# native binary in the meantime.
#
# Polls with exponential backoff (5 s doubling, capped at 60 s) until
# NPM_VISIBLE_TIMEOUT seconds (default 2400 = 40 min) have passed per spec.
# With --print-url, prints "<spec> <tarball-url>" per spec on stdout.
set -euo pipefail

print_url=0
if [ "${1:-}" = "--print-url" ]; then
  print_url=1
  shift
fi
if [ "$#" -eq 0 ]; then
  echo "usage: npm-wait-visible.sh [--print-url] <name@version>..." >&2
  exit 2
fi

timeout_s="${NPM_VISIBLE_TIMEOUT:-2400}"
max_delay="${NPM_VISIBLE_MAX_DELAY:-60}"

tarball_url() {
  # A failed `npm view --json` prints an error object; accept only a string
  # https URL. --prefer-online skips npm's local metadata cache.
  local out
  out="$(npm view "$1" dist.tarball --json --prefer-online 2>/dev/null)" || return 0
  printf '%s' "$out" | node -e '
    let s = "";
    process.stdin.on("data", (d) => (s += d)).on("end", () => {
      try {
        const v = JSON.parse(s);
        if (typeof v === "string" && v.startsWith("https://")) process.stdout.write(v);
      } catch {}
    });
  '
}

for spec in "$@"; do
  start=$SECONDS
  delay=5
  while :; do
    url="$(tarball_url "$spec")"
    if [[ "$url" == https://* ]] && curl --fail --silent --location --head --output /dev/null "$url"; then
      echo "$spec visible after $((SECONDS - start))s" >&2
      if [ "$print_url" -eq 1 ]; then
        printf '%s %s\n' "$spec" "$url"
      fi
      break
    fi
    elapsed=$((SECONDS - start))
    if [ "$elapsed" -ge "$timeout_s" ]; then
      echo "::error::$spec is not visible on the npm registry after ${elapsed}s" >&2
      exit 1
    fi
    echo "$spec not visible yet (${elapsed}s); retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$((delay * 2))
    if [ "$delay" -gt "$max_delay" ]; then
      delay=$max_delay
    fi
  done
done
