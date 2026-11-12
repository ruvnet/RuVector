#!/usr/bin/env bash
# Fail if any checked-in npm platform-binary manifest (NAPI-RS
# `<pkg>-linux-x64-gnu`, `<pkg>-darwin-arm64`, `<pkg>-win32-x64-msvc`, ...)
# is missing the canonical homepage, repository or bugs fields (ADR-356).
#
#   bash scripts/ci/check-platform-manifests.sh
#
# Platform variants share the parent package's page, so every one must carry
# "homepage": "https://ruv.io", a git repository pointing at its own directory,
# and the RuVector issue tracker. Workflow steps that generate these manifests
# at publish time write the same fields; this check covers the checked-in ones.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

suffix='-(linux|darwin|win32|android|freebsd)-(x64|arm64|ia32|arm|riscv64|s390x|ppc64|universal)(-(gnu|musl|msvc|gnueabihf))?$'
repo_url='git+https://github.com/ruvnet/RuVector.git'
bugs_url='https://github.com/ruvnet/RuVector/issues'

checked=0
failed=0
while IFS= read -r f; do
  name=$(jq -r '.name // empty' "$f" 2>/dev/null) || {
    echo "invalid JSON: $f"
    failed=$((failed + 1))
    continue
  }
  grep -Eq -- "$suffix" <<<"$name" || continue
  checked=$((checked + 1))
  dir=$(dirname "$f")
  problems=$(jq -r --arg url "$repo_url" --arg bugs "$bugs_url" --arg dir "$dir" '
    [ (if .homepage != "https://ruv.io" then "homepage" else empty end),
      (if (.repository | type) != "object" or .repository.url != $url
            or .repository.directory != $dir then "repository" else empty end),
      (if ((.bugs | type) == "object" and .bugs.url == $bugs) or .bugs == $bugs
         then empty else "bugs" end)
    ] | join(", ")' "$f")
  if [ -n "$problems" ]; then
    echo "$f ($name): bad $problems"
    failed=$((failed + 1))
  fi
done < <(git ls-files '*package.json')

if [ "$failed" -gt 0 ]; then
  echo "check-platform-manifests: $failed of $checked platform manifest(s) need homepage https://ruv.io, repository $repo_url (directory = own dir) and bugs $bugs_url" >&2
  exit 1
fi
echo "check-platform-manifests: $checked platform manifest(s) OK"
