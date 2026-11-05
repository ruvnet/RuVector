#!/usr/bin/env bash
# Publish the npm package in the current directory unless that exact
# name@version is already on the registry.
#
#   bash scripts/ci/npm-publish-if-missing.sh [extra `npm publish` args...]
#
# Reruns stay idempotent (an existing version is skipped with exit 0), but a
# real publish failure (auth, network, bad tarball, ...) exits non-zero so the
# CI step goes red. This replaces the old `npm publish ... || echo "..."`
# pattern, which reported every failure as success.
#
# Extra arguments are forwarded to `npm publish`; `--access public` is added
# when the caller does not pass an --access flag. NODE_AUTH_TOKEN and any
# provenance flags are left to the caller's environment/arguments.
set -euo pipefail

if [ ! -f package.json ]; then
  echo "npm-publish-if-missing: no package.json in $(pwd)" >&2
  exit 1
fi

name=$(node -p "require('./package.json').name")
version=$(node -p "require('./package.json').version")

if [ -z "$name" ] || [ "$name" = "undefined" ] || [ -z "$version" ] || [ "$version" = "undefined" ]; then
  echo "npm-publish-if-missing: package.json in $(pwd) lacks name/version" >&2
  exit 1
fi

# Compare stdout rather than trusting the exit code: some npm versions exit 0
# with empty output for a missing version of an existing package.
published=$(npm view "$name@$version" version 2>/dev/null || true)

if [ "$published" = "$version" ]; then
  echo "$name@$version already on npm, skipping"
  exit 0
fi

args=("$@")
has_access=0
for a in "${args[@]+"${args[@]}"}"; do
  case "$a" in --access|--access=*) has_access=1 ;; esac
done
if [ "$has_access" -eq 0 ]; then
  args+=(--access public)
fi

echo "Publishing $name@$version"
npm publish "${args[@]}"
