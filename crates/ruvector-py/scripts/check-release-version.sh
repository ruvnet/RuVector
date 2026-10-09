#!/usr/bin/env bash
# Fail unless pyproject.toml, Cargo.toml (and, if given, the git tag and the
# built artifacts in a dist dir) all carry the same version.
#
# usage: check-release-version.sh [tag] [dist-dir]
#   tag       e.g. python-v0.1.0 (empty/omitted: skip the tag comparison)
#   dist-dir  directory holding the wheels/sdist about to be published
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
first_version() { awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$1"; }

py=$(first_version "$here/pyproject.toml")
rs=$(first_version "$here/Cargo.toml")
fail=0

[ -n "$py" ] && [ -n "$rs" ] || { echo "::error::could not read version from pyproject.toml/Cargo.toml"; exit 1; }
if [ "$py" != "$rs" ]; then
  echo "::error::pyproject.toml version ($py) != Cargo.toml version ($rs)"; fail=1
fi

tag=${1:-}
if [ -n "$tag" ]; then
  tag_version=${tag#refs/tags/}
  tag_version=${tag_version#python-v}
  if [ "$tag_version" != "$py" ]; then
    echo "::error::tag '$tag' implies version $tag_version but pyproject.toml says $py"; fail=1
  fi
fi

dist=${2:-}
if [ -n "$dist" ]; then
  shopt -s nullglob
  files=("$dist"/ruvector-*)
  [ "${#files[@]}" -gt 0 ] || { echo "::error::no ruvector-* artifacts in $dist"; exit 1; }
  for f in "${files[@]}"; do
    v=$(basename "$f"); v=${v#ruvector-}; v=${v%%-*}; v=${v%.tar.gz}
    if [ "$v" != "$py" ]; then
      echo "::error::$(basename "$f") has version $v but pyproject.toml says $py"; fail=1
    fi
  done
fi

[ "$fail" -eq 0 ] && echo "version check OK: $py"
exit "$fail"
