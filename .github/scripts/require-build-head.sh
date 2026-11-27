#!/usr/bin/env bash
# Mutation jobs hold ruvector-main-mutation while using this exact source.
set -euo pipefail

if [[ ! ${GITHUB_SHA:-} =~ ^[a-f0-9]{40}$ ]] || [[ ${GITHUB_REF:-} != refs/heads/* ]]; then
  echo 'Artifact updates require an exact source SHA and a branch ref' >&2
  exit 1
fi
git check-ref-format "$GITHUB_REF"
if [[ $(git rev-parse HEAD) != "$GITHUB_SHA" ]]; then
  echo 'Checkout differs from the source that built these artifacts' >&2
  exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo 'Artifact updates require a clean source checkout' >&2
  exit 1
fi
remote_head=$(git ls-remote --exit-code origin "$GITHUB_REF" | cut -f1)
if [[ $remote_head != "$GITHUB_SHA" ]]; then
  echo 'Target branch advanced; refusing to apply artifacts from a stale build' >&2
  exit 1
fi
