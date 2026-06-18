#!/usr/bin/env bash
#
# extract-ruqu-rvdna-submodules.sh — ADR-257
#
# Extract `ruqu` (both clusters) and `rvdna` from the ruvector monorepo into
# standalone ruvnet repos and re-reference them as git submodules.
#
# SAFETY: dry-run by default. It prints every command and touches nothing until
# you pass --execute. The GitHub repo creation + history push are IRREVERSIBLE;
# review ADR-257 (docs/adr/ADR-257-ruqu-rvdna-standalone-submodules.md) first.
#
# Requirements: git, git-filter-repo (`pip install git-filter-repo`), gh (authed),
# cargo, npm.
#
# Usage:
#   scripts/extract-ruqu-rvdna-submodules.sh            # dry-run (default)
#   scripts/extract-ruqu-rvdna-submodules.sh --execute  # actually do it
#
set -euo pipefail

EXECUTE=0
[[ "${1:-}" == "--execute" ]] && EXECUTE=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

ORG="ruvnet"
WORK="$(mktemp -d)/extract"          # scratch area for filtered clones
RUQU_MOUNT="external/ruqu"
RVDNA_MOUNT="external/rvdna"

# Paths that move into ruvnet/ruqu (kept at the same internal sub-paths).
RUQU_PATHS=(
  crates/ruqu-core crates/ruqu-algorithms crates/ruqu-exotic
  crates/ruqu-wasm crates/ruQu npm/packages/ruqu-wasm
)
# Paths that move into ruvnet/rvdna.
RVDNA_PATHS=( examples/dna npm/packages/rvdna )

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
run()  { printf '  \033[2m$ %s\033[0m\n' "$*"; if [[ $EXECUTE -eq 1 ]]; then eval "$@"; fi; }
note() { printf '  \033[2m# %s\033[0m\n' "$*"; }

[[ $EXECUTE -eq 0 ]] && say "DRY-RUN (no changes). Re-run with --execute to apply."

# ---------------------------------------------------------------------------
say "0. Pre-flight"
# ---------------------------------------------------------------------------
run "git diff --quiet && git diff --cached --quiet"   # clean tree
command -v git-filter-repo >/dev/null 2>&1 || python -m git_filter_repo --version >/dev/null 2>&1 \
  || { echo "ERROR: git-filter-repo not found (pip install git-filter-repo)"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "ERROR: gh not authenticated"; exit 1; }
for r in ruqu rvdna; do
  if gh repo view "$ORG/$r" >/dev/null 2>&1; then
    echo "ERROR: $ORG/$r already exists — refusing to overwrite."; exit 1
  fi
done
note "scratch dir: $WORK"
run "mkdir -p '$WORK'"

# ---------------------------------------------------------------------------
extract_repo() {  # $1=name  $2=mount  shift 2 => paths
  local name="$1" mount="$2"; shift 2
  local paths=("$@")
  local src="$WORK/$name"
  say "Extract $name (history-preserving) -> $ORG/$name"
  run "git clone --no-local '$REPO_ROOT' '$src'"
  local filter_args=""
  for p in "${paths[@]}"; do filter_args+=" --path '$p'"; done
  # filter-repo keeps ONLY these paths, preserving their commit history.
  run "( cd '$src' && git filter-repo --force $filter_args )"
  run "gh repo create '$ORG/$name' --public --description 'Extracted from ruvnet/ruvector (ADR-257)' --disable-wiki"
  run "( cd '$src' && git remote add origin 'https://github.com/$ORG/$name.git' && git push -u origin HEAD:main )"
}

extract_repo ruqu  "$RUQU_MOUNT"  "${RUQU_PATHS[@]}"
extract_repo rvdna "$RVDNA_MOUNT" "${RVDNA_PATHS[@]}"

# ---------------------------------------------------------------------------
say "2. Remove originals from the superproject"
# ---------------------------------------------------------------------------
for p in "${RUQU_PATHS[@]}" "${RVDNA_PATHS[@]}"; do
  run "git rm -r --quiet '$p'"
done

# ---------------------------------------------------------------------------
say "3. Add submodules at consolidated mount points"
# ---------------------------------------------------------------------------
run "git submodule add 'https://github.com/$ORG/ruqu.git'  '$RUQU_MOUNT'"
run "git submodule add 'https://github.com/$ORG/rvdna.git' '$RVDNA_MOUNT'"

# ---------------------------------------------------------------------------
say "4. Rewrite references (see ADR-257 §Decision.5)"
# ---------------------------------------------------------------------------
note "root Cargo.toml: members crates/ruqu-* + crates/ruQu -> external/ruqu/crates/...; examples/dna -> external/rvdna/examples/dna"
note "examples/OSpipe/Cargo.toml:39  path ../../crates/ruqu-algorithms -> ../../external/ruqu/crates/ruqu-algorithms"
note "examples/rvf/Cargo.toml:27-28  path ../../crates/ruqu-{core,algorithms} -> ../../external/ruqu/crates/..."
note ".github/workflows/ci.yml: add 'submodules: recursive' to checkout; ruqu-quantum shard manifest-path -> external/ruqu/..."
note "npm/package.json workspaces: add external/ruqu/npm/* and external/rvdna/npm/* (or keep wrappers under npm/ via path)"
note "npm/packages/rvdna/package.json build:napi --cargo-cwd: now relative to external/rvdna/examples/dna"
note "npm/packages/ruqu-wasm/package.json: repository.directory + homepage -> ruvnet/ruqu"
note "examples/dna/adr/ADR-002:755, ADR-010:925 cross-repo links -> point at ruvnet/ruqu blob URLs"
note "(These edits are applied by the companion patch in this PR, not auto-generated here, so they stay reviewable.)"

# ---------------------------------------------------------------------------
say "5. Regenerate lockfiles + verify"
# ---------------------------------------------------------------------------
run "cargo metadata --no-deps --format-version 1 >/dev/null"   # workspace resolves
run "cargo check -p ruqu-core"
run "cargo check -p ospipe"          # consumer through the submodule
run "( cd npm && npm install )"      # regenerate npm/package-lock.json

say "Done. Review 'git status', then commit and open a PR bumping the submodule pointers."
[[ $EXECUTE -eq 0 ]] && say "This was a DRY-RUN. Nothing changed."
