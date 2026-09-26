#!/usr/bin/env bash
# Evolve the earth-pulse-harness with Darwin Mode using the OpenRouter LLM mutator,
# sourcing the OpenRouter API key from GCP Secret Manager at runtime.
#
# The key is fetched fresh on every run and exported only into this process's
# environment — it is NEVER written to the repo, a dotfile, or the logs.
#
# Usage:
#   ./scripts/evolve-openrouter.sh                 # real sandbox, 2 gens x 3 children
#   GENERATIONS=1 CHILDREN=2 SANDBOX=mock ./scripts/evolve-openrouter.sh
#
# Env overrides:
#   OPENROUTER_SECRET   GCP secret name        (default: OPENROUTER_API_KEY)
#   GCP_PROJECT         GCP project            (default: cognitum-20260110)
#   DARWIN_MUTATOR_MODEL OpenRouter model      (default: google/gemini-2.5-flash)
#   DARWIN_DIST         path to darwin dist    (for monorepo/local runs without npm i)
set -euo pipefail

SECRET="${OPENROUTER_SECRET:-OPENROUTER_API_KEY}"
PROJECT="${GCP_PROJECT:-cognitum-20260110}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "evolve-openrouter: gcloud not found; cannot source the OpenRouter key from GCP." >&2
  exit 1
fi

# Fetch the key from GCP Secret Manager into this process only.
OPENROUTER_API_KEY="$(gcloud secrets versions access latest --secret="$SECRET" --project="$PROJECT")"
export OPENROUTER_API_KEY
export DARWIN_MUTATOR_MODEL="${DARWIN_MUTATOR_MODEL:-google/gemini-2.5-flash}"

HARNESS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec node "$HARNESS_DIR/scripts/evolve-openrouter.mjs" "$HARNESS_DIR"
