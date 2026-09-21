/** Static help and version text for the `kge` CLI. */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const pkg = require('../../package.json') as { version: string };

export const VERSION = pkg.version;

export const HELP = `kge ${VERSION} — holographic knowledge-graph embeddings (HolE / RotatE)

Usage: kge <command> [options]

Commands:
  import    Build a model from a triples file and save it.
  train     Train the tables of a saved model.
  eval      Score a split: filtered MRR / Hits@k.
  predict   Rank candidates for an open slot of a triple.
  compose   Rank tails for a composed relation r1 ∘ r2 (RotatE only).
  similar   Rank relations by cosine similarity.
  optimize  Run a self-optimization campaign.
  bench     Run the benchmark harness (delegates to bench/run.mjs).
  serve     Start an HTTP server (POST /v1/predict, /v1/compose; GET /healthz).

Global:
  --version, -v   Print version.
  --help, -h      Print this help.
  --model <path>  Load a saved model (import/train create one instead).
  --scorer hole|rotate   Scorer for a fresh model (default: hole).
  --dims <n>      Embedding width, even (default: 256).
  --seed <n>      Init seed (default: 42).

import:
  --triples <path.jsonl>   One {"s","r","o"} per line (required).
  --out <path>             Where to write the model envelope (required).

train:   --model <path> [--config <path.json>] [--out <path>]  (saves the trained model)
eval:    --model <path> [--split test]
predict: --model <path> (--s <label> | --o <label>) --r <label> [-k 10]
compose: --model <path> --r1 <label> --r2 <label> --s <label> [-k 10]
similar: --model <path> --r <label> [-k 10]
optimize:--model <path> --budget <n> [--campaign <path.json>] [--receipts <path.jsonl>] [--out <path>]
serve:   --model <path> [--port 8788] [--host 127.0.0.1]

The engine makes no network requests and spawns no processes. Every command —
import / train / eval / predict / compose / similar / optimize and the ANN
index — runs locally end to end.
`;
