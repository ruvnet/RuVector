/** Static help and version text for the `typesafe` CLI. */

// Kept in sync with package.json by the build; read at runtime to avoid drift.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pkg = require('../../package.json') as { version: string };

export const VERSION = pkg.version;

export const HELP = `typesafe ${VERSION} — local typed decisions (choice / score / noul)

Usage: typesafe <command> [options]

Commands:
  decide    Answer a question batch for one state and print the response JSON.
  train     Admit labeled examples for one question and print a TrainReport.
  eval      Score a labeled dataset: accuracy, macro-F1, ECE, Brier, latency.
  optimize  Run a gated optimize campaign (ADR-004) and write receipts.
  bench     Run the benchmark harness (delegates to bench/run.mjs).
  serve     Start a Jev-compatible HTTP server (POST /v1/systemone, GET /healthz).

Global:
  --version, -v   Print version.
  --help, -h      Print this help (or 'typesafe <command> --help').

decide:
  --state <text>            The message to decide on.
  --state-file <path>       Read the state from a file instead.
  (no --state* flag reads the state from stdin.)
  --questions <path>        JSON map of Jev-shaped questions (required).
  --embedder hash|onnx      Embedder to use (default: hash, a test double).
  --model-dir <path>        ONNX model directory (with --embedder onnx).
  --manifest <path>         ONNX model manifest (with --embedder onnx).
  --jev                     Emit Jev-shape-only answers (strip additive fields).

train:
  --question <id>           Question id the examples label (required).
  --examples <path>         JSONL of {"text","label"} (required).
  --bank <path>             Load (if present) and persist the example bank here.

optimize:
  --questions <path>        The question definition (a QuestionWire or a map).
  --question <id>           Which question, when --questions is a map.
  --dataset <path>          JSONL of {"text","label","split"?}; split inferred if absent.
  --bank <path>             Also/instead take rows from an exported bank JSON.
  --receipts <path>         Write the hash-chained receipts as JSONL.
  --out <path>              Write the full CampaignReport JSON.
  --budget <n>              Per-day evaluation budget (default 64).
  --embedder hash|onnx      Embedder to use (default: hash).

eval:
  --questions <path>        JSON map of questions, or omit to use a bundled set.
  --dataset <path>          Labeled data: a decisions JSON or a JSONL of items.
  --split train|val|test    Split to score from a decisions JSON (default: test).
  --embedder hash|onnx      Embedder to use (default: hash).

serve:
  --port <n>                Port (default 8787; 0 picks a free port).
  --host <addr>             Bind address (default 127.0.0.1).
  --jev                     Emit Jev-shape-only answers.

The engine makes no network requests and spawns no processes. The default
'hash' embedder is a deterministic test double; its answers report
calibrated:false. Production accuracy needs the onnx embedder (ADR-002).
`;
