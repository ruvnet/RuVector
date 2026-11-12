// First-party fixture standing in for the AI4AI-Bench (arXiv:2608.20318)
// evaluator. Reads --task/--mutation, writes an evaluation JSON to --output.
// The mutation description steers behavior so tests can exercise the
// adapter's fail-loud paths: "crash" exits non-zero, "tuning" reports a
// tuning-only submission, anything else reports a genuine algorithm change
// whose score grows with the seed (deterministic, bounded).
import { readFile, writeFile } from "node:fs/promises";

const flag = (name) => {
  const index = process.argv.indexOf(name);
  return process.argv[index + 1];
};
const mutation = JSON.parse(await readFile(flag("--mutation"), "utf8"));
if (mutation.description.includes("crash")) {
  process.stderr.write("fixture evaluator: deliberate crash\n");
  process.exit(3);
}
const tuningOnly = mutation.description.includes("tuning");
const score = Math.min(0.9, 0.1 + (mutation.seed % 10) * 0.02);
await writeFile(
  flag("--output"),
  JSON.stringify({ score, algorithm_changed: !tuningOnly }),
);
