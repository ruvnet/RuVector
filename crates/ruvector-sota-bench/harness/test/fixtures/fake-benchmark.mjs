import { writeFile } from "node:fs/promises";

const flag = (name) => {
  const index = process.argv.indexOf(name);
  return process.argv[index + 1];
};
const efValues = flag("--ef-search").split(",");
const dataset = flag("--dataset");
const scores = efValues.map((ef) => ({
  index: `core-hnsw(m=${flag("--m")},ef=${ef})`,
  dataset,
  recall: { recall_at_10: 0.96 },
  qps: 1000 + Number(ef),
  memory_mb: 40,
  latency: { p99_us: 500 },
  params: {
    ef_search: ef,
    m: flag("--m"),
    ef_construction: flag("--ef-construction"),
    encoded_payload_bytes: "100",
    index_graph_bytes: "50"
  }
}));
await writeFile(flag("--json"), JSON.stringify({ scores }));
