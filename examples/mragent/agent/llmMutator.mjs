// GPU LLM write-layer for the MRAgent Darwin optimizer (ADR-260: "the real
// Darwin write-layer proposes leaps directly from failure traces").
//
// The built-in GA explores the genome by RANDOM mutation. This adds the missing
// directed-proposal layer: it shows a local, GPU-served code model (e.g.
// qwen2.5-coder on an OpenAI-compatible endpoint) the current genome + the tasks
// it is FAILING, and asks for an improved genome. Proposals are clamped to the
// declared gene bounds before they ever enter the population, so a bad LLM
// output can only ever be a no-op — never an unsafe gene.
//
// Fully opt-in + gracefully degrading (ADR-150): if no endpoint answers, the
// optimizer runs exactly as before (deterministic GA + coordinate-descent).
//
// Endpoint: OpenAI-compatible POST {url}/chat/completions.
//   MRAGENT_LLM_URL    (default http://localhost:11434/v1  — ollama on the GPU)
//   MRAGENT_LLM_MODEL  (default qwen2.5-coder:7b)

// Declared gene bounds — MUST stay in sync with agent/harness.mjs mutate().
const BOUNDS = {
  cueK: [1, 12, "int"],
  efSearch: [16, 256, "int"],
  hybridAlpha: [0, 1, "float"],
  traversalDepth: [1, 4, "int"],
  tagFanout: [1, 8, "int"],
  pruneThreshold: [0, 0.6, "float"],
  maxContent: [1, 20, "int"],
  haltConfidence: [0.2, 0.9, "float"],
  abstainThreshold: [0, 0.6, "float"],
};
const ENUMS = {
  fusion: ["rrf", "linear", "dbsf"],
  rerank: ["gnn", "none"],
  promptStrategy: ["terse", "evidence-first", "prune-explicit"],
};

/** Clamp/validate an arbitrary object into a safe genome, based on `baseline`. */
export function coerceGenome(obj, baseline) {
  const g = { ...baseline };
  if (!obj || typeof obj !== "object") return g;
  for (const [k, [lo, hi, kind]] of Object.entries(BOUNDS)) {
    const v = Number(obj[k]);
    if (Number.isFinite(v)) {
      const c = Math.max(lo, Math.min(hi, v));
      g[k] = kind === "int" ? Math.round(c) : c;
    }
  }
  for (const [k, opts] of Object.entries(ENUMS)) {
    if (typeof obj[k] === "string" && opts.includes(obj[k])) g[k] = obj[k];
  }
  return g;
}

function extractJson(text) {
  if (!text) return null;
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : text;
  const start = body.indexOf("{");
  const end = body.lastIndexOf("}");
  if (start === -1 || end <= start) return null;
  try {
    return JSON.parse(body.slice(start, end + 1));
  } catch {
    return null;
  }
}

/** Returns `{ url, model }` if a local LLM endpoint answers, else `null`. */
export async function detectEndpoint(timeoutMs = 2500) {
  const url = (process.env.MRAGENT_LLM_URL || "http://localhost:11434/v1").replace(/\/$/, "");
  const model = process.env.MRAGENT_LLM_MODEL || "qwen2.5-coder:7b";
  try {
    const ctrl = AbortSignal.timeout(timeoutMs);
    const r = await fetch(`${url}/models`, { signal: ctrl });
    if (!r.ok) return null;
    return { url, model };
  } catch {
    return null;
  }
}

/**
 * Ask the GPU model for `n` improved genomes given the current best + failure
 * traces. Each proposal is coerced into bounds. Returns `[]` on any failure.
 */
export async function llmProposeGenomes({ url, model, baseline, current, failures, n = 2, timeoutMs = 60000 }) {
  const genes =
    "cueK[1..12 int], efSearch[16..256 int], hybridAlpha[0..1], fusion(rrf|linear|dbsf), " +
    "traversalDepth[1..4 int], tagFanout[1..8 int], pruneThreshold[0..0.6], maxContent[1..20 int], " +
    "haltConfidence[0.2..0.9], rerank(gnn|none), promptStrategy(terse|evidence-first|prune-explicit), " +
    "abstainThreshold[0..0.6]";
  const sys =
    "You tune a graph-memory retrieval harness (cue search -> bounded traversal -> synthesis). " +
    "Goal: raise accuracy AND risk-adjusted utility (abstain on weak evidence; never confidently hallucinate) " +
    "while keeping traversal cheap. Reason briefly, then output ONLY a JSON array of genome objects.";
  const user =
    `Current genome:\n${JSON.stringify(current)}\n\n` +
    `Failing cases (id, why):\n${failures}\n\n` +
    `Genes and ranges: ${genes}\n\n` +
    `Propose ${n} distinct improved genomes as a JSON array. JSON only.`;

  let res;
  try {
    res = await fetch(`${url}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: sys },
          { role: "user", content: user },
        ],
        temperature: 0.5,
        max_tokens: 700,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch {
    return [];
  }
  if (!res.ok) return [];
  let content;
  try {
    content = (await res.json()).choices?.[0]?.message?.content ?? "";
  } catch {
    return [];
  }
  // Accept either a JSON array or a single object.
  const parsed = extractArray(content);
  return parsed.slice(0, n).map((o) => coerceGenome(o, baseline));
}

function extractArray(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : text;
  const a = body.indexOf("[");
  const b = body.lastIndexOf("]");
  if (a !== -1 && b > a) {
    try {
      const arr = JSON.parse(body.slice(a, b + 1));
      if (Array.isArray(arr)) return arr;
    } catch {
      /* fall through to single-object */
    }
  }
  const one = extractJson(text);
  return one ? [one] : [];
}
