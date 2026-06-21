# ADR-259: ruvllm as Local Mutator Backend for Darwin Mode

**Status:** Implemented (code + unit tests + CLI; live-serve e2e blocked by a ruvllm download bug — see Implementation status)  

## Implementation status (2026-06-18)

Implemented in `agent-harness-generator` (`@metaharness/darwin`):
- `src/ruvllm-mutator.ts` — `RuvllmMutator implements CodeGenerator`, OpenAI-compatible
  `POST /v1/chat/completions`, safe no-op fallback if the server is unreachable. Exported from `index.ts`.
- `src/cli.ts` — `--mutator deterministic|ruvllm` (+ `--ruvllm-url`/`--ruvllm-model`), wired to `config.generator`.
- `__tests__/ruvllm-mutator.test.ts` — 4 tests vs a real `node:http` mock (success, fence-strip,
  unreachable→no-op, malformed→no-op). Full darwin suite **354/354** green.

**Honest gap:** the live-serve e2e (evolve against a real local model) is **blocked by a ruvllm
2.1.0 download bug** — `ruvllm download phi` fails on `tokenizer_config.json` (the file is served
via an HTTP 307 redirect that ruvllm does not follow; `curl -L` fetches it fine). So the HTTP
*contract* is verified by unit tests, but an end-to-end run against a served model is pending a
ruvllm fix (or manual model placement). Recommend fixing the redirect-follow in `ruvllm download`.

**Value note (ADR-087):** the mutator is not the quality lever — deterministic and frontier-LLM
mutators both hit the 0.985 scorer ceiling. RuvllmMutator's benefit is *operational* (fully local,
air-gapped, zero API cost), not higher scores.

---

**Original status:** Proposed  
**Date:** 2026-06-18  
**Components:**
- `crates/ruvllm` — local LLM inference runtime (RDT/OpenMythos/GGUF)
- `npm/packages/ruvllm-cli` — HTTP serving layer (`ruvllm serve`)
- `packages/darwin-mode` (agent-harness-generator) — harness evolution loop

---

## Context

Darwin Mode (`@metaharness/darwin`, ADR-070…075) evolves an agent harness by mutating
one of seven approved surface files per generation, sandboxing each child variant, and
archiving scored descendants. The mutation step currently calls **OpenRouter**
(`google/gemini-2.5-flash` default) via HTTPS — a zero-setup path that works out of the
box but introduces three constraints:

1. **API key required at runtime.** Every `metaharness-darwin evolve` invocation needs
   `OPENROUTER_API_KEY`, blocking air-gapped, cost-sensitive, or latency-sensitive use.
2. **Network round-trip per mutation.** Each `generateMutation` call crosses the public
   internet: 200–800 ms median, unbounded tail latency.
3. **Token cost scales with evolution depth.** A 5-generation × 4-child run with 64 k
   population history produces ~O(1,000) OpenRouter calls. At current pricing that is
   $0.15–$0.30 per sweep; monthly sweeps across many repos add up.

Darwin Mode already defines a pluggable `CodeGenerator` interface in `mutator.ts` and the
`OpenRouterMutator` slots in behind the same `validateGeneratedCode` safety gate. A
**local ruvllm mutator** simply implements the same interface against the `ruvllm serve`
HTTP endpoint instead of `openrouter.ai`.

---

## Decision

Implement a `RuvllmMutator` class (new file `ruvllm-mutator.ts` in darwin-mode) that:

1. Implements the existing `CodeGenerator` interface — **zero changes** to the evolution
   loop, sandbox, scorer, or archive.
2. Targets `ruvllm serve --model <gguf> --port <N>` which exposes an OpenAI-compatible
   `POST /v1/chat/completions` endpoint (already implemented in `ruvllm-cli`).
3. Is activated via a new `--mutator ruvllm` CLI flag and `--ruvllm-url` / `--ruvllm-model`
   options, defaulting to `http://localhost:8080`.
4. Falls back gracefully (returns parent code unchanged) if the server is unreachable —
   same safe no-op contract as `OpenRouterMutator`.

The ruvllm server is started **externally** by the user; `darwin-mode` does not manage its
lifecycle, consistent with the "dependency-free" constraint on the darwin-mode package.

---

## Integration Architecture

```
┌───────────────────────────────────────┐
│  metaharness-darwin evolve <repo>     │
│                                       │
│  cli.ts                               │
│    └── evolve.ts (generation loop)    │
│          └── createChildVariant()     │
│                └── CodeGenerator      │
│                      ├── OpenRouterMutator  ← existing (default)
│                      └── RuvllmMutator      ← NEW (--mutator ruvllm)
└──────────────────┬────────────────────┘
                   │  POST /v1/chat/completions
                   │  { model, messages, max_tokens, temperature }
                   ▼
┌───────────────────────────────────────┐
│  ruvllm serve --model model.gguf      │
│               --port 8080             │
│  (ruvllm-cli / ruvllm-linux-x64-gnu)  │
│                                       │
│  Backends: Candle (CPU/CUDA/Metal),   │
│  GGUF GGML, RDT/OpenMythos            │
└───────────────────────────────────────┘
```

### New file: `packages/darwin-mode/src/ruvllm-mutator.ts`

```typescript
// SPDX-License-Identifier: MIT
// RuvllmMutator — local ruvllm server backend for Darwin Mode (ADR-259).
// Implements CodeGenerator against POST /v1/chat/completions (OpenAI-compatible).
// Zero runtime dependencies; uses Node fetch (built-in ≥ 18).

import type { CodeGenerator } from './mutator.js';
import type { MutationSurface } from './types.js';

export interface RuvllmMutatorOptions {
  /** Base URL of the ruvllm serve endpoint. Default: http://localhost:8080 */
  baseUrl?: string;
  /** Model name passed in the request body. Default: 'local' */
  model?: string;
  maxTokens?: number;
  temperature?: number;
  /** Request timeout in ms. Default: 30_000 */
  timeoutMs?: number;
}

export class RuvllmMutator implements CodeGenerator {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly maxTokens: number;
  private readonly temperature: number;
  private readonly timeoutMs: number;

  constructor(opts: RuvllmMutatorOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? process.env.RUVLLM_URL ?? 'http://localhost:8080')
      .replace(/\/$/, '');
    this.model   = opts.model ?? process.env.RUVLLM_MODEL ?? 'local';
    this.maxTokens   = opts.maxTokens   ?? 2000;
    this.temperature = opts.temperature ?? 0.4;
    this.timeoutMs   = opts.timeoutMs   ?? 30_000;
  }

  async generateMutation(input: {
    parentCode: string;
    surface: MutationSurface;
    repoSummary: string;
    parentScore: number;
    failedTraces: string[];
    nonce?: number;
  }): Promise<{ code: string; summary: string }> {
    const sys =
      'You improve ONE file of an AI agent harness. Output ONLY the full replacement file — ' +
      'no prose, no fences. HARD RULES: keep every exported name and signature identical; ' +
      'introduce NO new capabilities, imports, network, filesystem, shell, or env access; ' +
      'no new dependencies; pure refactor/tuning only. Make a small, plausibly score-improving ' +
      'change to the "' + input.surface + '" surface.';
    const user =
      `Surface: ${input.surface}\nParent score: ${input.parentScore}\n` +
      (input.repoSummary ? `Repo: ${input.repoSummary}\n` : '') +
      (input.failedTraces.length
        ? `Recent failures:\n${input.failedTraces.slice(0, 5).join('\n')}\n`
        : '') +
      `\n--- current file ---\n${input.parentCode}\n--- end ---\n` +
      'Return the improved full file.';

    let res: Response;
    try {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), this.timeoutMs);
      res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model:       this.model,
          messages:    [{ role: 'system', content: sys }, { role: 'user', content: user }],
          max_tokens:  this.maxTokens,
          temperature: this.temperature,
        }),
        signal: controller.signal,
      });
      clearTimeout(tid);
    } catch (e) {
      return {
        code:    input.parentCode,
        summary: `ruvllm:${this.baseUrl} unreachable (${(e as Error).message}) — no-op`,
      };
    }

    const j: any = await res.json();
    if (!j.choices?.[0]?.message?.content) {
      return { code: input.parentCode, summary: `ruvllm:${this.model} no content — no-op` };
    }
    const raw: string = j.choices[0].message.content;
    const code = unfence(raw);
    return { code, summary: `ruvllm:${this.model} regenerated ${input.surface}` };
  }
}

function unfence(text: string): string {
  const m = text.match(/```(?:[a-zA-Z0-9]+)?\n([\s\S]*?)\n```/);
  return (m ? m[1] : text).trim() + '\n';
}
```

### CLI changes: `packages/darwin-mode/src/cli.ts`

Add flags to the `evolve` command (additive, all opt-in):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mutator` | `openrouter\|ruvllm\|deterministic` | `openrouter` | Code generator backend |
| `--ruvllm-url` | string | `http://localhost:8080` | Base URL for `ruvllm serve` |
| `--ruvllm-model` | string | `local` | Model name in request body |
| `--ruvllm-timeout` | number | `30000` | Request timeout (ms) |

Wire-up in `cli.ts`:

```typescript
import { RuvllmMutator } from './ruvllm-mutator.js';

// Inside evolve command setup:
const gen: CodeGenerator = (() => {
  switch (opts.mutator) {
    case 'ruvllm':
      return new RuvllmMutator({
        baseUrl:     opts.ruvllmUrl,
        model:       opts.ruvllmModel,
        timeoutMs:   opts.ruvllmTimeout,
        temperature: 0.4,
      });
    case 'deterministic':
      return undefined; // use built-in DeterministicMutator
    default: // 'openrouter'
      return new OpenRouterMutator({ model: opts.model });
  }
})();
```

---

## Usage

### 1. Start ruvllm server (once, in a separate terminal)

```bash
# With a GGUF model (CPU):
ruvllm serve --model ~/.cache/models/codellama-7b-q4.gguf --port 8080

# With CUDA acceleration (RDT/OpenMythos — see ADR-258):
ruvllm serve --model ~/.cache/models/openmythos-512.gguf --port 8080 --backend cuda
```

### 2. Run Darwin Mode evolution against the local server

```bash
metaharness-darwin evolve /path/to/my-agent-repo \
  --mutator ruvllm \
  --ruvllm-url http://localhost:8080 \
  --generations 5 \
  --children 4 \
  --selection quality-diversity
```

### 3. Environment-variable alternative (no flags)

```bash
export RUVLLM_URL=http://localhost:8080
export RUVLLM_MODEL=local
metaharness-darwin evolve . --mutator ruvllm
```

---

## Consequences

### Benefits

| Dimension | OpenRouter (current) | ruvllm (proposed) |
|-----------|---------------------|-------------------|
| API key | Required | Not required |
| Network | Public HTTPS | localhost |
| Latency per call | 200–800 ms | 50–300 ms (GPU) |
| Cost per sweep | $0.15–$0.30 | $0 (power only) |
| Air-gap support | No | Yes |
| Model control | Platform-defined | User-controlled (any GGUF) |
| Code quality | `gemini-2.5-flash` | Depends on chosen model |

### Trade-offs / Limitations

- **Model quality is user responsibility.** A weak quantized model will produce
  lower-quality mutations, potentially more no-ops (safety gate rejections), and slower
  convergence. Recommended minimum: a 7B parameter code-capable model (CodeLlama,
  DeepSeek-Coder, or a fine-tuned derivative) at Q4_K_M or higher.
- **Server lifecycle is external.** Darwin Mode does not start or stop `ruvllm serve`.
  Users must manage the server process. A health-check probe (`GET /health` or a minimal
  completion) before the first generation would surface down-server failures early —
  **tracked as a follow-up improvement** rather than a blocker.
- **Context window matters.** The mutation prompt includes the full surface file (up to
  ~300 lines for complex surfaces). Models with < 4096 token context may truncate. Use a
  model with ≥ 8192 context for reliable results.
- **OpenAI-compatible subset only.** `ruvllm serve` must implement at minimum:
  `POST /v1/chat/completions` → `{ choices: [{ message: { content: string } }] }`.
  The existing `ruvllm-cli` already provides this (see `bin/ruvllm.js:86`).

### No changes to darwin-mode core invariants

- The `validateGeneratedCode` safety gate is unchanged — `RuvllmMutator` output passes
  through the same hard rules as OpenRouter output.
- The deterministic seed path (`--mutator deterministic`) remains the reproducibility
  baseline; `--mutator ruvllm` is explicitly non-deterministic (LLM sampling).
- Archive format, scoring, selection strategies, and sandbox are unaffected.

---

## Alternatives Considered

**A. Embed ruvllm as a Node.js native module directly in darwin-mode.**  
Rejected. darwin-mode's core constraint is "Node built-ins only, no runtime dependencies."
Embedding the NAPI ruvllm binary would violate this constraint and balloon the package
size. The HTTP interface preserves the dependency-free guarantee for the core package.

**B. Use `child_process.spawn` to call the `ruvllm` CLI directly.**  
Rejected. Spawning a process per mutation call would have high startup overhead
(model load time ≈ 1–5 s per invocation vs. a persistent server). The server-per-session
pattern already used by OpenRouter is the right model.

**C. OpenAI SDK proxy (point `OPENAI_BASE_URL` at ruvllm serve).**  
Partially viable. The `OpenRouterMutator` could be reused with `OPENAI_BASE_URL=http://localhost:8080`.
However, this would surface an implicit dependency on OpenAI SDK conventions (the
`Authorization: Bearer` header, which ruvllm may or may not enforce). The dedicated
`RuvllmMutator` is explicit, adds a timeout, drops the auth header, and is documented —
preferable for clarity and future configurability.

---

## Implementation Plan

1. **`ruvllm-mutator.ts`** — implement as shown above (~80 lines). No dependencies.
2. **`cli.ts`** — add `--mutator`, `--ruvllm-url`, `--ruvllm-model`, `--ruvllm-timeout`
   flags and wire the factory function.
3. **`__tests__/ruvllm-mutator.test.ts`** — unit tests with a mock HTTP server using
   `node:http`. Test: success, unreachable server (no-op), malformed response (no-op),
   fenced code stripping.
4. **README update** — add `--mutator ruvllm` section to the quick-start table.
5. **ruvllm-cli verification** — confirm `ruvllm serve` returns well-formed
   `{ choices: [{ message: { content } }] }` JSON for a simple prompt. If not, add the
   endpoint in a follow-up to `ruvllm-cli`.
