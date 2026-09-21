# `@ruvector/typesafe` benchmark harness

Measures the local engine against Jev (typesafe.ai) on the same terms
(ADR-006). Publishing is gated by these numbers, not by "tests pass".

## Run

```bash
node bench/run.mjs --suite tickets --arm both            # jev replay + local
node bench/run.mjs --suite tickets --arm jev             # Jev's frozen scorecard
node bench/run.mjs --suite banking77 --limit 1000 --arm local
node bench/run.mjs --suite tickets --arm both --gate --report-only
```

Flags: `--suite tickets|banking77|clinc150|hwu64|all` · `--arm jev|local|both` ·
`--embedder hash|onnx` · `--shots N` (default 8) · `--zero-shot` ·
`--limit N` · `--out PATH` · `--gate` · `--report-only` ·
`--baseline-receipt PATH`. One receipt JSON is written per run to
`bench/results/` (gitignored).

## Arms

- **jev** — replays `jev-baseline-2026-09-21.json`. **No network.** Only the
  `test` split has per-item rows. The headline is Jev's **gen-0 (baseline)**
  number: its `test_rows.baseline` were scored with the same gen-0 criteria the
  local arm sends, so the comparison is apples-to-apples. Jev's champion used
  mutated criteria whose examples are literal ticket texts (memorisation,
  ADR-004) and is reported as informational only. Jev exposes no `noul`
  probability, so urgent AUROC/Brier are not defined for the jev arm.
- **local** — the ruvector binding (`../index.js`: `Engine`, `version`,
  `backend`). Builds one request per item and times `decideJson`. Supports
  SetFit-style few-shot (train on the train split, N/class) and zero-shot,
  reported separately. When the core is not built the arm reports
  "engine unavailable" rather than crashing.

## Guards (the harness refuses to run on failure)

- **Fixture hashes** — every frozen file is checked against
  `fixtures/HASHES.json` (`scripts/hash-fixtures.mjs`). A drifted fixture is
  not the fixture Jev was measured against.
- **Split disjointness** — train / calibration / validation / transfer / test
  are assigned deterministically from item ids and asserted instance-disjoint
  (`assertDisjoint`, ADR-004). For the single-dataset tickets suite the
  calibration and transfer splits are deterministic holdouts carved from the
  native train split; true cross-domain transfer uses a different `--suite`.
- **Vocabulary disjointness** — generator content words (the retrieval corpus's
  `topics`) must not leak into the gen-0 decision criteria (ADR-006 §4). The one
  reviewed benign overlap — "software", shared between a corpus topic and the
  plain-English `technical` criterion, from two unrelated generators — is
  recorded in `ACCEPTED_OVERLAPS`, so the guard fires on new leaks, not this
  one. A planted corpus-topic word in a criterion is caught.

## Gates (`gates.json`, ADR-006)

`--gate` evaluates the measurable release gates and prints a PASS/FAIL/SKIP
table; strict mode exits non-zero on any FAIL. `--report-only` keeps the exit
code 0 (CI uses this until an ONNX embedder is present — the hash embedder is a
placeholder, not the bge-small target the latency and accuracy gates assume). A
gate whose precondition does not apply to the run (OOS AUROC on tickets;
transfer regression with no `--baseline-receipt`) is SKIP, never a silent pass.

## Datasets

`datasets/{banking77,clinc150,hwu64}.mjs` download from the canonical public
sources into `bench/.cache/` (gitignored), verify a pinned sha256, and convert
to `{id, text, label}` plus a `label → description` criteria map. CLINC150's OOS
slice is the abstain set for the OOS AUROC gate. Nothing is redistributed in the
package; an unreachable source fails loudly with the URL and the suite is marked
`skipped: unavailable`.
