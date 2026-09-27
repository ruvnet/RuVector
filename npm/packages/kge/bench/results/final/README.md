# Final (test-split) results ledger — ADR-007 §2.6

This directory is the record of truth for every **test-split** score the
benchmark campaign publishes. Test is read only by
`ruvector-kge-bench --final`, once per `(dataset, seed)`.

**Only origin's protected branch `kge-final-ledger` counts.** `--final` reads
these files with `git fetch origin kge-final-ledger` + `git show`, never from a
working tree, so editing them locally unlocks nothing. On `main` this directory
holds an empty `ledger.jsonl`; the branch is created in plan M5.

| File | Written | Meaning |
|---|---|---|
| `ledger.jsonl` | by `--final` only, append-only | one JSON line per event: an `intent` before test is scored, a `result` after |
| `selection.json` | once, after M7 HPO, before the first `--final` | `{"datasets": {"<name>": {"config_hash": "…", "hpo_receipts": ["…"]}}}` — the valid-MRR argmax over C1–C8 (seed 100) |
| `SIGNERS` | M5, at the tag | fingerprints allowed to sign `kge-prereg-v1` (one per line, `#` comments) |
| `ERRATA.md` | release-manager | retractions; ledger lines are never deleted |

## Line schema

```json
{"kind":"intent","dataset":"wn18rr","seed":0,"config_hash":"…","tag_sha":"…","checkpoint_sha256":"…","utc":"…"}
{"kind":"result","dataset":"wn18rr","seed":0,"config_hash":"…","tag_sha":"…","checkpoint_sha256":"…","receipt_sha256":"…","bottom_ranks_sha256":"…","random_ranks_sha256":"…","utc":"…"}
```

`tag_sha` is the `kge-prereg-v1` tag object; `checkpoint_sha256` is the sha256 of
the scored tables (`best/weights.bin`); the rank hashes are sha256 over the
per-query ranks as u32 little-endian (tail query at `2i`, head at `2i+1`).

## `--final` refuses unless

- `kge-prereg-v1` exists on origin, `git verify-tag` passes with a signer in
  `SIGNERS` at the tag, and the tag is an ancestor of HEAD;
- the seed is in {0,1,2,3,4} and the run is a completed 500-epoch,
  no-early-stop run of a grid config (C1–C8);
- origin's `selection.json` names this `config_hash` for the dataset;
- origin's ledger has no line for `(dataset, seed)` — except an `intent` with
  no `result` and the same checkpoint sha256, which is retried without a new
  intent.

It pushes the `intent` as a fast-forward before scoring (a rejected push scores
nothing), scores test once, writes the receipt with the Bottom and RANDOM rank
vectors and their sha256, then pushes the `result`.

`--verify-final <receipt>` recomputes the ranks from the exported tables and
compares hashes. It is recorded as `verification`, never `final`, and writes no
ledger line.

## CI

`scripts/ci/kge-ledger-check.sh [BASE_REF]` (Rust binary `kge-ledger-check`)
fails on duplicate `(dataset, seed)` keys, seeds outside {0..4}, a result
without an intent, a second or non-selected `config_hash`, mixed tag SHAs, any
modified or deleted line, and `selection.json` changing after the first intent.
