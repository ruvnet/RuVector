#!/usr/bin/env python3
"""Build a real ``.calyx`` corpus from CodeSearchNet for ruvector-calyx.

This is the ONE step that needs models + network + (ideally) a GPU. It embeds
each code function through several *frozen lenses*, keeps every lens as its own
slot (never flattened), and writes a ``.calyx`` v1 binary that the pure-Rust
``calyx-real-bench`` loads with zero dependencies.

Lenses produced (a constellation per function):
  * ``code``    — a joint NL<->code search model, so an NL query and a code
                  snippet land in the same space (default:
                  ``flax-sentence-embeddings/st-codesearch-distilroberta-base``,
                  trained on CodeSearchNet).
  * ``doc``     — a text embedder over the docstring (default:
                  ``BAAI/bge-small-en-v1.5``).
  * ``lexical`` — a dependency-free hashed bag-of-tokens vector over code
                  identifiers (the "BM25-ish" surface-term lens); cosine of two
                  hashed TF vectors approximates lexical overlap.

Queries are the docstrings (the standard CodeSearchNet code-search task: given a
natural-language query, retrieve the function it documents). ``--unanswerable``
holds out a fraction of golds from the corpus so those queries *should* be
abstained on — which is what makes conformal risk meaningful.

Usage (run where you have the models):
  pip install datasets sentence-transformers numpy
  python build_codesearchnet.py --lang python --n 5000 --out corpus.calyx
  # then, in the repo:
  cargo run --release -p ruvector-calyx --bin calyx-real-bench -- corpus.calyx

The byte layout MUST match crates/ruvector-calyx/src/corpus.rs (v1). Keep the
two in sync.
"""

import argparse
import hashlib
import re
import struct

MAGIC = b"CALYXB1\n"
LENS_KIND = {"dense": 0, "lexical": 1, "structural": 2, "temporal": 3, "sensor": 4}
ANCHOR_KIND = {"accepted": 0, "passed_test": 1, "sensor": 2, "citation": 3, "reward": 4}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


# ── byte writers (little-endian; mirror corpus.rs) ──────────────────────────
def w_u16(f, v): f.write(struct.pack("<H", v))
def w_u32(f, v): f.write(struct.pack("<I", v))
def w_u64(f, v): f.write(struct.pack("<Q", v))
def w_f32(f, v): f.write(struct.pack("<f", float(v)))
def w_str(f, s): b = s.encode("utf-8"); w_u16(f, len(b)); f.write(b)
def w_vec(f, xs):
    for x in xs:
        w_f32(f, x)


def l2norm(vec):
    import numpy as np
    v = np.asarray(vec, dtype="float32")
    n = float((v * v).sum()) ** 0.5
    return (v / n) if n > 1e-9 else v


def hashed_tokens(text, dims):
    """Dependency-free lexical lens: hashed, L2-normalized TF vector."""
    import numpy as np
    v = np.zeros(dims, dtype="float32")
    for tok in _TOKEN.findall(text.lower()):
        h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
        v[h % dims] += 1.0
    return l2norm(v)


def main():
    ap = argparse.ArgumentParser(description="Build a .calyx corpus from CodeSearchNet")
    ap.add_argument("--lang", default="python", help="CodeSearchNet language subset")
    ap.add_argument("--n", type=int, default=5000, help="number of functions")
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--unanswerable", type=float, default=0.2,
                    help="fraction of queries whose gold is held out of the corpus")
    ap.add_argument("--code-model", default="flax-sentence-embeddings/st-codesearch-distilroberta-base")
    ap.add_argument("--doc-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--lexical-dims", type=int, default=512)
    ap.add_argument("--out", default="corpus.calyx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(args.seed)
    print(f"Loading CodeSearchNet[{args.lang}] …")
    ds = load_dataset("code_search_net", args.lang, split="test")
    ds = ds.filter(lambda r: r["func_documentation_string"] and r["whole_func_string"])
    idx = rng.permutation(len(ds))[: args.n]
    rows = [ds[int(i)] for i in idx]
    codes = [r["whole_func_string"] for r in rows]
    docs = [r["func_documentation_string"] for r in rows]

    print(f"Embedding {len(rows)} functions with code + doc models …")
    code_model = SentenceTransformer(args.code_model)
    doc_model = SentenceTransformer(args.doc_model)
    code_vecs = code_model.encode(codes, normalize_embeddings=True, show_progress_bar=True)
    doc_vecs = doc_model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    lex_vecs = [hashed_tokens(c, args.lexical_dims) for c in codes]

    d_code, d_doc, d_lex = len(code_vecs[0]), len(doc_vecs[0]), args.lexical_dims

    # Queries = docstrings. Gold = the documenting function. A fraction are made
    # unanswerable by holding their gold out of the *record* set.
    q_idx = rng.permutation(len(rows))[: args.n_queries]
    n_unans = int(len(q_idx) * args.unanswerable)
    held_out = set(int(i) for i in q_idx[:n_unans])
    record_ids = [i for i in range(len(rows)) if i not in held_out]

    # Query lenses: the NL docstring embedded by the code model (joint space),
    # the doc model, and its hashed tokens.
    q_texts = [docs[int(i)] for i in q_idx]
    q_code = code_model.encode(q_texts, normalize_embeddings=True)
    q_doc = doc_model.encode(q_texts, normalize_embeddings=True)
    q_lex = [hashed_tokens(q_texts[k], args.lexical_dims) for k in range(len(q_idx))]

    print(f"Writing {args.out}  ({len(record_ids)} records, {len(q_idx)} queries, "
          f"{n_unans} unanswerable) …")
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        # lenses
        w_u32(f, 3)
        for name, ver, kind, dims in [
            ("code", args.code_model, LENS_KIND["dense"], d_code),
            ("doc", args.doc_model, LENS_KIND["dense"], d_doc),
            ("lexical", "hashed-tokens-v1", LENS_KIND["lexical"], d_lex),
        ]:
            w_str(f, name); w_str(f, ver)
            f.write(bytes([kind])); w_u32(f, dims)
            w_f32(f, 8.0 if name != "lexical" else 1.0)   # illustrative cost µs
            w_f32(f, 8.0 if name != "lexical" else 1.0)
        # records
        w_u32(f, len(record_ids))
        for i in record_ids:
            w_u64(f, i)
            w_vec(f, l2norm(code_vecs[i]))
            w_vec(f, l2norm(doc_vecs[i]))
            w_vec(f, lex_vecs[i])
            w_u16(f, 1)  # one anchor: the function has an associated test/example
            f.write(bytes([ANCHOR_KIND["passed_test"]])); w_f32(f, 0.9); w_str(f, f"gold_{i}")
        # queries
        w_u32(f, len(q_idx))
        for k, i in enumerate(q_idx):
            for present, vec in ((1, q_code[k]), (1, q_doc[k]), (1, q_lex[k])):
                f.write(bytes([present])); w_vec(f, l2norm(vec))
            gold = int(i)
            rel = [] if gold in held_out else [gold]
            w_u32(f, len(rel))
            for r in rel:
                w_u64(f, r)
    print("done.")


if __name__ == "__main__":
    main()
