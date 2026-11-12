# Native attention quickstart

Use published `ruvector-attention==0.2.0` to blend synthetic embeddings with scaled dot product attention. This example calls the RuVector Rust runtime and checks its output against an independent NumPy reference.

## Install and run

Python 3.10 or later; the native `ruvector==0.1.3` wheel must support your platform. Save this README and `attention_quickstart.py` together.

```sh
uv venv .venv
uv pip install --python .venv/bin/python ruvector-attention==0.2.0 ruvector==0.1.3 numpy==2.2.6
.venv/bin/python attention_quickstart.py
```

On Windows use `.venv/Scripts/python.exe` in the two commands above. The exact script is [attention_quickstart.py](attention_quickstart.py).

Expected output:

```text
ruvector-attention=0.2.0 ruvector=0.1.3
weights=0.575975,0.283995,0.140029
blended=0.435946,0.283995
PASS: native attention matches reference; invalid inputs rejected
```

## What the example proves

The three candidates act as both keys and values. The runtime computes `softmax(query @ candidates.T / sqrt(dimension))` and their weighted sum. The first candidate receives the greatest weight. The example also verifies rejection of a dimension mismatch and a nonfinite query.

Fresh PyPI installation and the exact script passed on October 12, 2026, using CPython 3.10.22, Linux x86_64, glibc 2.39 and NumPy 2.2.6. This is a numerical correctness smoke test, with no speed or quality benchmark. This run does not establish fresh macOS or Windows execution.

## Scope and limitations

There is no model download, text generation, trained cross encoder or model training. Attention ordering is dot product ordering. The package exposes this attention operation, not full npm feature parity. The wrapper converts inputs to contiguous float32; conversion can allocate before shape validation. Use external resource limits for untrusted workloads.

The published 0.2.0 wheel requires `ruvector==0.1.3`. Current repository main changes that dependency to 0.1.4. This tutorial deliberately reproduces the immutable published artifact and does not install unreleased source.

## Sources and release recovery

[PyPI package](https://pypi.org/project/ruvector-attention/0.2.0/) · [Native runtime](https://pypi.org/project/ruvector/0.1.3/) · [Immutable source](https://github.com/ruvnet/RuVector/tree/fccf36e9a76d35bce6750dfd3b25802cae3d683b/python/portfolio/ruvector-attention) · [Portfolio PR #1188](https://github.com/ruvnet/RuVector/pull/1188) · [Recovery ADR](../../../docs/python-portfolio/resume-release-adr.md)

The portfolio release is partially delivered. Recovery checks every existing filename and checksum against the original artifact ledger, then permits only missing original files through protected publishing. That does not change the successfully published attention wheel used here.
