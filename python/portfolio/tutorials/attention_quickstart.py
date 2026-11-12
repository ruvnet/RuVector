"""Synthetic scaled dot product attention using published RuVector packages.

Requires ruvector-attention==0.2.0, ruvector==0.1.3 and numpy==2.2.6.
This validates attention math, not model training or text generation.
"""
from __future__ import annotations

from importlib.metadata import version

import numpy as np
from ruvector_attention import attend


def main() -> None:
    assert version("ruvector-attention") == "0.2.0"
    assert version("ruvector") == "0.1.3"
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    result = attend(query, candidates)

    # Independent float64 reference; subtracting max prevents exp overflow.
    logits = candidates.astype(np.float64) @ query.astype(np.float64) / np.sqrt(2.0)
    reference = np.exp(logits - logits.max())
    reference /= reference.sum()
    np.testing.assert_allclose(result.weights, reference, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.blended, reference @ candidates, rtol=1e-6, atol=1e-7)
    assert result.blended.shape == (2,)
    assert np.isclose(result.weights.sum(), 1.0)
    assert int(np.argmax(result.weights)) == 0

    rejected = 0
    for bad_query in ([1.0], [float("nan"), 0.0]):
        try:
            attend(bad_query, candidates)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid input was accepted")
    assert rejected == 2

    print("ruvector-attention=0.2.0 ruvector=0.1.3")
    print("weights=" + ",".join(f"{weight:.6f}" for weight in result.weights))
    print("blended=" + ",".join(f"{coordinate:.6f}" for coordinate in result.blended))
    print("PASS: native attention matches reference; invalid inputs rejected")


if __name__ == "__main__":
    main()
