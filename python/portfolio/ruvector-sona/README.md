# ruvector-sona

![RuVector sona animated overview](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/sona.svg)

Validated batch inference over the native SONA LoRA hooks.

## Getting started

Requires Python 3.10 or later and a platform supported by the `ruvector==0.1.4` native wheel.

```sh
python -m pip install ruvector-sona==0.2.0
```

```python
from ruvector_sona import LoraTransform
transform = LoraTransform(2)
result = transform.transform([[1, 0], [0, 1]])
assert result.tolist() == [[1, 0], [0, 1]]
```

## How it works

Python sequences → finite contiguous float32 validation → shared RuVector native Rust runtime → typed Python or NumPy results. No Node.js process is used.

## Scope and limits

Fresh engines produce identity transforms. Python does not expose trajectory recording or online learning. Native save_state preserves patterns and flags, not LoRA weights. This package does not claim adaptation quality gains.

This is a Python convenience API backed by the shared native runtime, not complete npm API parity. Invalid dimensions and nonfinite inputs are rejected.

## Documentation

[Portfolio and architecture](https://github.com/ruvnet/RuVector/tree/main/python/portfolio) · [Source](https://github.com/ruvnet/RuVector/tree/main/python/portfolio/ruvector-sona) · [Issues](https://github.com/ruvnet/RuVector/issues)
