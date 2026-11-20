# Rust expert paging comparison

The experimental CPU runner uses the Qwen3.5 forward pass from Lattice 0.11.1, the engine pinned by ruvllm. It replaces only the CPU routed expert lookup in an isolated build. The reference materializes every expert as f32. The candidate reads the selected BF16 gate, up and down weights by file offset, converts them to f32, and keeps two experts per layer. Both paths retain the original attention, router, shared expert and sampling kernels.

Run with Python 3.12 or later, Rust 1.93.1, a C compiler, the Unix patch utility and an empty working directory. The default checkpoint download is 2.22 GB. The public checkpoint is a community conversion with eight experts and two selected in each of 24 layers.

```sh
python3 crates/ruvllm/benchmarks/ssd_experts/run_cpu.py /path/to/empty/work
```

Each fresh process generates two continuations of 256 tokens using greedy sampling. The fixed horizon ignores EOS and reports the configured EOS ID. The script compares all four token ID arrays and checks the granted cache slots, eviction count and logical bytes for each miss. Decode throughput uses the 255 intervals between generated tokens. The first pass starts with an empty application cache. Filesystem pages may already be cached. Process attributed storage reads and logical expert bytes are recorded separately.

The macOS job measures the actual Metal device and runs the portable CPU comparison on Apple Silicon. The results apply to the numerical experiment and checkpoint recorded in the report. The benchmark does not measure model quality. Cold physical SSD throughput requires storage residency evidence on the target device. All modifications to the pinned dependency live in the benchmark working directory.
