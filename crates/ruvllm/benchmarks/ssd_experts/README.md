# Rust expert paging comparison

The experimental CPU runner uses the Qwen3.5 forward pass from Lattice 0.11.1, the engine pinned by ruvllm. It replaces only the CPU routed expert lookup in an isolated build. The reference materializes every expert as f32. The candidate reads the selected BF16 gate, up and down weights by file offset, converts them to f32, and keeps two experts per layer. Both paths retain the original attention, router, shared expert and sampling kernels.

Run with Python 3.12 or later, Rust 1.93.1, a C compiler, the Unix patch utility and an empty working directory. The default checkpoint download is 2.22 GB. The public checkpoint is a community conversion with eight experts and two selected in each of 24 layers.

```sh
python3 crates/ruvllm/benchmarks/ssd_experts/run_cpu.py /path/to/empty/work
```

Each fresh process generates two continuations of 256 tokens using greedy sampling. The fixed horizon ignores EOS and reports the configured EOS ID. The script compares all four token ID arrays and checks the granted cache slots, eviction count and logical bytes for each miss. Decode throughput uses the 255 intervals between generated tokens. The first pass starts with an empty application cache. Filesystem pages may already be cached. Process attributed storage reads and logical expert bytes are recorded separately.

The macOS job measures the actual Metal device and runs the portable CPU comparison on Apple Silicon. The results apply to the numerical experiment and checkpoint recorded in the report. The benchmark does not measure model quality. Cold physical SSD throughput requires storage residency evidence on the target device. All modifications to the pinned dependency live in the benchmark working directory.

## Native ruvllm on Metal

For an existing native Lattice Q4 checkpoint on Apple Silicon, run the following from a RuVector checkout containing commit `979c31acc57b970c1a7e0771ed527c1703c137e8`. Reserve space for a disposable worktree and release build.

```sh
python3 crates/ruvllm/benchmarks/ssd_experts/run_metal.py "$PWD" /path/to/empty/work --q4 /path/to/native-q4
```

The driver checks the Metal working set before allocating. It measures the live slot count for every layer, primes every expert in the resident run and requires zero later cache misses. The paged run must evict experts. Both runs use the normal ruvllm streaming backend. The observer takes token IDs from the engine's completed generation and verifies that its per-token timing callbacks cover the same number of steps. The paged memory limit is 16 GiB. Host swap counters include activity from other applications.

To compare all 248,320 F32 logits before each greedy selection, run `run_diagnostics.py WORK NEW_OUTPUT` after the CPU benchmark. The diagnostic adds hashing overhead. On Linux it evicts only the private checkpoint from the file cache after the first generated token and verifies the remaining resident page count with `mincore`. The second pass uses the same process and warm filesystem cache.
