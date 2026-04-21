# TurboQuant Vector Search Benchmark Results

Date: 2026-04-19 19:01:49
Platform: darwin
Python: 3.14.4
PyTorch: 2.10.0

## Results

| Dataset | Method | Dims | N | Bits/dim | Compress | R@1 | R@10 | R@100 | p50 ms | Trials | Memory MB |
|---------|--------|------|---|----------|----------|-----|------|-------|--------|--------|-----------|
| sift1m | f32_baseline | 128 | 100,000 | 32.0 | 1.0x | 1.000 | 1.000 | 1.000 | 1.47±0.02 | 3 | 51.2 |
| sift1m | scalar_int8 | 128 | 100,000 | 8.0 | 4.0x | 0.986 | 0.989 | 0.993 | 1.48 | 3 | 13.6 |
| sift1m | int4 | 128 | 100,000 | 4.0 | 8.0x | 0.750 | 0.850 | 0.902 | 1.48 | 3 | 13.6 |
| sift1m | binary | 128 | 100,000 | 1.0 | 32.0x | 0.000 | 0.000 | 0.003 | 1.44±0.03 | 3 | 1.6 |
| sift1m | product_quant_8sub | 128 | 100,000 | 0.5 | 64.0x | 0.081 | 0.189 | 0.338 | 1.49±0.02 | 3 | 0.9 |
| sift1m | turboquant_mse_3bit | 128 | 100,000 | 3.0 | 10.7x | 0.283±0.018 | 0.411±0.015 | 0.548±0.019 | 1.47 | 3 | 4.8 |
| sift1m | turboquant_mse_4bit | 128 | 100,000 | 4.0 | 8.0x | 0.448±0.032 | 0.577±0.036 | 0.691±0.035 | 1.51±0.07 | 3 | 6.4 |
| sift1m | turboquant_full_3bit | 128 | 100,000 | 3.0 | 10.7x | 0.168±0.012 | 0.249±0.007 | 0.377±0.003 | 13.94±0.16 | 3 | 5.0 |
| glove200 | f32_baseline | 200 | 100,000 | 32.0 | 1.0x | 1.000 | 1.000 | 1.000 | 2.67 | 3 | 80.0 |
| glove200 | scalar_int8 | 200 | 100,000 | 8.0 | 4.0x | 0.997 | 0.993 | 0.994 | 2.71±0.03 | 3 | 20.8 |
| glove200 | int4 | 200 | 100,000 | 4.0 | 8.0x | 0.912 | 0.904 | 0.917 | 2.72±0.02 | 3 | 20.8 |
| glove200 | binary | 200 | 100,000 | 1.0 | 32.0x | 0.514 | 0.503 | 0.498 | 2.77±0.01 | 3 | 2.5 |
| glove200 | product_quant_8sub | 200 | 100,000 | 0.3 | 100.0x | 0.182 | 0.205 | 0.269 | 2.74±0.02 | 3 | 1.0 |
| glove200 | turboquant_mse_3bit | 200 | 100,000 | 3.0 | 10.7x | 0.820±0.007 | 0.826±0.003 | 0.845 | 2.79±0.05 | 3 | 7.5 |
| glove200 | turboquant_mse_4bit | 200 | 100,000 | 4.0 | 8.0x | 0.896 | 0.903±0.003 | 0.917 | 2.76 | 3 | 10.0 |
| glove200 | turboquant_full_3bit | 200 | 100,000 | 3.0 | 10.7x | 0.661±0.005 | 0.680 | 0.685 | 26.52±0.48 | 3 | 7.7 |
| pkm384 | f32_baseline | 384 | 117 | 32.0 | 1.0x | 1.000 | 1.000 | 1.000 | 0.01 | 3 | 0.2 |
| pkm384 | scalar_int8 | 384 | 117 | 8.0 | 4.0x | 0.950 | 0.990 | 1.000 | 0.01 | 3 | 0.0 |
| pkm384 | int4 | 384 | 117 | 4.0 | 8.0x | 0.900 | 0.960 | 0.991 | 0.01 | 3 | 0.0 |
| pkm384 | binary | 384 | 117 | 1.0 | 32.0x | 0.800 | 0.805 | 0.963 | 0.01 | 3 | 0.0 |
| pkm384 | product_quant_8sub | 384 | 117 | 0.2 | 192.0x | 1.000 | 1.000 | 1.000 | 0.01 | 3 | 0.2 |
| pkm384 | turboquant_mse_3bit | 384 | 117 | 3.0 | 10.7x | 0.900 | 0.932±0.006 | 0.989 | 0.01 | 3 | 0.0 |
| pkm384 | turboquant_mse_4bit | 384 | 117 | 4.0 | 8.0x | 0.900 | 0.955±0.008 | 0.994±0.001 | 0.01 | 3 | 0.0 |
| pkm384 | turboquant_full_3bit | 384 | 117 | 3.0 | 10.7x | 0.817±0.085 | 0.880±0.004 | 0.979±0.003 | 0.39 | 3 | 0.0 |

## Notes

- All searches are brute-force (no HNSW acceleration) to isolate quantization quality.
- Vectors are L2-normalized before quantization (inner product = cosine similarity).
- All methods pre-reconstruct to f32 before search (fair latency comparison).
- TurboQuant full uses the unbiased inner product estimator from the paper.
- Ground truth computed with exact f32 inner product.
- Each configuration run 3x with different seeds (stochastic methods) or repeated (deterministic).
- ±values show standard deviation across trials.
