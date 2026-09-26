# Seismic Data

Real broadband / long-period seismic observations go here. **The harness never
fabricates observations** — leave this empty until real data is added.

## Expected format

- Per-station waveform files (e.g. miniSEED / SAC), OR
- Precomputed time series at the 26 s line: `amplitude(t)` and cross-station
  `coherence(t)`, as CSV/Parquet with columns `time, station, amplitude,
  coherence`.
- A sidecar manifest noting **network/station codes, instrument response, sample
  rate, time span, source, and license**.

This is the prediction **target** (pulse amplitude) and the signal-quality
input (coherence) for the benchmark. See
`../../docs/research/benchmark-design.md` §5.
