# Ocean Wave Data

Real ocean-wave reanalysis / observations go here. **The harness never
fabricates observations** — leave this empty until real data is added.

## Expected format

- Time series at (or gridded over) the Gulf of Guinea source region with
  columns `time, sig_wave_height, peak_period, mean_direction` (and optionally
  wind sea / swell partitions), as CSV/Parquet or NetCDF.
- A sidecar manifest noting **source/product name, spatial coverage, units,
  time span, and license**.

These are the primary ocean-forcing features (swell height/direction) for the
benchmark. See `../../docs/research/benchmark-design.md` §5.
