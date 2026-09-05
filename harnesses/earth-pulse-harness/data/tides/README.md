# Tide Data

Real tidal model output / gauge observations go here. **The harness never
fabricates observations** — leave this empty until real data is added.

## Expected format

- Time series near the source region with columns `time, tide_height,
  tide_phase` (phase in radians or hours within the tidal cycle), as
  CSV/Parquet.
- A sidecar manifest noting **tide model or gauge source, datum, units, time
  span, and license**.

Used for the tide-phase baseline (B3) and as a candidate feature. See
`../../docs/research/benchmark-design.md` §5.
