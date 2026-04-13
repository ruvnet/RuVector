# Therapeutic Response Simulation: Detection + Entrainment

**Command:** `cargo run --release -p seizure-therapeutic-sim`

---

## The Metronome Hypothesis — Tested

Two identical 16-channel EEG simulations. One gets no intervention. One gets alpha-frequency entrainment starting at the detection boundary (second 315).

```
================================================================
  | Metric              | Control   | Intervention| Change    |
  |---------------------|-----------|-------------|-----------|
  | Seizure onset       | 360s      | 420s        | +60s      |
  | Alpha at onset      | 0.030     | 0.105       | +252%     |
  | Gamma at onset      | 0.110     | 0.041       | -62%      |
  | Total warning time  | 45s       | 115s        | +155%     |
================================================================
```

The entrainment:
- **Partially restored alpha rhythm** (+252% — from 3% of baseline back to 10.5%)
- **Reduced gamma hyperexcitability** (-62% — from 5.3x increase down to 2x)
- **Delayed seizure onset by 60 seconds** (from 360s to 420s)
- **More than doubled the total warning window** (from 45s to 115s)

The brain found its rhythm again before the song broke. The entrainment didn't fully prevent the seizure in this parameter regime, but it **bought 60 more seconds** — enough for a VNS activation, a phone call, or reaching a safe position.

---

## Reproducibility

```bash
cargo run --release -p seizure-therapeutic-sim
```

Runs in ~10 seconds. No external data needed.
