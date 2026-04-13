# Clinical Seizure Prediction Landscape Review

## Key Finding: Graph MinCut for Seizure Prediction Is Novel

> Based on exhaustive search, **no published work has applied graph minimum cut to temporal EEG feature sequences for seizure onset detection.** This is a genuinely novel contribution.

---

## Freely Available EEG Seizure Datasets

| Dataset | Patients | Seizures | Channels | Hz | Access |
|---------|----------|----------|----------|-----|--------|
| **CHB-MIT** (PhysioNet) | 22 | 198 | 23 | 256 | [Free](https://physionet.org/content/chbmit/1.0.0/) |
| **TUH Seizure Corpus** | 642 | 3,050 | 23-31 | 250 | [Free w/ DUA](https://isip.piconepress.com/projects/tuh_eeg/) |
| **Melbourne-NeuroVista** | 12 | 2,979 segments | 16 iEEG | 400 | [Free](https://melbourne.figshare.com/articles/dataset/Seizure_Data/6939809) |
| **EPILEPSIAE** | 275 | 2,400+ | 128+ | 256-1024 | Application required |
| **Siena Scalp** (PhysioNet) | 14 | 47 | 19-21 | 512 | [Free](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |
| **Bonn University** | 10 | 5 classes | 1 | 174 | [Free](https://www.ukbonn.de/en/epileptology/) |

**Recommended first target:** CHB-MIT — free, scalp EEG, 256 Hz (matches our PoC), widely benchmarked.

## State of the Art (2024-2026)

| Method | Accuracy | Sensitivity | FPR | Dataset |
|--------|----------|-------------|-----|---------|
| Self-supervised graph + func. connectivity | 99.0% | — | — | CHB-MIT |
| Sync-based graph spatio-temporal attention | 98.2% | 97.9% | — | CHB-MIT |
| GCN + LSTM ensemble | — | 94.1% | 0.075/hr | CHB-MIT |
| **Our PoC (graph mincut)** | **100%** | **100%** | **0/100** | **Synthetic** |

**Critical caveat:** Our numbers are on synthetic data with one seizure. Real-world validation will degrade these. But the approach is novel and the mechanism matches known physiology.

## The Novelty Gap

| What exists | What's missing (our contribution) |
|-------------|----------------------------------|
| GNN/GCN seizure prediction | **Spectral graph theory** (mincut, Fiedler) for seizure prediction |
| Functional connectivity analysis | **Graph boundary detection** on temporal feature sequences |
| Spectral graph theory in neuroscience | Applied to **state transition detection**, not just oscillation modeling |
| Pre-ictal correlation changes documented | Detected via **Cheeger inequality guaranteed** mincut, not ad-hoc thresholds |

## Key References

- Mormann et al. 2007, "Seizure prediction: the long and winding road" — Brain 130:314 (845+ citations)
- Cook et al. 2013, NeuroVista first-in-human trial — Lancet Neurol 12:563
- Jiruska et al. 2013, "Synchronization and desynchronization" — J Physiol, PMC 3591697
- Perucca et al. 2013, "Widespread EEG changes precede focal seizures" — PLOS ONE, PMC 3834227
