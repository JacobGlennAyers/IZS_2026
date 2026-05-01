# Data Description

This folder documents the synthetic datasets used in both experiments of the paper *"Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy"*.

Neither experiment uses real-world data. Both datasets are fully reproducible from the generation scripts in their respective `experiment1/` and `experiment2/` directories.

---

## Environment Setup

Before running any data generation scripts, create and activate the conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate izs
```

This ensures that all dependencies (numpy, scipy, librosa, joblib, etc.) are pinned to the versions used in the paper. See [`../environment.yml`](../environment.yml) for the full specification.

---

## Experiment 1: Synthetic Autoregressive Time Series

**Purpose:** Validate that PECEP converges to the known conditional differential entropy bound for Gaussian AR processes.

**Generation script:** `experiment1/experiment1_data_generation.ipynb`

### Data generation

Each dataset is a *p*-th order vector autoregressive (VAR) process in ℝ^*d*:

> **x**_*k* = **A**₁ **x**_*k*−1 + **A**₂ **x**_*k*−2 + ⋯ + **A**_*p* **x**_*k*−*p* + **ε**_*k*,  where **ε**_*k* ∼ 𝒩(**0**, σ²**I**)

The coefficient matrices {**A**_*i*} are band-diagonal with non-zero elements drawn from U(−1, 1), normalized by their Frobenius norm, and scaled by an exponential decay factor 0.85^*i* so that recent lags carry more influence. Because the conditional entropy of this process is known analytically (it equals the entropy of the additive Gaussian noise), this setup lets us verify that PECEP converges to the true bound.

### Parameters

| Parameter | Value |
|---|---|
| Dimensionality (*d*) | 32 |
| AR order / context size (*p*) | 8 |
| Matrix normalization | Frobenius norm |
| Temporal decay factor | 0.85^*i* (per lag *i*) |
| Coefficient sampling | U(−1, 1), band-diagonal |
| Dataset size (*N*) | 1,000,000 |
| Number of independent trials | 30 |
| Additive noise variance (σ²) | {0.001, 0.01, 0.1, 1.0} |

Each of the 30 trials generates a fresh random set of coefficient matrices. For each trial, one dataset of *N* = 10⁶ samples is produced per noise variance, yielding **120 datasets** total (4 variances × 30 trials).

### Data format

Each trial is saved as a pickle file at `data/{variance}/trial_{t}.pkl` containing the generated multivariate time-series array.

### How to reproduce

```bash
conda activate izs
cd experiment1
# adjust num_trials and n_jobs as needed
jupyter nbconvert --execute experiment1_data_generation.ipynb
```

---

## Experiment 2: Synthetic Bio-Inspired Audio

**Purpose:** Test whether PECEP-based complexity ranking recovers a known monotonic complexity ordering when the true conditional entropy is intractable.

**Generation script:** `experiment2/experiment2_data_generation.py`  
**Audio generator class:** `experiment2/BioInspiredAudioGenerator.py`  
**Annotation post-processing:** `experiment2/annotation_adjustment.py`

### Generation pipeline overview

Each 60-second audio clip is built through a three-step pipeline implemented in `BioInspiredAudioGenerator`. The figure below illustrates the steps for a single clip from Species 5.


<img width="2945" height="2414" alt="Synthetic Vocalization Generation Pipeline" src="https://github.com/user-attachments/assets/31b04e0f-2968-41c4-94ae-64f1d8a5099b" />

**Step 1 — Temporal structure (bouts, calls, sleep periods).** The generator alternates between active *bouts* and silent *sleep* periods. Within each bout, individual calls are placed at irregular intervals controlled by the species' call spacing range. The bout duration, call duration, call spacing, and sleep duration ranges all widen with species index, producing increasingly irregular temporal patterns for higher-complexity species.

**Step 2 — Correlated noise generation and envelope shaping.** Each call is synthesized in the frequency domain. A fundamental frequency is drawn from the species-specific base frequency range, and integer harmonics are added with exponentially decaying amplitudes. Spectral continuity between consecutive STFT frames is enforced by Wiener entropy and Euclidean distance thresholds — frames that deviate too far from the previous one are rejected and resampled. Random phases are applied per frequency bin before an inverse FFT converts back to the time domain. Each call then receives a smooth amplitude envelope (squared sine/cosine onset and offset) to avoid spectral artifacts from abrupt transitions.

**Step 3 — Formant-based bandpass filtering and noise floor addition.** Species-specific formant resonances are applied as bandpass filters that simulate vocal tract shaping, with formant center frequencies scaling linearly with the species' base frequency range. A spectral tilt (−8 dB/octave) reduces harshness. Finally, two layers of noise are added: a constant background noise floor (−40 dB) across the entire clip simulating ambient recording conditions, and a small content-dependent noise variance applied only within active vocalizations to introduce subtle amplitude perturbations.

Although each step is deterministic given its parameters, the interaction of non-linear spectral filtering, history-dependent frame transitions, multi-scale temporal variability, and complex harmonic interactions makes analytical computation of the conditional entropy intractable — which is exactly what makes this dataset useful for evaluating complexity ranking.

### Fixed parameters (constant across all species)

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Clip duration | 60 s |
| Clips per species | 1,200 |
| Background noise floor | −40 dB (applied to the entire clip, simulating ambient recording noise) |
| Content-dependent noise variance | 0.01 (additional perturbation applied only within active vocalizations, scaled by 0.1) |
| Spectral tilt | −8 dB/octave |
| STFT window (n_fft) | 512 |
| Hop length | 256 |

### Per-species hyperparameters

The ten synthetic species are designed with monotonically increasing acoustic complexity. The full parameter table is provided in [`experiment2_species_hyperparameters.csv`](experiment2_species_hyperparameters.csv). Key complexity dimensions include:

- **Frequency bandwidth** — wider ranges allow more spectral variation
- **Bout and call timing** — larger ranges in bout duration, call spacing, and call duration produce more irregular temporal patterns
- **Harmonic richness** — more harmonics and more varied decay produce richer spectra
- **Spectral continuity thresholds** — higher Wiener entropy and Euclidean distance thresholds permit larger frame-to-frame spectral changes
- **Sleep duration variability** — longer and more variable rest periods between bouts

### Data format

Generated audio is saved as WAV files organized by species:

```
data/
├── species_0/
│   ├── 2025-01-01_00-00-00.wav
│   ├── ...
│   └── (1200 clips)
├── species_0.csv              # clip-level annotations (after annotation_adjustment.py)
├── species_0_utterances.csv   # call-level annotations (after annotation_adjustment.py)
├── species_1/
│   └── ...
└── species_9/
    └── ...
```

Annotation CSV columns follow the [VoCallBase]([https://github.com/VocAllBase](https://evolvinglanguage.ch/vocallbase-main/)) format: `onset`, `offset`, `duration`, `minFrequency`, `maxFrequency`, `species`, `individual`, `filename`, `channelIndex`, `train`.

### How to reproduce

```bash
conda activate izs
cd experiment2

# 1. Generate the audio (adjust n_jobs as needed)
python experiment2_data_generation.py

# 2. Post-process annotations into clip-level and utterance-level CSVs
#    and add train/test splits
python annotation_adjustment.py
```

---

## Collecting species hyperparameters

To regenerate `experiment2_species_hyperparameters.csv`:

```bash
cd data_description
python collect_species_hyperparameters.py
```
