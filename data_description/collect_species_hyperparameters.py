"""
Collect the per-species hyperparameters from the experiment 2 synthetic data generation
and save them as a CSV for the data_description folder.

This mirrors the species parameter construction in experiment2_data_generation.py
so that readers can inspect exactly which parameters vary across species.
"""

import pandas as pd
import copy

# Base parameters (constant across all species)
base_params = {
    "sample_rate": 16_000,
    "duration": 60,
    "individual": "none",
    "noise_variance": 0.01,
    "noise_floor_db": -40,
    "spectral_tilt": -8,
}

base_freq_centers = [500, 650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850]

rows = []

for i in range(10):
    center_freq = base_freq_centers[i]
    range_width = 300 + (i * 80)

    base_freq_min = center_freq - (range_width / 2)
    base_freq_max = center_freq + (range_width / 2)

    bout_min = max(1.0, 3.0 - i * 0.2)
    bout_max = 5.0 + i * 1.5

    spacing_min = max(0.3, 1.5 - i * 0.1)
    spacing_max = 1.8 + i * 0.3

    call_min = max(0.2, 0.5 - i * 0.03)
    call_max = 1.0 + i * 0.2

    harmonic_min = 3
    harmonic_max = min(12, 6 + i * 2)

    decay_min = max(0.2, 0.4 - i * 0.02)
    decay_max = min(0.9, 0.6 + i * 0.05)

    wiener_threshold = min(0.2 + i * 0.1, 0.6)
    euclidean_threshold = min(0.2 + i * 0.1, 0.5)

    sleep_min = max(0.5, 0.8 - i * 0.05)
    sleep_max = 2.0 + i * 0.5

    # Formant parameters
    base_freq_mid = center_freq
    formant_params_raw = [
        (base_freq_mid, base_freq_mid * 0.2, 1.0),
        (base_freq_mid * 2, base_freq_mid * 0.3, 0.5),
        (base_freq_mid * 3, base_freq_mid * 0.4, 0.25),
    ]
    n_safe_formants = sum(
        1 for fc, bw, _ in formant_params_raw if fc + (bw / 2) < 8000
    )

    rows.append({
        "species": f"species_{i}",
        "freq_center_hz": center_freq,
        "base_freq_min_hz": base_freq_min,
        "base_freq_max_hz": base_freq_max,
        "freq_bandwidth_hz": range_width,
        "bout_duration_min_s": bout_min,
        "bout_duration_max_s": bout_max,
        "call_spacing_min_s": spacing_min,
        "call_spacing_max_s": spacing_max,
        "call_duration_min_s": call_min,
        "call_duration_max_s": call_max,
        "harmonic_count_min": harmonic_min,
        "harmonic_count_max": harmonic_max,
        "harmonic_decay_min": decay_min,
        "harmonic_decay_max": decay_max,
        "wiener_entropy_threshold": wiener_threshold,
        "euclidean_distance_threshold": euclidean_threshold,
        "sleep_duration_min_s": sleep_min,
        "sleep_duration_max_s": sleep_max,
        "n_formants": n_safe_formants,
    })

df = pd.DataFrame(rows)
df.to_csv("experiment2_species_hyperparameters.csv", index=False)
print(df.to_string(index=False))
print(f"\nSaved to experiment2_species_hyperparameters.csv")
