import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import random

def sigmoid(x):
    # Clip x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Define clip length in seconds
clip_length_seconds = 30

# Define directories with species indices
species_dirs = {
    'species_0': '/home/jacob/data/jacob/borg_species_sweep_izs/species_0/',
    'species_5': '/home/jacob/data/jacob/borg_species_sweep_izs/species_5/',
    'species_9': '/home/jacob/data/jacob/borg_species_sweep_izs/species_9/'
}

def get_random_wav_file(dir_path):
    wav_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.wav')]
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {dir_path}")
    return os.path.join(dir_path, random.choice(wav_files))

# Randomly select one file from each species folder
full_path_1 = get_random_wav_file(species_dirs['species_0'])
full_path_2 = get_random_wav_file(species_dirs['species_5'])
full_path_3 = get_random_wav_file(species_dirs['species_9'])

# Load the audio files
y1, sr1 = librosa.load(full_path_1, sr=None, mono=True)
y2, sr2 = librosa.load(full_path_2, sr=None, mono=True)
y3, sr3 = librosa.load(full_path_3, sr=None, mono=True)

# Clip the audio to desired length
def clip_audio(y, sr, clip_len):
    if len(y) < sr * clip_len:
        print(f"Warning: File is shorter than {clip_len} seconds. Using full file.")
        return y
    return y[:sr * clip_len]

y1_clip = clip_audio(y1, sr1, clip_length_seconds)
y2_clip = clip_audio(y2, sr2, clip_length_seconds)
y3_clip = clip_audio(y3, sr3, clip_length_seconds)

# Print clip info with species index
print(f"Species 0 - {os.path.basename(full_path_1)}: {len(y1_clip)/sr1:.2f}s at {sr1} Hz")
print(f"Species 5 - {os.path.basename(full_path_2)}: {len(y2_clip)/sr2:.2f}s at {sr2} Hz")
print(f"Species 9 - {os.path.basename(full_path_3)}: {len(y3_clip)/sr3:.2f}s at {sr3} Hz")

# Fixed spectrogram function
def create_power_spectrogram(audio, sr):
    stft = librosa.stft(audio, n_fft=510, hop_length=255)
    power_spec = np.abs(stft) ** 1
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    power_spec_safe = np.maximum(power_spec, epsilon)
    #return power_spec_safe
    #return sigmoid(power_spec)
    return sigmoid(np.log10(power_spec_safe))

# Create spectrograms
power_spec_db1 = create_power_spectrogram(y1_clip, sr1)
power_spec_db2 = create_power_spectrogram(y2_clip, sr2)
power_spec_db3 = create_power_spectrogram(y3_clip, sr3)

# Normalize color scale
vmin = min(power_spec_db1.min(), power_spec_db2.min(), power_spec_db3.min())
vmax = max(power_spec_db1.max(), power_spec_db2.max(), power_spec_db3.max())

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Store the image objects for colorbar
images = []

# Species indices for titles
species_indices = [0, 5, 9]

for i, (spec, sr, path, ax, species_idx) in enumerate(zip(
    [power_spec_db1, power_spec_db2, power_spec_db3],
    [sr1, sr2, sr3],
    [full_path_1, full_path_2, full_path_3],
    axes,
    species_indices
)):
    im = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=255,
        x_axis='time',
        y_axis='linear',
        cmap='gray',
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )
    images.append(im)
    ax.set_title(f"Species {species_idx} - Power Spectrogram ({clip_length_seconds}s)\n{os.path.basename(path)}")
    ax.set_ylabel('Frequency (Hz)' if i == 0 else '')
    ax.set_xlabel('Time (s)')

# Add a single colorbar using the first image
cbar = fig.colorbar(images[0], ax=axes, shrink=0.6, aspect=30, pad=0.02)
cbar.set_label('Amplitude', rotation=270, labelpad=15)

# Save figure with species indices in filename
plt.savefig("species_0_5_9_spectrograms_grayscale.png", dpi=150, bbox_inches='tight')
print("Saved spectrograms as 'species_0_5_9_spectrograms_grayscale.png'")
#plt.show()