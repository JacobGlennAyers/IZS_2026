import pandas as pd
import os
import librosa
import numpy as np


experiment2_data_path = "data"
# Get all CSV files
annotation_files = []
for f in os.listdir(experiment2_data_path):
    if f.endswith(".csv") and f.startswith("species_") and not f.endswith("_utterances.csv"):
        annotation_files.append(f)

# Process each annotation file
for annotation_file in annotation_files:
    print(f"Processing {annotation_file}")
    df = pd.read_csv(annotation_file)
    
    # Get unique audio clips
    audio_clips = df["filename"].unique()
    
    # Create train-test split (80-20)
    np.random.seed(42)  # For reproducibility
    train_clips = np.random.choice(audio_clips, size=int(0.8 * len(audio_clips)), replace=False)
    
    # Create a mapping of clips to train/test
    clip_to_train = {}
    for clip in audio_clips:
        clip_to_train[clip] = 1 if clip in train_clips else 0
    
    # Add train column to original dataframe
    df['train'] = df['filename'].map(clip_to_train)
    
    # Save original fine-grain annotations with train column
    original_filename = annotation_file.replace('.csv', '_utterances.csv')
    df.to_csv(original_filename, index=False)
    print(f"Saved fine-grain annotations to {original_filename}")
    
    # Create clip-level dataframe (one row per clip with full duration)
    clip_data = []
    
    for clip in audio_clips:
        try:
            # Load audio file to get duration
            y, sr = librosa.load(clip, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get other info from the first occurrence of this clip
            first_occurrence = df[df['filename'] == clip].iloc[0]
            
            clip_data.append({
                'onset': 0.0,
                'offset': duration,
                'duration': duration,
                'minFrequency': first_occurrence['minFrequency'],
                'maxFrequency': first_occurrence['maxFrequency'],
                'species': first_occurrence['species'],
                'individual': first_occurrence['individual'],
                'filename': clip,
                'channelIndex': first_occurrence['channelIndex'],
                'train': clip_to_train[clip]
            })
            
        except Exception as e:
            print(f"Error processing {clip}: {e}")
            continue
    
    # Create clip-level dataframe
    clip_df = pd.DataFrame(clip_data)
    
    # Save clip-level annotations
    clip_df.to_csv(annotation_file, index=False)
    print(f"Saved clip-level annotations to {annotation_file}")
    print(f"Train clips: {len(clip_df[clip_df['train'] == 1])}, Test clips: {len(clip_df[clip_df['train'] == 0])}")
