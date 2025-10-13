import os
import torch
import pandas as pd
import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from scipy.ndimage import gaussian_filter1d

# Add the project directory to Python path (adjust as needed)

from models import get_model
from data_setup import get_data
from criterions import get_loss

def sigmoid(x):
    """Sigmoid function with clipping to prevent overflow"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def robust_logdet(matrix, threshold=1e-10):
    """Compute log determinant robustly using SVD."""
    U, S, V = torch.linalg.svd(matrix)
    return torch.sum(torch.log(S))

def PECEP(error_cov, N):
    """Compute Differential Entropy Correlated Matrix estimate."""
    dimension = error_cov.shape[0]
    dimension_constant = 0.5 * dimension * torch.log(torch.tensor(2 * torch.pi * torch.e))
    PECEP = dimension_constant + 0.5 * (dimension * torch.log(torch.tensor(1/N)) + robust_logdet(error_cov))
    return PECEP, N

def load_experiment_parameters(experiment_path):
    """Load experiment parameters from pickle file."""
    pkl_path = os.path.join(experiment_path, "experiment_parameters.pkl")
    with open(pkl_path, "rb") as file:
        experiment_parameters = pickle.load(file)
    return experiment_parameters

def extract_species_number(experiment_name):
    """Extract species number from experiment directory name."""
    parts = experiment_name.split('species_')
    if len(parts) > 1:
        species_part = parts[1].split('_')[0]
        try:
            return int(species_part)
        except ValueError:
            pass
    return None

def load_model_and_data(experiment_path, dataset_dir):
    """Load trained model and test dataset for a species."""
    print(f"Loading experiment: {os.path.basename(experiment_path)}")
    
    # Check if model file exists
    model_file = os.path.join(experiment_path, "experiment_model.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model file found in {experiment_path}")
    
    # Load experiment parameters
    experiment_parameters = load_experiment_parameters(experiment_path)
    
    # Extract species number
    experiment_name = os.path.basename(experiment_path)
    species_num = extract_species_number(experiment_name)
    
    if species_num is None:
        raise ValueError(f"Could not extract species number from {experiment_name}")
    
    # Update experiment parameters to point to the correct species
    experiment_parameters["dataset(s)"] = f"species_{species_num}"
    experiment_parameters["data_path"] = dataset_dir
    
    # Load data and model
    _, test_dataset, _, _ = get_data(experiment_parameters)
    model = get_model(experiment_parameters, test_dataset).to(experiment_parameters["device"])
    model.load_state_dict(torch.load(model_file, map_location=experiment_parameters["device"], weights_only=True))
    model.eval()
    
    return model, test_dataset, experiment_parameters, species_num

def load_utterances_data(dataset_dir, species_num):
    """Load utterances CSV for a species."""
    utterances_file = os.path.join(dataset_dir, f"species_{species_num}_utterances.csv")
    if not os.path.exists(utterances_file):
        raise FileNotFoundError(f"Utterances file not found: {utterances_file}")
    
    utterances_df = pd.read_csv(utterances_file)
    test_utterances = utterances_df[utterances_df['train'] == 0].copy()
    return test_utterances

def predict_single_clip_spectrogram(model, test_dataset, experiment_parameters, clip_idx):
    """
    Generate predictions for a single clip using your trained model.
    Returns both ground truth and predicted spectrograms.
    Based on your plot_spectrogram_prediction function.
    """
    model.eval()
    
    # Select the clip - this follows your existing pattern
    test_dataset.select_clip(clip_idx, state='test')
    
    # Get the original spectrogram for shape reference
    cur_spec = test_dataset._load_spec(clip_idx).squeeze().detach().numpy()
    
    # Initialize predicted spectrogram with same shape as ground truth
    predicted_spec = np.zeros(cur_spec.shape)
    
    # Create dataloader and predict frame by frame
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        ndx = 0
        for batch in dataloader:
            x, y = batch
            x = x.to(experiment_parameters["device"])
            y = y.to(experiment_parameters["device"])
            x = model.reshape_input(x)
            y = model.reshape_output(y)
            
            pred_y = model(x)
            
            # Store prediction in the correct column of the spectrogram
            # This follows the exact pattern from your plot_spectrogram_prediction
            predicted_spec[:, ndx] = pred_y.squeeze().cpu().detach().numpy()
            ndx += 1
    
    return cur_spec, predicted_spec

def get_utterance_times_from_csv(utterances_df, clip_filename):
    """Extract utterance onset/offset times for a specific clip from CSV."""
    clip_utterances = utterances_df[utterances_df['filename'] == clip_filename]
    onset_offset_pairs = []
    
    for _, row in clip_utterances.iterrows():
        onset_offset_pairs.append((row['onset'], row['offset']))
    
    return onset_offset_pairs

def frames_to_time(frame_indices, sr, hop_length):
    """Convert frame indices to time in seconds"""
    return librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length)

def time_to_spectrogram_frames(times, duration, n_frames):
    """Convert time values to spectrogram frame indices."""
    return [int((t / duration) * n_frames) for t in times]

def create_comparison_plot_with_utterances(spec1, spec2, pred1, pred2, 
                                          utterance_times1, utterance_times2,
                                          sr1, sr2, duration1, duration2,
                                          hop_length=255, species_names=["Species 0", "Species 9"]):
    """
    Create comparison plot with ground truth, predicted, and difference.
    Uses utterance times from CSV data.
    Fixed colorbar overlap and uniform difference scaling.
    Modified to use dashed green lines for utterance boundaries.
    """
    # Calculate differences
    diff1 = spec1 - pred1
    diff2 = spec2 - pred2
    
    # Set up the plot with more space for colorbars
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    # Use same colorbar range for both ground truth and predicted
    vmin_all = min(spec1.min(), spec2.min(), pred1.min(), pred2.min())
    vmax_all = max(spec1.max(), spec2.max(), pred1.max(), pred2.max())
    
    # For difference plots, use a reduced range to avoid outliers dominating
    # Use 95th percentile to set the range instead of max values
    diff1_flat = diff1.flatten()
    diff2_flat = diff2.flatten()
    all_diffs = np.concatenate([diff1_flat, diff2_flat])
    diff_95th = np.percentile(np.abs(all_diffs), 95)
    vmin_diff = -diff_95th
    vmax_diff = diff_95th
    
    print(f"Colorbar ranges - Combined GT/Pred: [{vmin_all:.4f}, {vmax_all:.4f}], Diff (95th percentile): [{vmin_diff:.4f}, {vmax_diff:.4f}]")
    
    # Row 1: Ground Truth
    im1 = librosa.display.specshow(spec1, sr=sr1, hop_length=hop_length, x_axis='time', 
                                   y_axis='linear', cmap='gray', ax=axes[0,0],
                                   vmin=vmin_all, vmax=vmax_all)
    axes[0,0].set_title(f'{species_names[0]} - Ground Truth')
    axes[0,0].set_ylabel('Frequency (Hz)')
    
    im2 = librosa.display.specshow(spec2, sr=sr2, hop_length=hop_length, x_axis='time', 
                                   y_axis='linear', cmap='gray', ax=axes[0,1],
                                   vmin=vmin_all, vmax=vmax_all)
    axes[0,1].set_title(f'{species_names[1]} - Ground Truth')
    
    # Row 2: Predicted
    im3 = librosa.display.specshow(pred1, sr=sr1, hop_length=hop_length, x_axis='time', 
                                  y_axis='linear', cmap='gray', ax=axes[1,0],
                                  vmin=vmin_all, vmax=vmax_all)
    axes[1,0].set_title(f'{species_names[0]} - Predicted')
    axes[1,0].set_ylabel('Frequency (Hz)')
    
    im4 = librosa.display.specshow(pred2, sr=sr2, hop_length=hop_length, x_axis='time', 
                                  y_axis='linear', cmap='gray', ax=axes[1,1],
                                  vmin=vmin_all, vmax=vmax_all)
    axes[1,1].set_title(f'{species_names[1]} - Predicted')
    
    # Row 3: Difference with utterance lines from CSV (modified to use dashed green lines)
    im5 = librosa.display.specshow(diff1, sr=sr1, hop_length=hop_length, x_axis='time', 
                                  y_axis='linear', cmap='RdBu_r', ax=axes[2,0],
                                  vmin=vmin_diff, vmax=vmax_diff)
    axes[2,0].set_title(f'{species_names[0]} - Difference (GT - Predicted)')
    axes[2,0].set_ylabel('Frequency (Hz)')
    axes[2,0].set_xlabel('Time (s)')
    
    # Add utterance lines for species 0 (from CSV) - changed to dashed green lines
    for onset_time, offset_time in utterance_times1:
        axes[2,0].axvline(x=onset_time, color='darkgreen', linestyle='-', linewidth=1.1, alpha=0.9)
        axes[2,0].axvline(x=offset_time, color='darkgreen', linestyle='-', linewidth=1.1, alpha=0.9)
    
    im6 = librosa.display.specshow(diff2, sr=sr2, hop_length=hop_length, x_axis='time', 
                                  y_axis='linear', cmap='RdBu_r', ax=axes[2,1],
                                  vmin=vmin_diff, vmax=vmax_diff)
    axes[2,1].set_title(f'{species_names[1]} - Difference (GT - Predicted)')
    axes[2,1].set_xlabel('Time (s)')
    
    # Add utterance lines for species 9 (from CSV) - changed to dark green solid lines
    for onset_time, offset_time in utterance_times2:
        axes[2,1].axvline(x=onset_time, color='darkgreen', linestyle='-', linewidth=1.1, alpha=0.9)
        axes[2,1].axvline(x=offset_time, color='darkgreen', linestyle='-', linewidth=1.1, alpha=0.9)
    
    # Adjust layout first to prevent overlap
    plt.subplots_adjust(left=0.08, right=0.82, top=0.95, bottom=0.08, hspace=0.3, wspace=0.3)
    
    # Add colorbars with better positioning to avoid overlap
    # Combined colorbar for ground truth and predicted (rows 1 and 2)
    cbar1_ax = fig.add_axes([0.84, 0.54, 0.02, 0.39])  # Spans both GT and Predicted rows
    cbar1 = fig.colorbar(im1, cax=cbar1_ax)
    cbar1.set_label('Amplitude (GT & Predicted)', rotation=270, labelpad=15)
    
    # Separate colorbar for difference (row 3)
    cbar2_ax = fig.add_axes([0.84, 0.12, 0.02, 0.25])
    cbar2 = fig.colorbar(im5, cax=cbar2_ax)
    cbar2.set_label('Difference (GT - Predicted)', rotation=270, labelpad=15)
    
    return fig

def compute_utterance_pecep_for_clip(model, test_dataset, experiment_parameters, utterances_df, clip_idx):
    """
    Compute PECEP scores for utterances in a specific clip.
    Based on your existing PECEP computation code.
    """
    model.eval()
    dimension = model.d_model
    
    # Get clip filename
    if test_dataset.vocallbase_format and hasattr(test_dataset, 'base_dataset'):
        try:
            clip_filename = test_dataset.base_dataset["filename"][clip_idx]
            print("Filename:", clip_filename)
        except (KeyError, IndexError):
            return []
    else:
        clip_filename = f'clip_{clip_idx}'
    
    # Get utterances for this clip
    clip_utterances = utterances_df[utterances_df['filename'] == clip_filename]
    if len(clip_utterances) == 0:
        return []
    
    # Get the original spectrogram for time mapping
    original_spec = test_dataset._load_spec(clip_idx).squeeze().detach().numpy()
    n_spec_columns = original_spec.shape[1]
    
    # Get clip duration
    if test_dataset.vocallbase_format:
        try:
            onset = test_dataset.base_dataset["onset"][clip_idx]
            offset = test_dataset.base_dataset["offset"][clip_idx]
            clip_duration = offset - onset
        except (KeyError, IndexError):
            clip_duration = 60.0
    else:
        clip_duration = 60.0
    
    # Run inference on the full clip
    test_dataset.select_clip(clip_idx, state='test')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    error_residuals = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(experiment_parameters["device"])
            y = y.to(experiment_parameters["device"])
            x = model.reshape_input(x)
            y = model.reshape_output(y)
            
            pred_y = model(x)
            error = y - pred_y
            error = error.reshape(dimension, 1)
            error_residuals.append(error.cpu())
    
    if len(error_residuals) == 0:
        return []
    
    error_residuals_tensor = torch.cat(error_residuals, dim=1)
    
    # Compute PECEP for each utterance
    utterance_pecep_scores = []
    
    for _, utterance_row in clip_utterances.iterrows():
        # Calculate utterance position in clip
        if test_dataset.vocallbase_format:
            clip_onset = test_dataset.base_dataset["onset"][clip_idx]
            utterance_start_in_clip = utterance_row['onset'] - clip_onset
            utterance_end_in_clip = utterance_row['offset'] - clip_onset
        else:
            utterance_start_in_clip = utterance_row['onset']
            utterance_end_in_clip = utterance_row['offset']
        
        # Convert to frame indices
        start_col = int((utterance_start_in_clip / clip_duration) * n_spec_columns) if clip_duration > 0 else 0
        end_col = int((utterance_end_in_clip / clip_duration) * n_spec_columns) if clip_duration > 0 else n_spec_columns
        
        start_col = max(0, min(start_col, error_residuals_tensor.shape[1] - 1))
        end_col = max(start_col + 1, min(end_col, error_residuals_tensor.shape[1]))
        
        # Extract errors for this utterance
        utterance_errors = error_residuals_tensor[:, start_col:end_col]
        
        if utterance_errors.shape[1] == 0:
            continue
        
        # Compute PECEP
        sum_of_error_outer_products = torch.zeros((dimension, dimension))
        N_utterance = utterance_errors.shape[1]
        
        for frame_idx in range(N_utterance):
            error_frame = utterance_errors[:, frame_idx:frame_idx+1]
            outer_product = error_frame @ error_frame.T
            sum_of_error_outer_products += outer_product
        
        pecep_value, N_frames = PECEP(sum_of_error_outer_products, N_utterance)
        
        utterance_pecep_scores.append({
            'onset': utterance_row['onset'],
            'offset': utterance_row['offset'],
            'PECEP': pecep_value.item(),
            'N_frames': N_frames
        })
    
    return utterance_pecep_scores

def main():
    """Main function to run the complete analysis with trained models."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Configuration paths
    model_dir = "models"
    dataset_dir = "data"
    
    # Species to compare
    species_0_experiment = None
    species_9_experiment = None
    
    # Find experiment directories for species 0 and 9
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            species_num = extract_species_number(item)
            if species_num == 0:
                species_0_experiment = item_path
            elif species_num == 9:
                species_9_experiment = item_path
    
    if not species_0_experiment or not species_9_experiment:
        print(f"Could not find experiments for species 0 and/or 9 in {model_dir}")
        return
    
    print("Loading models and data...")
    
    # Load models and datasets
    model_0, test_dataset_0, exp_params_0, _ = load_model_and_data(species_0_experiment, dataset_dir)
    model_9, test_dataset_9, exp_params_9, _ = load_model_and_data(species_9_experiment, dataset_dir)
    
    # Load utterances data
    utterances_0 = load_utterances_data(dataset_dir, 0)
    utterances_9 = load_utterances_data(dataset_dir, 9)
    
    print(f"Species 0: {len(utterances_0)} test utterances")
    print(f"Species 9: {len(utterances_9)} test utterances")
    
    # Fixed clip selection instead of random (seed is already set above for additional reproducibility)
    # Using modulo to ensure we don't exceed dataset bounds
    clip_idx_0 = 5 % test_dataset_0.clip_count  # Fixed clip index for species 0
    clip_idx_9 = 7 % test_dataset_9.clip_count  # Fixed clip index for species 9
    
    print(f"Selected clip {clip_idx_0} for species 0, clip {clip_idx_9} for species 9")
    
    # Generate predictions for selected clips
    print("Generating predictions...")
    
    # Use the same method as your plot_spectrogram_prediction function
    gt_spec_0, pred_spec_0 = predict_single_clip_spectrogram(model_0, test_dataset_0, exp_params_0, clip_idx_0)
    gt_spec_9, pred_spec_9 = predict_single_clip_spectrogram(model_9, test_dataset_9, exp_params_9, clip_idx_9)
    
    print(f"Ground truth shapes: Species 0: {gt_spec_0.shape}, Species 9: {gt_spec_9.shape}")
    print(f"Prediction shapes: Species 0: {pred_spec_0.shape}, Species 9: {pred_spec_9.shape}")
    print(f"GT value ranges: Species 0: [{gt_spec_0.min():.4f}, {gt_spec_0.max():.4f}], Species 9: [{gt_spec_9.min():.4f}, {gt_spec_9.max():.4f}]")
    print(f"Pred value ranges: Species 0: [{pred_spec_0.min():.4f}, {pred_spec_0.max():.4f}], Species 9: [{pred_spec_9.min():.4f}, {pred_spec_9.max():.4f}]")
    
    # Get utterance times for selected clips
    clip_filename_0 = test_dataset_0.base_dataset["filename"][clip_idx_0] if test_dataset_0.vocallbase_format else f'clip_{clip_idx_0}'
    clip_filename_9 = test_dataset_9.base_dataset["filename"][clip_idx_9] if test_dataset_9.vocallbase_format else f'clip_{clip_idx_9}'
    print(clip_filename_0, clip_filename_9)
    
    utterance_times_0 = get_utterance_times_from_csv(utterances_0, clip_filename_0)
    utterance_times_9 = get_utterance_times_from_csv(utterances_9, clip_filename_9)
    
    print(f"Species 0 clip has {len(utterance_times_0)} utterances")
    print(f"Species 9 clip has {len(utterance_times_9)} utterances")
    
    # Create comparison plot
    print("Creating comparison plot...")
    
    # Estimate sample rates and durations (adjust based on your data format)
    sr_0 = sr_9 = 16000  # Updated to 16kHz
    duration_0 = gt_spec_0.shape[1] * 255 / sr_0  # hop_length=255
    duration_9 = gt_spec_9.shape[1] * 255 / sr_9
    
    fig = create_comparison_plot_with_utterances(
        gt_spec_0, gt_spec_9, pred_spec_0, pred_spec_9,
        utterance_times_0, utterance_times_9,
        sr_0, sr_9, duration_0, duration_9,
        species_names=["Species 0", "Species 9"]
    )
    
    # Save the plot
    output_filename = f"figures/izs_exp2_vis1_fig2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as '{output_filename}'")
    
    #plt.show()

if __name__ == "__main__":
    main()