import os
import torch
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys

# Add the project directory to Python path (adjust as needed)
sys.path.append('/home/jacob/projects/jacob')

from models import get_model
from data_setup import get_data
from criterions import get_loss

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
    # Handle format like '_dataset(s)species_0_20250801_175054'
    parts = experiment_name.split('species_')
    if len(parts) > 1:
        species_part = parts[1].split('_')[0]
        try:
            return int(species_part)
        except ValueError:
            pass
    return None

def compute_utterance_pecep(model, test_dataset, experiment_parameters, utterances_df):
    """
    Compute PECEP scores for each utterance in the test set.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        experiment_parameters: Experiment configuration
        utterances_df: DataFrame with utterance information (already filtered to test set)
    
    Returns:
        DataFrame with PECEP scores added
    """
    model.eval()
    criterion = get_loss(experiment_parameters)
    dimension = model.d_model
    
    # Add PECEP column if it doesn't exist
    if 'PECEP' not in utterances_df.columns:
        utterances_df['PECEP'] = None
    if 'N_frames' not in utterances_df.columns:
        utterances_df['N_frames'] = None
    
    # Create a mapping from test dataset clips to utterances
    # First, get all unique filenames from the test dataset
    test_filenames = set()
    if test_dataset.vocallbase_format and hasattr(test_dataset, 'base_dataset'):
        for i in range(test_dataset.clip_count):
            try:
                filename = test_dataset.base_dataset["filename"][i]
                test_filenames.add(filename)
            except (KeyError, IndexError):
                continue
    
    # Filter utterances to only those that have corresponding clips in test set
    test_utterances_subset = utterances_df[utterances_df['filename'].isin(test_filenames)].copy()
    
    if len(test_utterances_subset) == 0:
        print("No utterances found matching test dataset clips")
        return utterances_df
    
    print(f"Processing {len(test_utterances_subset)} utterances across {len(test_filenames)} test clips")
    
    # Group utterances by filename for efficient processing
    grouped = test_utterances_subset.groupby('filename')
    
    with torch.no_grad():
        for clip_idx in tqdm(range(test_dataset.clip_count), desc="Processing clips"):
            test_dataset.select_clip(clip_idx, state='test')
            
            # Get clip filename - handle vocallbase format correctly
            if test_dataset.vocallbase_format and hasattr(test_dataset, 'base_dataset'):
                try:
                    clip_filename = test_dataset.base_dataset["filename"][clip_idx]
                except (KeyError, IndexError):
                    print(f"Could not get filename for clip {clip_idx}, skipping...")
                    continue
            else:
                clip_filename = f'clip_{clip_idx}'
            
            # Skip if no utterances for this clip
            if clip_filename not in grouped.groups:
                continue
            
            clip_utterances = grouped.get_group(clip_filename)
            
            # Get the original spectrogram for time mapping
            original_spec = test_dataset._load_spec(clip_idx).squeeze().detach().numpy()
            n_spec_columns = original_spec.shape[1]
            
            # For vocallbase format, get clip duration from the data
            if test_dataset.vocallbase_format:
                try:
                    onset = test_dataset.base_dataset["onset"][clip_idx]
                    offset = test_dataset.base_dataset["offset"][clip_idx]
                    clip_duration = offset - onset
                except (KeyError, IndexError):
                    clip_duration = 60.0  # Default fallback
            else:
                clip_duration = 60.0  # Default for other formats
            
            # Run inference on the full clip to get error residuals
            dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            error_residuals = []
            
            for batch in dataloader:
                x, y = batch
                x = x.to(experiment_parameters["device"])
                y = y.to(experiment_parameters["device"])
                x = model.reshape_input(x)
                y = model.reshape_output(y)
                
                pred_y = model(x)
                error = y - pred_y
                error = error.reshape(dimension, 1)  # Reshape to column vector
                error_residuals.append(error.cpu())
            
            # Convert error_residuals to tensor: (dimension, n_time_frames)
            if len(error_residuals) > 0:
                error_residuals_tensor = torch.cat(error_residuals, dim=1)
            else:
                continue
            
            # Process each utterance in this clip
            for idx, utterance_row in clip_utterances.iterrows():
                # For vocallbase format, onset/offset are relative to clip start
                if test_dataset.vocallbase_format:
                    # Utterance times are absolute, clip has its own onset
                    clip_onset = test_dataset.base_dataset["onset"][clip_idx]
                    utterance_start_in_clip = utterance_row['onset'] - clip_onset
                    utterance_end_in_clip = utterance_row['offset'] - clip_onset
                else:
                    utterance_start_in_clip = utterance_row['onset']
                    utterance_end_in_clip = utterance_row['offset']
                
                # Convert time to spectrogram column indices
                start_col = int((utterance_start_in_clip / clip_duration) * n_spec_columns) if clip_duration > 0 else 0
                end_col = int((utterance_end_in_clip / clip_duration) * n_spec_columns) if clip_duration > 0 else n_spec_columns
                
                # Ensure indices are within bounds
                start_col = max(0, min(start_col, error_residuals_tensor.shape[1] - 1))
                end_col = max(start_col + 1, min(end_col, error_residuals_tensor.shape[1]))
                
                # Extract error residuals for this utterance
                utterance_errors = error_residuals_tensor[:, start_col:end_col]
                
                if utterance_errors.shape[1] == 0:
                    print(f"No frames for utterance at {utterance_row['onset']}-{utterance_row['offset']} in {clip_filename}")
                    continue
                
                # Compute sum of outer products for this utterance
                sum_of_error_outer_products = torch.zeros((dimension, dimension))
                N_utterance = utterance_errors.shape[1]
                
                for frame_idx in range(N_utterance):
                    error_frame = utterance_errors[:, frame_idx:frame_idx+1]  # Keep as column vector
                    outer_product = error_frame @ error_frame.T
                    sum_of_error_outer_products += outer_product
                
                # Compute PECEP for this utterance
                pecep_value, N_frames = PECEP(sum_of_error_outer_products, N_utterance)
                
                # Store results in the original dataframe using the index
                utterances_df.loc[idx, 'PECEP'] = pecep_value.item()
                utterances_df.loc[idx, 'N_frames'] = N_frames
    
    return utterances_df

def process_experiment(experiment_path, dataset_dir):
    """Process a single experiment directory."""
    print(f"\nProcessing experiment: {os.path.basename(experiment_path)}")
    
    # Check if model file exists
    model_file = os.path.join(experiment_path, "experiment_model.pt")
    if not os.path.exists(model_file):
        print(f"No model file found in {experiment_path}, skipping...")
        return
    
    # Load experiment parameters
    try:
        experiment_parameters = load_experiment_parameters(experiment_path)
    except Exception as e:
        print(f"Error loading experiment parameters: {e}")
        return
    
    # Extract species number from directory name
    experiment_name = os.path.basename(experiment_path)
    species_num = extract_species_number(experiment_name)
    
    if species_num is None:
        print(f"Could not extract species number from {experiment_name}")
        return
    
    # Load utterances CSV
    utterances_file = os.path.join(dataset_dir, f"species_{species_num}_utterances.csv")
    if not os.path.exists(utterances_file):
        print(f"Utterances file not found: {utterances_file}")
        return
    
    print(f"Loading utterances from: {utterances_file}")
    utterances_df = pd.read_csv(utterances_file)
    
    # Filter to test set only (train column == 0)
    test_utterances = utterances_df[utterances_df['train'] == 0].copy()
    print(f"Found {len(test_utterances)} test utterances")
    
    if len(test_utterances) == 0:
        print("No test utterances found, skipping...")
        return
    
    # Update experiment parameters to point to the correct species
    experiment_parameters["dataset(s)"] = f"species_{species_num}"
    experiment_parameters["data_path"] = dataset_dir
    
    try:
        # Load data and model
        print("Loading test dataset...")
        _, test_dataset, _, _ = get_data(experiment_parameters)
        
        print("Loading model...")
        model = get_model(experiment_parameters, test_dataset).to(experiment_parameters["device"])
        model.load_state_dict(torch.load(model_file, map_location=experiment_parameters["device"], weights_only=True))
        
        # Compute PECEP scores
        print("Computing PECEP scores...")
        test_utterances_with_pecep = compute_utterance_pecep(model, test_dataset, experiment_parameters, test_utterances)
        
        # Save results
        output_file = os.path.join(experiment_path, f"species_{species_num}_test_utterances_with_PECEP.csv")
        test_utterances_with_pecep.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")
        
        # Print summary statistics
        valid_pecep = test_utterances_with_pecep['PECEP'].dropna()
        if len(valid_pecep) > 0:
            print(f"PECEP statistics - Mean: {valid_pecep.mean():.4f}, Std: {valid_pecep.std():.4f}, Count: {len(valid_pecep)}")
        
    except Exception as e:
        print(f"Error processing experiment {experiment_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process all experiments."""
    # newer experiments with updated audio data generation process
    model_dir = "/home/jacob/data/jacob/experiments/izs_exp_simpler"
    dataset_dir = "/home/jacob/projects/jacob/izs"
    # older audio data generation process
    #model_dir = "/home/jacob/data/jacob/experiments/izs_exp_2"
    #dataset_dir = "/home/jacob/data/jacob/borg_species_sweep_izs"
    
    print(f"Scanning model directory: {model_dir}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Get all experiment directories
    experiment_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            experiment_dirs.append(item_path)
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process each experiment
    for experiment_path in sorted(experiment_dirs):
        process_experiment(experiment_path, dataset_dir)
    
    print("\nAll experiments processed!")

if __name__ == "__main__":
    main()