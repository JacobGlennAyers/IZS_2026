import copy
import itertools
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any, Union
from experiment_runner import run_single_experiment


def create_parameter_combinations(base_params: Dict[str, Any], 
                                variable_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create all possible combinations of parameters from the variable_params dictionary.
    
    Args:
        base_params: Base parameter dictionary with default values
        variable_params: Dictionary where keys are parameter names and values are lists of options
        
    Returns:
        List of parameter dictionaries, one for each combination
    """
    if not variable_params:
        return [base_params]
    
    # Get all parameter names and their possible values
    param_names = list(variable_params.keys())
    param_values = list(variable_params.values())
    
    # Create all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        # Start with base parameters
        param_dict = copy.deepcopy(base_params)
        
        # Update with current combination
        for param_name, param_value in zip(param_names, combination):
            # Handle nested parameter updates (e.g., "optimizer_parameters.lr")
            if '.' in param_name:
                keys = param_name.split('.')
                current_dict = param_dict
                for key in keys[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                current_dict[keys[-1]] = param_value
            else:
                param_dict[param_name] = param_value
        
        combinations.append(param_dict)
    
    return combinations


def generate_experiment_id(params: Dict[str, Any], variable_params: Dict[str, List[Any]]) -> str:
    """
    Generate a unique experiment ID based on the varying parameters.
    
    Args:
        params: Parameter dictionary for this experiment
        variable_params: Dictionary of parameters that vary across experiments
        
    Returns:
        String identifier for the experiment
    """
    id_parts = []
    
    for param_name in variable_params.keys():
        if '.' in param_name:
            # Handle nested parameters
            keys = param_name.split('.')
            value = params
            for key in keys:
                value = value[key]
        else:
            value = params[param_name]
        
        # Convert value to string and clean it up
        value_str = str(value).replace('/', '_').replace(' ', '_').replace('.', 'p')
        id_parts.append(f"{param_name.split('.')[-1]}{value_str}")
    
    return "_".join(id_parts)


def run_batch_experiments(base_params: Dict[str, Any], 
                         variable_params: Dict[str, List[Any]],
                         experiment_prefix: str = "batch_exp",
                         save_summary: bool = True) -> pd.DataFrame:
    """
    Run multiple experiments with different parameter combinations.
    
    Args:
        base_params: Base parameter dictionary with default values
        variable_params: Dictionary where keys are parameter names and values are lists of options
        experiment_prefix: Prefix for experiment names
        save_summary: Whether to save a summary CSV of all experiments
        
    Returns:
        DataFrame containing results from all experiments
    """
    # Create all parameter combinations
    param_combinations = create_parameter_combinations(base_params, variable_params)
    
    print(f"Running {len(param_combinations)} experiments...")
    print(f"Variable parameters: {list(variable_params.keys())}")
    print("-" * 60)
    
    all_results = []
    
    for i, params in enumerate(param_combinations):
        # Generate experiment ID
        experiment_id = f"{experiment_prefix}_{generate_experiment_id(params, variable_params)}"
        
        try:
            # Run the experiment
            result = run_single_experiment(params, experiment_id=experiment_id, verbose=True)
            all_results.append(result)
            
        except Exception as e:
            print(f"Error in experiment {experiment_id}: {str(e)}")
            # Log the error but continue with other experiments
            error_result = {
                "experiment_name": experiment_id,
                "error": str(e),
                "status": "failed"
            }
            all_results.append(error_result)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    if save_summary:
        # Save summary to the base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"batch_experiment_summary_{experiment_prefix}_{timestamp}.csv"
        summary_path = os.path.join(base_params["output_directory"], summary_filename)
        results_df.to_csv(summary_path, index=False)
        print(f"\nBatch experiment summary saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*60)
    
    successful_experiments = results_df[~results_df.get('error', pd.Series()).notna()]
    
    if len(successful_experiments) > 0:
        print(f"Successful experiments: {len(successful_experiments)}")
        print(f"Failed experiments: {len(results_df) - len(successful_experiments)}")
        
        if 'final_val_loss' in successful_experiments.columns:
            print(f"\nValidation Loss Statistics:")
            print(f"Best (min) validation loss: {successful_experiments['final_val_loss'].min():.6f}")
            print(f"Worst (max) validation loss: {successful_experiments['final_val_loss'].max():.6f}")
            print(f"Mean validation loss: {successful_experiments['final_val_loss'].mean():.6f}")
            
            # Show best performing experiment
            best_idx = successful_experiments['final_val_loss'].idxmin()
            best_exp = successful_experiments.loc[best_idx]
            print(f"\nBest performing experiment: {best_exp['experiment_name']}")
            print(f"Best validation loss: {best_exp['final_val_loss']:.6f}")
    else:
        print("No successful experiments!")
    
    return results_df


# Example usage and templates
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    base_experiment_params = {
        "output_directory": "models",
        "experiment_name": None,  # Will be auto-generated
        "device": "cuda",
        "random_seed": 42,
        "save_model": True,
        "data_path": "data",
        "vocallbase_format": True,
        "dataset(s)": None,  # This will be overridden
        "sample_rate": 16_000,
        "audio_normalization": None,
        "spectrogram_type": "STFT_log10",
        "spectrogram_parameters": {
            "sample_rate": 16_000,
            "n_fft": 512,
            "hop_length": 256,
            "normalized": True,
            "power": 1
        },
        "validation_ratio": 0.2,
        "context_size": 128,
        "batch_size": 128,
        "model" : "FCN",
        "model_parameters" : {
            "hidden_dim1" : 1024,
            "hidden_dim2" : 1024
        },
        "criterion": "MSE",
        "criterion_parameters": {
            "reduction": "mean"
        },
        "optimizer": "AdamW",
        "optimizer_parameters": {
            "lr": 5e-5,
            "weight_decay": 0.01
        },
        "scheduler": "CosineAnnealing",
        "scheduler_parameters": {
            "T_max": 50,
            "eta_min": 1e-6
        },
        "epoch_count": 50,  
        "Training_GIF": False,  # Disabled for batch runs
        "run_test": False
    }
    
    # Example 1: Test different datasets
    print("Example 1: Testing different datasets")
    dataset_variable_params = {
        "dataset(s)": ["species_0","species_1","species_2","species_3","species_4","species_5","species_6","species_7","species_8", "species_9"]
    }
    
    # Uncomment to run:
    dataset_results = run_batch_experiments(
         base_experiment_params, 
         dataset_variable_params,
         experiment_prefix=""
     )
    
    