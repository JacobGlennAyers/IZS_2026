import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob

def collect_pecep_data(model_dir):
    """
    Collect all PECEP data from experiment CSV files.
    
    Args:
        model_dir: Directory containing experiment subdirectories with CSV files
    
    Returns:
        combined_df: DataFrame with all PECEP scores and species information
    """
    all_data = []
    
    # Find all CSV files matching the pattern
    csv_pattern = os.path.join(model_dir, "*", "*_test_utterances_with_PECEP.csv")
    csv_files = glob(csv_pattern)
    
    print(f"Found {len(csv_files)} PECEP CSV files")
    
    for csv_file in csv_files:
        try:
            # Load the CSV
            df = pd.read_csv(csv_file)
            
            # Extract species number from filename
            filename = os.path.basename(csv_file)
            if filename.startswith("species_"):
                species_part = filename.split("_")[1]
                try:
                    species_num = int(species_part)
                except ValueError:
                    print(f"Could not extract species number from {filename}")
                    continue
            else:
                print(f"Unexpected filename format: {filename}")
                continue
            
            # Add species information
            df['species'] = species_num
            df['experiment_path'] = os.path.dirname(csv_file)
            
            # Only keep rows with valid PECEP scores
            valid_df = df.dropna(subset=['PECEP'])
            
            if len(valid_df) > 0:
                all_data.append(valid_df)
                print(f"Loaded {len(valid_df)} valid PECEP scores from species {species_num}")
            else:
                print(f"No valid PECEP scores found in {csv_file}")
                
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        raise ValueError("No valid PECEP data found!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined data: {len(combined_df)} utterances across {combined_df['species'].nunique()} species")
    
    return combined_df

def create_pecep_boxplot(combined_df, output_dir=None):
    """
    Create a box plot of PECEP scores by species with summary statistics table.
    
    Args:
        combined_df: DataFrame with PECEP scores and species information
        output_dir: Optional directory to save the plot
    """
    plt.style.use('default')
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get unique species
    unique_species = sorted(combined_df['species'].unique())
    
    # Prepare data for box plot
    species_data_list = []
    species_labels = []
    
    for species in unique_species:
        species_pecep = combined_df[combined_df['species'] == species]['PECEP']
        species_data_list.append(species_pecep)
        species_labels.append(f'Species {species}')
    
    # Create box plot with custom styling
    box_plot = ax.boxplot(species_data_list, 
                         labels=species_labels, 
                         patch_artist=True,
                         medianprops={'color': 'orange', 'linewidth': 2})
    
    # Color all boxes steel blue
    for patch in box_plot['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.8)
    
    # Make other elements more visible
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_plot[element], color='black')
    
    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('PECEP Score', fontsize=12)
    ax.set_title('PECEP Score Distribution by Species', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'exp2_vis2_fig4.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # Create and display summary statistics table
    print("\n" + "="*80)
    print("PECEP SCORE SUMMARY STATISTICS BY SPECIES")
    print("="*80)
    
    summary_stats = []
    for species in unique_species:
        species_data = combined_df[combined_df['species'] == species]['PECEP']
        q1 = species_data.quantile(0.25)
        q3 = species_data.quantile(0.75)
        iqr = q3 - q1
        
        stats = {
            'Species': species,
            'Count': len(species_data),
            'Mean': f"{species_data.mean():.4f}",
            'Median': f"{species_data.median():.4f}",
            'Std': f"{species_data.std():.4f}",
            'Min': f"{species_data.min():.4f}",
            'Q1': f"{q1:.4f}",
            'Q3': f"{q3:.4f}",
            'IQR': f"{iqr:.4f}",
            'Max': f"{species_data.max():.4f}"
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Print the table with nice formatting
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Save summary table if output directory is specified
    if output_dir:
        summary_file = os.path.join(output_dir, 'pecep_summary_statistics_all.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to: {summary_file}")
    
    # Print overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total utterances: {len(combined_df)}")
    print(f"Number of species: {len(unique_species)}")
    print(f"Species included: {unique_species}")
    all_pecep = combined_df['PECEP']
    overall_q1 = all_pecep.quantile(0.25)
    overall_q3 = all_pecep.quantile(0.75)
    overall_iqr = overall_q3 - overall_q1
    print(f"Overall PECEP - Mean: {all_pecep.mean():.4f}, Median: {all_pecep.median():.4f}")
    print(f"Overall PECEP - Std: {all_pecep.std():.4f}, IQR: {overall_iqr:.4f}")
    print(f"Overall PECEP - Range: [{all_pecep.min():.4f}, {all_pecep.max():.4f}]")

def main():
    """Main function to collect data and create plots."""
    # Update these paths to match your setup
    #model_dir = "/home/jacob/data/jacob/experiments/izs_exp_2"
    model_dir = "models"
    output_dir = "figures"  # Optional: where to save plots
    
    try:
        # Collect all PECEP data
        print("Collecting PECEP data from experiment files...")
        combined_df = collect_pecep_data(model_dir)
        
        # Create visualizations
        print("\nCreating box plot visualization...")
        create_pecep_boxplot(combined_df, output_dir)
        
        # Optionally save the combined data
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            combined_csv = os.path.join(output_dir, 'combined_pecep_data_all.csv')
            combined_df.to_csv(combined_csv, index=False)
            print(f"Combined data saved to: {combined_csv}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()