# Experiment 2: Complexity Ranking of Bio-Inspired Synthetic Audio Data

This directory contains the code used to reproduce the results for Experiment 2 in the paper “Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy.”

---

## Result Reproduction Workflow

1. **Generate Synthetic Data**  
   Run `experiment2_data_generation.py` to create the artificial “species” datasets described in Section III-B (1) of the paper.
   Each dataset corresponds to one synthetic species, whose vocalizations exhibit systematically increasing temporal and spectral complexity.

2. **Adjust Audio Annotations**  
   Run `annotation_adjustment.py` to ensure that the automatically generated call annotations and audio metadata are properly aligned with the format expected by the training and testing scripts.
   This step standardizes metadata consistency across all generated species.

3. **Train Neural Network Predictors**  
   Run `model_training.py` to train a fully connected neural network (FCN) for each species dataset.
   - Training follows the architecture and procedure described in Section III-B (2).
   - The script automatically scans the data/ directory for available species folders and trains a separate model for each.
   - Note: If multiple datasets are present, the script will train a model for each dataset found—ensure you have only the desired datasets in data/ before launching training.

4. **Visualize Example Spectrograms and Corresponding Predictions (Figure 2)**  
   Run `species_example_predictions.py` to reproduce Figure 2 from the paper.
   This figure compares the ground-truth spectrograms and the neural network predictions for Species 0 (simplest) and Species 9 (most complex), along with the residuals.
   The generated `.png` file is automatically saved to the `figures/` folder.

5. **Compute Utterance-Level PECEP Scores**  
   Run collect_utterance_PECEP_scores.py to compute the **Prediction Error Conditional Entropy Proxy (PECEP)** point estimates for each vocalization utterance.
   These scores correspond to the quantities described in the Experiment 2 Methods section of the paper.

6. **Generate Boxplots of PECEP Scores (Figure 4)**  
   Run `pecep_boxplots.py` to reproduce **Figure 4** from the paper.
   This visualization shows how the median utterance-level PECEP scores increase monotonically with the known complexity ordering of the synthetic species.

---

## Additional Files

All other scripts in this directory provide support functions used by the experiment:

- `BioInspiredAudioGenerator.py` – Class that encapsulates the bioinspired audio generation process that is used in `experiment2_data_generation.py`.

- `SequentialSpectrogramPredictionDataset.py`, `data_setup.py` – Dataset class for preparing spectrogram frame sequences for training and evaluation. Also some general helper functions for torch model compatibility

- `models.py`, `criterions.py`, `optimizers.py`, `schedulers.py`, `train.py`, `validate.py`, 'tests.py' – Define network architectures, loss functions, optimization algorithms, training utilities, and our defined testing framework.

- `visualization_tools.py` – Helper routines for plotting and figure generation.

- `experiment_runner.py` – Functions that enable conducting multiple experiments, used in `model_training.py`

---

## Notes

- All generated data, results, and figures are saved in the automatically created `data/`, `results/`, and `figures/` directories.

- The neural network and PECEP computation pipelines assume that spectrograms have already been pre-normalized to the [0, 1] range.

- Training and visualization can be computationally intensive; GPU acceleration is recommended if available.