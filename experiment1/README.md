# Experiment 1: Verification of Variance Bounds with Known Conditional Entropy

This directory contains the code used to reproduce the results for **Experiment 1** in the paper *“Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy.”*

---

## Result Reproduction Workflow

1. **Generate Synthetic Data**  
   Run `experiment1_data_generation.ipynb` to create the autoregressive (AR) synthetic datasets.  
   This script produces datasets for multiple trials and noise variance levels.

2. **Visualize Synthetic Data and Coefficients (Figure 1)**  
   Run `experiment1_vis1_fig1.ipynb` to reproduce **Figure 1** from the paper.  
   - The notebook additionally includes a panel showing mean-squared error between ground truth and prediction, which was omitted from the paper for space.  
   - It also visualizes the true and estimated coefficient matrices, which were also omitted from the final paper version.

3. **Compute Prediction Error Conditional Entropy Proxy (PECEP) Scores**  
   Run `experiment1_prediction.py` to compute all PECEP scores for all variance levels and trials using both the **Oracle** and **Ordinary Least Squares (OLS)** predictors.  
   Note: This script assumes that the synthetic data has already been generated in Step 1.

4. **Generate Figure 3 (PECEP Convergence Plot)**  
   Run `exp1_vis2_fig3.py` to create **Figure 3** from the paper.  
   This figure shows how both solvers converge to the true conditional differential entropy lower bound as the dataset size increases.

---

## Additional Files

- **`utils.py`** – Contains helper functions used throughout the experiment scripts.  
- **`oracle_data_scarce_setting_sanity_check.py`** – Auxiliary script for a quick experiment mentioned (but not shown) in the paper, examining the Oracle’s consistent underestimation of conditional entropy in the **small-sample (200 data points)** regime.

---

## Notes

- The experiment can be run on CPU for smaller dataset sizes; however, GPU acceleration is recommended for faster processing of large trials.  
- All output figures and computed results will be stored in their respective `results/` and `figures/` which the scripts automatically create.