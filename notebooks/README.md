# DAPRO Notebooks

This directory contains Jupyter Notebooks for executing DAPRO pipelines, estimating metrics, and visualizing results. 

## End-to-End Demos
- **[dapro_lpb_demo.ipynb](dapro_lpb_demo.ipynb)**: An interactive, minimal walkthrough of the DAPRO pipeline. It demonstrates loading synthetic survival data, running DAPRO for budget allocation, and calibrating the Lower Probability Bound (LPB).
- **[dapro_metric_estimation_demo.ipynb](dapro_metric_estimation_demo.ipynb)**: An interactive demonstration focused on unbiased metric estimation. It utilizes DAPRO allocators and the `MetricsEngine` to simulate trajectories and estimate critical safety metrics (e.g., Cumulative Jailbreak Rate, Restricted Mean Time To Unsafe).

## Visualization Notebooks
- **[plot_intro_figure.ipynb](plot_intro_figure.ipynb)**: Generates the introductory figure comparing the expected censoring times of our dynamic projection method against the static baseline.
- **[summarize_results.ipynb](summarize_results.ipynb)**: Consolidates LPB and UPB (Upper Probability Bound) coverage results, generating combined boxplots across various settings like AutoIF, Toxicity, and Red-teaming.
- **[visualize_coverage_bound_results.ipynb](visualize_coverage_bound_results.ipynb)**: Visualizes and compares the theoretically guaranteed lower coverage bounds of our proposed method versus bounds from previous literature.
- **[visualize_lpb_upb_metrics.ipynb](visualize_lpb_upb_metrics.ipynb)**: Processes both LPB and UPB coverage results, mapping raw dataset runs to clear LLM display names, and creates detailed boxplots and bar charts.

## Additional Helpers
- **[common.py](common.py)** & **[visualize_ablation_results.py](visualize_ablation_results.py)**: Helper scripts and visualizers used by the notebooks for executing robust pipelines.
