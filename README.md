# Project 23: Reliable Surrogate Models of Noisy Data 


This project was part of the 2024 Bayesian Optimization Hackathon for Chemistry and Materials (https://ac-bo-hackathon.github.io/). The goal of this project was to explore the impact of noisy measurements on the performance of various Bayesian optimizers. We explored this problem using both the BayBE package as well as BoTorch. The BoTorch version evaluated the impact of incorporating noise assumptions in the surrogate model, while the BayBE approach uses the default BayBE model which does incorporate some noise assumptions (?). We used the 2-dimensional Schwefel function as a minimization target. This function has a global minimum at ~[420, 420]. We mainly evaluated the optmimization methods in the range of [-50,50]. Over this range the Schwefel function has high frequency optimizations leading to many local minima, making this a challenging optimization problem. 

## Guide to Project:

1. BoTorch arm: The BoTorch half of the project can be viewed in the `analyse_grid_experiment.ipynb`, `botorch_results_plots.ipnb`', and `line_plot.ipynb` notebooks. Source code is in `run_experiment.py`. `run_grid_botorch.py`, and `run_grid_experiments.py`.

2. The BayBE half of the project can be viewed in the `noisy_optimization_BayBE_original_bounds.ipynb` (for [-50,50] bounds) and `noisy_optimization_BayBE_extendedBounds.ipynb` (for [0,500] bounds). Source code is in src/baybe_utils.

## Team

- Darby Brown
- Karim Ben Hicham
- Joe Manning
- Brenden Pelkie
- Utkarsh Pratiush
