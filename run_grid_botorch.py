import torch
import pandas as pd
from run_grid_experiments import run_grid_experiments
from run_experiment import run_experiment


seeds = list(range(5))
n_inits = [2, 4, 8, 10]
noise_levels = [1, 5, 10, 20]
noise_bools = [True, False]
budget = 30


run_grid_experiments(seeds, n_inits, noise_levels, noise_bools, budget)


