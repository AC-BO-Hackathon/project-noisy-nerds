# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from botorch.models.gp_regression import (
    SingleTaskGP,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
import os
import gc



from baybe.targets import NumericalTarget
from baybe.objective import Objective
from baybe.parameters import (
    NumericalContinuousParameter
)

from baybe.recommenders import (
    SequentialGreedyRecommender,
    RandomRecommender
)

from baybe.searchspace import SearchSpace
from baybe import Campaign
from baybe import simulation

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# SMOKE_TEST = True
print("SMOKE_TEST", SMOKE_TEST)
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16
batch_size = 1



# %%
from botorch.utils.sampling import draw_sobol_samples
from src.schwefel import SchwefelProblem
from time import time

def run_experiment(n_init, noise_level, budget, seed, noise_bool):

    N_DIMS_SCHWEF = 2
    ITERATION_BATCH_SIZE = 1
        

    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = SchwefelProblem(n_var=N_DIMS_SCHWEF, noise_level=noise_level)

    bounds = torch.tensor(problem.bounds, **tkwargs)

    target = NumericalTarget(name = 'schwefel', mode = "MIN")
    parameters = [
        NumericalContinuousParameter(f'schwefel{i+1}', bounds = (-50,50)) for i in range(N_DIMS_SCHWEF)
    ]
    
    objective = Objective(mode = "SINGLE", targets = [target])
    searchspace = SearchSpace.from_product(parameters)
    
    if recommender_init is None:
       recommender_init = RandomRecommender()
    if recommender_main is None:
        recommender_main = SequentialGreedyRecommender(acquisition_function_cls='EI')


    print("Collecting initial observations")
    campaign_init = Campaign(searchspace, objective, recommender_init)
    random_params = campaign_init.recommend(n_init)
    
    y_init = problem.y(random_params.to_numpy())
    y_init_real = problem.f(random_params.to_numpy())
    
    random_params.insert(N_DIMS_SCHWEF, 'schwefel', y_init)
    
    optimization_campaign = Campaign(searchspace, objective, recommender_main)
    optimization_campaign.add_measurements(random_params)

    y_real = []
    print('Beginning optimization campaign')
    for i in range(budget):
        reccs = optimization_campaign.recommend(ITERATION_BATCH_SIZE)
    
        y_vals = problem.y(reccs.to_numpy())
        y_real = problem.f(reccs.to_numpy())
    
        reccs.insert(N_DIMS_SCHWEF, 'schwefel', y_vals)
    
        optimization_campaign.add_measurements(reccs)

    measurements = optimization_campaign.measurements

    # get X and noisy y values
    x_names = [f'schwefel{i+1}' for i in range(N_DIMS_SCHWEF)]
    x_train = measurements[x_names].to_numpy()
    y_train = measurements['schwefel'].to_numpy()

    # compile noise-free ground truth vals
    y_real_complete = np.zeros(len(y_init_real) + len(y_real))

    for i, val in enumerate(y_init_real):
        y_real_complete[i] = val

    for i, val in enumerate(y_real):
        y_real_complete[i+len(y_init_real)] = val
    
    os.makedirs('results', exist_ok=True)
    fname = f"results/{problem.__class__.__name__[:5]}_n_init_{n_init}_noiselvl_{noise_level}_budget_{budget}_seed_{seed}_noise_{noise_bool}.pt"
    torch.save((train_X, train_Y, train_Y_real, model), fname)

    train_X = torch.from_numpy(x_train)
    train_Y = torch.from_numpy(y_train)
    train_Y_real = torch.from_numpy(y_real_complete)
    
    return train_X, train_Y, train_Y_real, None

        
        
if __name__ == "__main__":
    run_experiment(5, 0.1, 5, 0, True)
    run_experiment(5, 0.1, 5, 0, False)