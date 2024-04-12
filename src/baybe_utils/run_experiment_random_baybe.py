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

def run_experiment(n_init, noise_level, budget, seed, noise_bool, bounds, fp):
    """
    Run a bayesian optimization campaign on the 2-dimensional
    schwefel function using the specified parameters. Uses BayBE with the
    SequentialGreedyRecommender with Expected improvement acquisition function.

    :param n_init: Number of randomly selected initial trials to run
    :type n_init: int
    :param noise_level: Variance of Gaussian noise to add to scwhefel function values
    :type noise_level: float
    :param budget: Number of optimization trials to run (in addition to n_init)
    :type budget: int
    :param seed: Random seed
    :type seed: int
    :param noise_bool: Artifact from Botorch implementation, does nothing
    :type noise_bool: bool
    :return train_X: Scwhefel X values evaluated
    :type train_X: Tensor
    :return train_Y: The Schwefel function values associated with train_X points, including noise
    :type train_Y: Tensor
    :return train_Y_real: The 'true' noise-free Y values
    :type train_Y_real: Tensor 
    """

    N_DIMS_SCHWEF = 2
    ITERATION_BATCH_SIZE = 1
        

    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = SchwefelProblem(n_var=N_DIMS_SCHWEF, noise_level=noise_level, range = bounds)

    #bounds = torch.tensor(problem.bounds, **tkwargs)

    target = NumericalTarget(name = 'schwefel', mode = "MIN")
    parameters = [
        NumericalContinuousParameter(f'schwefel{i+1}', bounds = bounds) for i in range(N_DIMS_SCHWEF)
    ]
    
    objective = Objective(mode = "SINGLE", targets = [target])
    searchspace = SearchSpace.from_product(parameters)
    
  
    recommender_init = RandomRecommender()
    #recommender_main = SequentialGreedyRecommender(acquisition_function_cls='EI')


    print("Collecting random observations observations")
    campaign_init = Campaign(searchspace, objective, recommender_init)
    random_params = campaign_init.recommend(n_init + budget)
    
    y_init = problem.y(random_params.to_numpy())
    y_init_real = problem.f(random_params.to_numpy())
    
    random_params.insert(N_DIMS_SCHWEF, 'schwefel', y_init)
    
    #optimization_campaign = Campaign(searchspace, objective, recommender_main)
    campaign_init.add_measurements(random_params)

    #y_real = []
    #print('Beginning optimization campaign')
    #for i in range(budget):
    #    reccs = optimization_campaign.recommend(ITERATION_BATCH_SIZE)
   # 
   #     y_vals = problem.y(reccs.to_numpy())
   #     y_real.append(problem.f(reccs.to_numpy()))
   # 
   #     reccs.insert(N_DIMS_SCHWEF, 'schwefel', y_vals)
   # 
   #     optimization_campaign.add_measurements(reccs)

    measurements = campaign_init.measurements

    # get X and noisy y values
    x_names = [f'schwefel{i+1}' for i in range(N_DIMS_SCHWEF)]
    x_train = measurements[x_names].to_numpy()
    y_train = measurements['schwefel'].to_numpy()

    # compile noise-free ground truth vals
    y_real_complete = y_init_real

    #for i, val in enumerate(y_init_real):
    #    y_real_complete[i] = val#

    #for i, val in enumerate(y_real):
    #    y_real_complete[i+len(y_init_real)] = val
    


    train_X = torch.from_numpy(x_train)
    train_Y = torch.from_numpy(y_train)
    train_Y_real = torch.from_numpy(y_real_complete)

    os.makedirs(fp, exist_ok=True)
    fname = f"{fp}/{problem.__class__.__name__[:5]}_n_init_{n_init}_noiselvl_{noise_level}_budget_{budget}_seed_{seed}_noise_{noise_bool}.pt"
    torch.save((train_X, train_Y, train_Y_real, None), fname)
    
    return train_X, train_Y, train_Y_real, None

        
        
if __name__ == "__main__":
    run_experiment(5, 0.1, 5, 0, True)
    run_experiment(5, 0.1, 5, 0, False)