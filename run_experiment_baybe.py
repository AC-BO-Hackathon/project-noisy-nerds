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
def generate_initial_data(problem, n: int, bounds: torch.Tensor) -> tuple:
    X_init = draw_sobol_samples(
        bounds=bounds, n=n, q=1, seed=torch.randint(100000, (1,)).item()
    ).squeeze(-1)
    Y_init = torch.tensor(problem.y(X_init.numpy()))
    Y_init_real = torch.tensor(problem.f(X_init.numpy()))
    return X_init, Y_init, Y_init_real

# %%
def initialize_model(train_x, train_y, noise_bool=True) -> tuple:
    # define models for objective and constraint
    train_y= -train_y  # negative because botorch assumes maximization

    if noise_bool:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y.unsqueeze(-1),
            train_Yvar=torch.full_like(train_y.unsqueeze(-1), 1e-6),
            outcome_transform=Standardize(m=1),
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model

# %%
def optimize_acqf_loop(problem, acq_func):

    standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
    standard_bounds[1] = 1
    options = {"batch_limit": batch_size, "maxiter": 2000}

    while options["batch_limit"] >= 1:
        try:
            torch.cuda.empty_cache()
            x_cand, acq_val = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options=options,
            )
            torch.cuda.empty_cache()
            gc.collect()
            break
        except RuntimeError as e:
            if options["batch_limit"] > 1:
                print(
                    "Got a RuntimeError in `optimize_acqf`. "
                    "Trying with reduced `batch_limit`."
                )
                options["batch_limit"] //= 2
                continue
            else:
                raise e

    return x_cand, acq_val


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

    y_real = np.
    
    
    os.makedirs('results', exist_ok=True)
    fname = f"results/{problem.__class__.__name__[:5]}_n_init_{n_init}_noiselvl_{noise_level}_budget_{budget}_seed_{seed}_noise_{noise_bool}.pt"
    torch.save((train_X, train_Y, train_Y_real, model), fname)
    
    return train_X, train_Y, train_Y_real, model
        
        
if __name__ == "__main__":
    run_experiment(5, 0.1, 5, 0, True)
    run_experiment(5, 0.1, 5, 0, False)