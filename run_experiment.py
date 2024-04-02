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
    ).squeeze(1)
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
        

    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = SchwefelProblem(n_var=2, noise_level=noise_level)

    bounds = torch.tensor(problem.bounds, **tkwargs)

    train_X, train_Y, train_Y_real= generate_initial_data(problem, n_init, bounds)

    start_time = time()

    for i in range(budget):
        print(f"Starting iteration {i}, total time: {time() - start_time:.3f} seconds.")
        
        train_x = normalize(train_X, bounds)
        mll, model = initialize_model(train_x, train_Y, noise_bool)
        fit_gpytorch_model(mll)
        
        # optimize the acquisition function and get the observations
        X_baseline = train_x
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            prune_baseline=True,
            sampler=sampler,
        )

        x_cand, acq_func_val = optimize_acqf_loop(problem, acq_func)
        X_cand = unnormalize(x_cand, bounds)
        Y_cand = torch.tensor(problem.y(X_cand.numpy()))
        Y_cand_real = torch.tensor(problem.f(X_cand.numpy()))
        print(f"New candidate: {X_cand}, {Y_cand}")

        # update the model with new observations
        train_X = torch.cat([train_X, X_cand], dim=0)
        train_Y = torch.cat([train_Y, Y_cand], dim=0)
        train_Y_real = torch.cat([train_Y_real, Y_cand_real], dim=0)        
        
    train_x = normalize(train_X, bounds)
    mll, model = initialize_model(train_x, train_Y, noise_bool)
    fit_gpytorch_model(mll)
    
    os.makedirs('results_botorch', exist_ok=True)
    fname = f"results_botorch/{problem.__class__.__name__[:5]}_n_init_{n_init}_noiselvl_{noise_level}_budget_{budget}_seed_{seed}_noise_{noise_bool}.pt"
    torch.save((train_X, train_Y, train_Y_real, model), fname)
    
    return train_X, train_Y, train_Y_real, model
        
        
if __name__ == "__main__":
    run_experiment(5, 0.1, 5, 0, True)
    run_experiment(5, 0.1, 5, 0, False)