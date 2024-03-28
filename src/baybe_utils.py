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

from src import schwefel

import numpy as np

from tqdm import tqdm


def run_optimization_campaign(
    NUM_ITERATIONS,
    NUM_INIT_OBS,
    N_DIMS_SCHWEF,
    NOISE_LEVEL_SCHWEF,
    ITERATION_BATCH_SIZE,
    recommender_init = None,
    recommender_main = None
):
    """
    Utility function for running a bayesian optimization campaign.

    NUM_ITERATIONS: Number of bayesian optimization iterations to run for
    NUM_INIT_OBS: number of random initial observations to make before starting BO
    N_DIMS_SCWEF: number of x dimensions of schwefel function to optimize
    NOISE_LEVEL_SCHWEF: variance of noise added to schwefel function
    ITERATION_BATCH_SIZE: number of observations to make per BO batch
    reccomender_init: (BayBE Reccomender): recommender to use for initial sampling. Default random
    recommender_main: (BayBE reccomender): recommender to use for main BO loop. Default baybe sequential greedy with EI

    """
    # Define Schweffel oracle
    schweffer = schwefel.SchwefelProblem(n_var = N_DIMS_SCHWEF, noise_level=NOISE_LEVEL_SCHWEF)
    target = NumericalTarget(name = 'schwefel', mode = "MIN")
    parameters = [
        NumericalContinuousParameter(f'schwefel{i+1}', bounds = (-500,500)) for i in range(N_DIMS_SCHWEF)
    ]
    
    objective = Objective(mode = "SINGLE", targets = [target])
    searchspace = SearchSpace.from_product(parameters)
    
    if recommender_init is None:
       recommender_init = RandomRecommender()
    if recommender_main is None:
        recommender_main = SequentialGreedyRecommender(acquisition_function_cls='EI')

    print("Collecting initial observations")
    campaign_init = Campaign(searchspace, objective, recommender_init)
    random_params = campaign_init.recommend(NUM_INIT_OBS)
    
    y_init = schweffer.f(random_params.to_numpy())
    
    random_params.insert(N_DIMS_SCHWEF, 'schwefel', y_init)
    
    optimization_campaign = Campaign(searchspace, objective, recommender_main)
    optimization_campaign.add_measurements(random_params)

    print('Beginning optimization campaign')
    for i in tqdm(range(NUM_ITERATIONS)):
        reccs = optimization_campaign.recommend(ITERATION_BATCH_SIZE)
    
        y_vals = schweffer.f(reccs.to_numpy())
    
        reccs.insert(N_DIMS_SCHWEF, 'schwefel', y_vals)
    
        optimization_campaign.add_measurements(reccs)

    return optimization_campaign



def iteration_noise_grid_search(iterations_list, noise_list, NUM_INIT_OBS, N_DIMS_SCHWEF, ITERATION_BATCH_SIZE):
    """
    Utility to run a parameter grid experiment varying noise level and number of BO iterations. Runs full grid in serial.
    Params:
    -------
    iteraitons_list: list of ints - number of BO iterations to run
    noise_list - list of floats - noise values to run
    NUM_INIT_JOBS - int
    N_DIMS_SCWEF - int
    ITERATION_BATCH_SIZE - int

    returns:
    ---------
    iteration_results: an abomination of dictionaries. Outer level: results keyed by number of iterations; next layer: results keyed by noise level. Values are BayBE campaign objects of completed campaign
    """
    iteration_results = {}
    
    for its in iterations_list:
        noise_results = {}
        for noise_level in noise_list:
            opt_campaign = run_optimization_campaign(its, NUM_INIT_OBS, N_DIMS_SCHWEF, noise_level, ITERATION_BATCH_SIZE)
            noise_results[str(noise_level)] = opt_campaign
        iteration_results[str(its)] = noise_results

    return iteration_results

def process_grid_searh_results(grid_search_results):
    """
    Process results from iteration_noise_grid_search function to extract a performance matrix of best result for each campaign

    Params:
    --------
    grid_search_results: dict of dicts of BayBE campaigns

    Returns:
    -------
    n_its - list - iteration numbers used
    n_noise - list - noise values used
    performance_matrix: matrix of best observed values (min schwefel val) for each campaign from grid search, arranged with iteration varying on axis 0 and noise on axis 1
    """
    
    
    n_its = len(grid_search_results)
    n_noise = len(grid_search_results[list(grid_search_results.keys())[0]])
    
    performance_matrix = np.zeros((n_its, n_noise))
    
    
    
    # fill out performance matrix
    iteration_vals = []
    noise_vals = []
    
    first_pass = True
    for i, (its, entry) in enumerate(grid_search_results.items()):
        iteration_vals.append(its)
        for j, (noise, camp) in enumerate(entry.items()):
            if first_pass:
                noise_vals.append(noise)
            best_result = camp.measurements['schwefel'].min()
            # hack to flip matrix BO iterations upside down for plotting 
            performance_matrix[n_its - 1 - i, j] = best_result
        first_pass = False
    
    iteration_vals = iteration_vals[::-1]

    return iteration_vals, noise_vals, performance_matrix

