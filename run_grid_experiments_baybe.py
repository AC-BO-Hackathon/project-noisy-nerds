#import ray
import argparse
from time import time, sleep
from run_experiment_baybe import run_experiment
from datetime import datetime
import gc

MAX_NUM_PENDING_TASKS = 12


#@ray.remote
def worker(n_init, noise_level, budget, seed, noise_bool, bounds):

    try:
        run_experiment(n_init, noise_level, budget, seed, noise_bool,bounds)
        # saved file looks like this: results\Schwe_n_init_6_noiselvl_0_budget_0_seed_2_noise_False.pt
    except Exception as e:
        print(e)
        print(f'problem {n_init} noise {noise_level} budget {budget} seed {seed} failed')
        return 1
        
    return 0

def run_grid_experiments(seeds, n_inits, noise_levels, noise_bools, budget, bounds):
    
    # ray.init(local_mode=True)
    #ray.init(ignore_reinit_error=True)
    start_time = time()
    tasks = []
    
    for seed in seeds:
        for n_init in n_inits:
            for noise_level in noise_levels:
                for noise_bool in noise_bools:
                    #if len(tasks) > MAX_NUM_PENDING_TASKS:
                    #    completed_tasks, tasks = ray.wait(tasks, num_returns=1)
                    #    ray.get(completed_tasks[0])

                    #sleep(1)
                    task = worker(n_init, noise_level, budget, seed, noise_bool, bounds)
                    tasks.append(task)
                    print(f'Started problem {n_init} noise {noise_level} budget {budget} seed {seed}, time: {time() - start_time:.2f}s')
                    #gc.collect()
    
   # while len(tasks) > 0:
   #     completed_tasks, tasks = ray.wait(tasks, num_returns=1)
   #     print(ray.get(completed_tasks[0]))
    

    print('all experiments done, time: %.2fs' % (time() - start_time))

if __name__ == "__main__":
    
    seeds = [0]
    n_inits = [2, 4, 6 ,8, 10]
    noise_levels = [0, 0.01, 0.1, 0.5]
    # budgets = [10, 20, 50]
    noise_bools = [True, False]
    budget = 10
    run_grid_experiments(seeds, n_inits, noise_levels, noise_bools, budget)
