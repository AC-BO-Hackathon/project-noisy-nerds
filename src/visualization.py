import seaborn as sn
import pandas as pd

def grid_search_heatmap(iterations_list, noise_list, performance_matrix):
    """
    plot a heatmap 

    Params:
    -------
    iterations_list: list of number of iterations used
    noise list: list of noise values used
    performance matrix: np array of dims len(iterations_list) x len(noise_list) with smallest noise, smallest iterations in lower left corner ('origin')

    returns:
    ------
    matplotlib ax object 
    """

    df_heatmap = pd.DataFrame(performance_matrix, index = [str(its) for its in iterations_list], columns = [str(noise) for noise in noise_list])
    
    ax = sn.heatmap(df_heatmap, annot=True, fmt = '.3g', cmap = 'crest')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Number of initial samples')

    return ax