import numpy as np

def schwefel_1d(x):

    return 418.9829  - x * np.sin(np.sqrt(np.abs(x)))

def schwefel_nd(args):
    output = 0
    
    for dim in range(args):
        output += schwefel_1d(args[dim])

def add_gaussian_noise(signal, noise_level):

    return signal + np.random.normal(0, noise_level, 1)[0]

def schwefel_1d_with_noise(x, noise_level = 0.01):
    # Calculate the Schwefel function value

    schwefel_value = schwefel_1d(x)

    # Add Gaussian noise to the Schwefel function value

    noisy_schwefel_value = add_gaussian_noise(schwefel_value, noise_level)

    return noisy_schwefel_value

def schwefel_nd_with_noise(args, noise_level = 0.01):
    # Calculate the Schwefel function value

    schwefel_value = schwefel_nd(args)

    # Add Gaussian noise to the Schwefel function value

    noisy_schwefel_value = add_gaussian_noise(schwefel_value, noise_level)

    return noisy_schwefel_value