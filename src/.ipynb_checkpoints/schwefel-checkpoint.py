import numpy as np

class SchwefelProblem:
    def __init__(self, n_var=1, noise_level=0.01, range = (-50, 50)):
        """
        y = f(x) + eps
        """
        self.noise_level = noise_level
        self.n_var = n_var  # Number of variables/dimensions
        self.bounds = np.array([[range[0]] * self.n_var, [range[1]] * self.n_var])

    def _schwefel_individual(self, x):
        return x * np.sin(np.sqrt(np.abs(x)))

    def f(self, x):
        return 418.9829 * self.n_var - np.sum(self._schwefel_individual(x), axis=1) 

    def eps(self, x):
        # Assuming the noise is independent of x for simplicity
        return np.random.normal(0, self.noise_level, x.shape[0])

    def y(self, x):
        f = self.f(x)
        eps = self.eps(x)
        return f + eps

# Test code if this file is the main program being run
if __name__ == "__main__":
    # Create a SchwefelProblem instance with 3 variables/dimensions
    schwefel = SchwefelProblem(n_var=3, noise_level=1.)
    x_test = np.array([[420, 420, 420], [420, 420, 420]])  # Example input vector
    print("Objective function value (f):", schwefel.f(x_test))
    print("Noisy objective function value (y):", schwefel.y(x_test))
    
