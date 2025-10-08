import torch as t
from itertools import product

class CandidateSet:
    def __init__(self, res, D, minimum, maximum):
        # Number of candidate points and dimensionality
        self.res = res
        self.D = D
        self.K = res**D
        self.minimum = minimum
        self.maximum = maximum

        # Initialize uniformally
        self.grid = self.uniform_grid()

    def uniform_grid(self):
        if not isinstance(self.res, int) or self.res <= 0:
            raise ValueError('resolution must be a positive integer')

        # 1-D coordinates for each dimension
        axes = [t.linspace(self.minimum, self.maximum, self.res) for _ in range(self.D)]
        
        # Cartesian product -> list of tuples, then stack
        grid = t.tensor(list(product(*axes)))   # shape (K, D)
        return grid

    def get_grid(self):
        return self.grid