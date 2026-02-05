import torch as t
from itertools import product



class CandidateSet:
    '''Candidate set to perform optimization over.'''
    def __init__(self, device, dtype, res, D, minimum, maximum):
        self.device = t.device(device)
        self.dtype = dtype

        # Number of candidate points and dimensionality
        self.res = res          # Number of points per axes
        self.D = D              # Number of axes' to include
        self.K = res**D         # Total number of candidate points
        self.minimum = minimum  # Minimum (i.e, start of resolution)
        self.maximum = maximum  # Maximum (i.e, end of resolution)

        # Initialize uniformally
        self.grid = self.create_uniform_grid()
        self.cost_grid = self.create_cost_grid()
        self.used_mask = t.zeros(self.K, dtype=t.bool, device=self.device)


    def create_uniform_grid(self):
        '''Function for creating the grid'''
        
        if not isinstance(self.res, int) or self.res <= 0:
            raise ValueError('resolution must be a positive integer')

        # Initialize the axes'
        axes = [t.linspace(self.minimum, self.maximum, self.res,
                           device=self.device, dtype=self.dtype) for _ in range(self.D)]

        # Stack the axes'
        mesh = t.stack(t.meshgrid(*axes, indexing='ij'), dim=-1)

        # Reshape to correct shape
        grid = mesh.reshape(-1, self.D)  # shape (K, D)
        return grid
    

    def create_cost_grid(self):
        '''Function for creating costs of the grid'''
        cost_grid = t.randint(
            low=1,
            high=101,
            size=(self.K,),
            device=self.device,
            dtype=t.int64
        )

        return cost_grid.to(dtype=t.float32)
    
    
    def mark_as_visited(self, idx):
        self.used_mask[idx] = True

    def get_mask(self):
        return self.used_mask

    def get_grid(self):
        return self.grid
    
    def get_cost_grid(self):
        return self.cost_grid