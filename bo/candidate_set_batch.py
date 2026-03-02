import torch as t



class BatchedCandidateSet:
    '''Candidate set to perform optimization over.'''
    def __init__(self, device, dtype, B, res, D, minimum, maximum, jitter_frac = 0.35):
        self.device = t.device(device)
        self.dtype = dtype

        # Number of candidate points and dimensionality
        self.B = B                     # Number of batches
        self.res = res                 # Number of points per axes
        self.D = D                     # Number of axes' to include
        self.K = res**D                # Total number of candidate points
        self.minimum = minimum         # Minimum (i.e, start of resolution)
        self.maximum = maximum         # Maximum (i.e, end of resolution)
        self.jitter_frac = jitter_frac # Fraction for amount of jitter to each point

        # Initialize uniformally and determinstically
        base = self._create_uniform_grid()

        # Apply jitter and expand grid to [B, k, d]
        self.grid = self._make_batched_jitter(base)


    def _create_uniform_grid(self):
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


    def _make_batched_jitter(self, base_grid):
        # If resolution less than 1:
        if self.res <= 1:
            base_grid.unsqueeze(-1).expand(self.B, -1, -1)

        # Create uniformily sampled jitter between [-amp, amp]
        spacing = (self.maximum - self.minimum) / (self.res - 1)
        amp = self.jitter_frac * spacing
        jitter = (2 * t.rand((self.B, self.K, self.D), device=self.device, dtype=self.dtype) - 1) * amp

        # Create grid
        grid = base_grid.unsqueeze(0) + jitter
        grid = t.clamp(grid, min=self.minimum, max=self.maximum)

        return grid


    def get_grid(self):
        return self.grid