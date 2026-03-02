import torch as t
from base import BaseProblemFamily


class ToyRBFProblemFamily(BaseProblemFamily):
    '''
    Class mimicking a problem family with different parameters for
    each lane in a batch. Works by sampling a family of related 
    optimization problems.
    '''

    def __init__(self, device, dtype, lb, ub, M=3, min_cost=1):
        '''
        Initialize the problem with an upper (ub) and lower bound (lb)
        to the problem. Aswell as the total number components (M) in the
        toy objective and a minimum cost (min_cost).
        '''
        self.device = device
        self.dtype = dtype

        self.lb = t.tensor(lb, device=device, dtype=dtype)
        self.ub = t.tensor(ub, device=device, dtype=dtype)

        self.M = M
        self.min_cost = min_cost


    def sample_params(self, B, seed=None):
        '''Sample parameters for B lanes'''
        # Set seed for reproducibility
        if seed is not None:
            g = t.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None
        
        # Get dimensionality of domain
        d = self.ub.numel()
        
        # Sample params for each lanes objective function
        objective_func_params = self._sample_objective_params(g, B, d)

        # Sample params for each lanes cost functions
        cost_func_params = self._sample_cost_params(g, B, d)

        return {**objective_func_params, **cost_func_params}


    def _sample_objective_params(self, generator, B, d):
        '''Sample the objective function params for each lane B'''
        # ----------------------------------------------------------
        # Current TOY example objective with RBF function w/ M peaks
        # ----------------------------------------------------------
            # Where is each bump / peak located 
        centers = self.lb + (self.ub - self.lb) * t.rand((B, self.M, d), device=self.device, dtype=self.dtype, generator=generator)

            # Height of each bump / peak
        amps = t.rand((B, self.M), device=self.device, dtype=self.dtype, generator=generator) + 0.5

            # Width of each bump / peak
        lengthscale = 0.3 * (self.ub - self.lb) * (t.rand((B, self.M, d), device=self.device, dtype=self.dtype, generator=generator) + 0.2)

        return {'centers' : centers, 'amps' : amps, 'lengthscale' : lengthscale}


    def _sample_cost_params(self, generator, B, d):
        '''Sample the cost function params for each lane B'''
        # ----------------------------------------------------------
        # Current TOY example cost function: smooth surface per lane
        # ----------------------------------------------------------
            # Cheap random center, costs increase away from p (via squared distance)
        p = self.lb + (self.ub - self.lb) * t.rand((B, d), device=self.device, dtype=self.dtype, generator=generator)

            # Strength of the quadratic distance term
        alpha = t.rand((B, 1), device=self.device, dtype=self.dtype, generator=generator) + 0.1

            # Linear slope term
        beta = t.randn((B, d), device=self.device, dtype=self.dtype, generator=generator) + 0.1

            # Shifts overall costs up or down
        bias = t.randn((B, 1), device=self.device, dtype=self.dtype, generator=generator) + 0.1

        return {'p' : p, 'alpha' : alpha, 'beta' : beta, 'bias': bias}


    def evaluate(self, X, params):
        '''
        Evaluate the toy objective: sum of M Gaussian/RBF bumps per lane.
        X: [B, N, d]  or [B, 1, d] --> returns [B, N, 1]
        '''
        # Unpack params
        centers = params["centers"]        # [B, M, d]
        amps = params["amps"]              # [B, M]
        lengthscale = params["lengthscale"]  # [B, M, d]

        # Broadcasting setup:
        # X            : [B, N, d]      -> [B, 1, N, d]
        # centers      : [B, M, d]      -> [B, M, 1, d]
        # lengthscale  : [B, M, d]      -> [B, M, 1, d]
        X_exp = X.unsqueeze(1)                 # [B, 1, N, d]
        C_exp = centers.unsqueeze(2)           # [B, M, 1, d]
        L_exp = lengthscale.unsqueeze(2)       # [B, M, 1, d]

        # Normalized squared distance for each bump m at each candidate n
        # ((x - mu) / ell)^2 summed over d
        z2 = ((X_exp - C_exp) / (L_exp + 1e-12)).pow(2).sum(dim=-1)  # [B, M, N]

        # Gaussian bump values: exp(-0.5 * z2)
        bumps = t.exp(-0.5 * z2)  # [B, M, N]

        # Weight by amplitudes and sum over bumps:
        # amps: [B, M] -> [B, M, 1] for broadcast over N
        y = (amps.unsqueeze(-1) * bumps).sum(dim=1)  # [B, N]

        return y.unsqueeze(-1)  # [B, N, 1]


    def costs(self, X, params):
        """
        Compute a positive, smooth cost surface per lane.

        We use:
          raw = alpha * ||x - p||^2 + beta^T x + bias
          cost = min_cost + softplus(raw)

        X:      [B, N, d]
        returns [B, N]
        """
        p = params["p"]          # [B, d]
        alpha = params["alpha"]  # [B, 1]
        beta = params["beta"]    # [B, d]
        bias = params["bias"]    # [B, 1]

        # ||x - p||^2:
        # X: [B, N, d]
        # p: [B, d] -> [B, 1, d] to broadcast over N
        diff = X - p.unsqueeze(1)                 # [B, N, d]
        dist2 = (diff ** 2).sum(dim=-1)           # [B, N]

        # beta^T x:
        # beta: [B, d] -> [B, 1, d] broadcast over N
        lin = (X * beta.unsqueeze(1)).sum(dim=-1) # [B, N]

        # raw: [B, N]
        raw = alpha * dist2 + lin + bias          # alpha/bias broadcast to [B, N]

        # softplus keeps it positive but smooth; add min_cost baseline
        cost = self.min_cost + t.nn.functional.softplus(raw)  # [B, N]

        return cost


    def optimal_value_on_grid(self, X, params):
        '''Evaluate the largest value with respect to X and params'''
        y = self.evaluate(X, params).squeeze(-1) # [B, N]
        return y.max(dim=1).values