import torch as t
from gaussian_process import GPWrapper
from candidate_set import CandidateSet
from functions.blackbox_function import BlackBoxFunc
from functions.own_functions import weird_func

class GPFactory:
    '''Callable factory that creates independent GPWrapper instance'''
    def __init__(self, device: str = "cpu", dtype = t.float32, **kwargs):
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs # store the rest of the keyword args to forward to GPWrapper

    def __call__(self):
        # Create a new GPWrapper instance with stored kwargs
        gp = GPWrapper(
            device=t.device(self.device),
            dtype=self.dtype,
            **self.kwargs
        )
        return gp


class CandidateFactory:
    '''Callable factory that creates independent CandidateSet instance'''
    def __init__(self, device: str = "cpu", dtype = t.float32, **kwargs):
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs # store the rest of the keyword args to forward to CandidateSet

    def __call__(self, num_candidates):
        # Create a new candidateset
        c_set = CandidateSet(
            device=t.device(self.device),
            dtype=self.dtype,
            **self.kwargs
        )

        # TODO: Fix costs, now just random between 1 and 101.
        costs = t.randint(1, 101, (num_candidates,))

        return c_set.get_grid(), costs