import torch as t
from gaussian_process import GPWrapper
from candidate_set import CandidateSet
from functions.blackbox_function import BlackBoxFunc
from functions.own_functions import weird_func

class GPFactory:
    '''Callable factory that creates independent GPWrapper instance'''
    def __init__(self, device: str = "cpu", dtype = t.float32):
        self.device = device
        self.dtype = dtype

    def __call__(self, **kwargs):
        # Create a new GPWrapper
        gp = GPWrapper(
            device=t.device(self.device),
            dtype=self.dtype,
            **kwargs
        )

        return gp


class CandidateFactory:
    '''Callable factory that creates independent CandidateSet instance'''
    def __init__(self, device: str = "cpu", dtype = t.float32):
        self.device = device
        self.dtype = dtype

    def __call__(self, **kwargs):
        # Create a new candidateset
        c_set = CandidateSet(
            device=t.device(self.device),
            dtype=self.dtype,
            **kwargs
        )

        return c_set.get_grid(), c_set.get_cost_grid()