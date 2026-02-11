import torch as t
import gpytorch as gpy
from gaussian_process import GPWrapper
from candidate_set import CandidateSet

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
    

#class KernelFactory:
#    def __init__(self, device: str = "cpu", dtype = t.float32):
#        self.device = device
#        self.dtype = dtype
#
#    def __call__(self, kernels, addition: bool, batch_size, ard_num_dims, num_mixtures):
#        if len(kernels) > 1:
#            
#            for kernel in kernels:
#                    if len(kernel)
#                    kernel(batch_size = batch_size, ard_num_dims=ard_num_dims)
#
#        return gpy.kernels.ScaleKernel()