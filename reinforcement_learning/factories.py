import torch as t
from gaussian_process import GPWrapper

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
