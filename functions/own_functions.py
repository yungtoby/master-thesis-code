import torch as t

def sin_func(x):
    return (t.sin(x) * x / 2).to(x.device)

def weird_func(x):
    return - ((x - 5)**4) / 100 + 0.05 * x * (x*x)/5 * t.sin(x).to(x.device)

def weird_func_2(x):
    # x: (K, D) -> returns (K,) on same device
    return (t.sin(x.sum(dim=-1)) * (x.sum(dim=-1) / 2.0)).to(x.device)