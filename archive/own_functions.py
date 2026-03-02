import torch as t

def sin_func(x):
    return (t.sin(x) * x / 2).to(x.device)

def weird_func(x):
    return - ((x - 5)**4) / 100 + 0.05 * x * (x*x)/5 * t.sin(x).to(x.device)

def weird_func_2(x):
    # x: (K, D) -> returns (K,) on same device
    return (t.sin(x.sum(dim=-1)) * (x.sum(dim=-1) / 2.0)).to(x.device)


def not_too_easy_unique_opt(x):
    # x: (K, D) -> returns (K,) on same device
    s = x.sum(dim=-1)

    # main peak: smooth, unique maximum at s = pi with value 1
    peak = 1.0 - (s - t.pi)**2

    # strictly non-positive perturbation with a *unique* zero at s = pi
    # (sin(s) is zero at many points, but the (s - pi)^2 factor makes it zero only at s = pi)
    wiggle_penalty = -0.15 * (s - t.pi)**2 * t.sin(3.0 * s)**2

    return peak + wiggle_penalty