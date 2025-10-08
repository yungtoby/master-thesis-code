import torch as t

def sin_func(x):
    return t.sin(x) * x / 2

def weird_func(x):
    return - ((x - 5)**4) / 100 + 0.05 * x * (x*x)/5 * t.sin(x)