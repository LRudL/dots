import torch as t
import math

flatten = lambda l : [item for sl in l for item in sl]

def entropy(x, base=math.e):
    if isinstance(x, list) or isinstance(x, tuple):
        x = t.tensor(x)
    x /= x.sum()
    return - (x * t.log(x) / t.log(t.tensor(base))).sum()

def get_device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")