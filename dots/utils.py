import torch as t
import torch.utils.data as tdata
import math
import numpy as np
from einops import rearrange

flatten = lambda l : [item for sl in l for item in sl]

def is_tensor(obj):
    return isinstance(obj, t.Tensor)

def entropy(x, base=math.e):
    if isinstance(x, list) or isinstance(x, tuple):
        x = t.tensor(x)
    x /= x.sum()
    return - (x * t.log(x) / t.log(t.tensor(base))).sum()

def get_device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

def range_batch(start, end, n, move_to_device=True):
    if not isinstance(start, t.Tensor):
        start = t.tensor([start])
        end = t.tensor([end])
    tensor = t.rand((n,) + start.shape) * (end - start) + start
    if move_to_device == True:
        return tensor.to(get_device())
    elif move_to_device == False:
        return tensor
    else:
        return tensor.to(move_to_device)

def random_batch(n, shape, move_to_device=True):
    if isinstance(shape, int):
        shape = (shape,)
    tensor = t.rand((n,) + shape)
    if move_to_device == True:
        return tensor.to(get_device())
    elif move_to_device == False:
        return tensor
    else:
        return tensor.to(move_to_device)

def tensor_of_dataset(dataset, indices=None):
    if indices == None:
        indices = range(len(dataset))
    subset_ds = tdata.Subset(dataset, indices)
    subset_dl = tdata.DataLoader(subset_ds, batch_size=len(subset_ds))
    return next(iter(subset_dl))[0].to(get_device())
