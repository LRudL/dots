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

def range_batch(start, end, n, move_to_device=True, sort=False):
    if not isinstance(start, t.Tensor):
        start = t.tensor([start])
        end = t.tensor([end])
    tensor = t.rand((n,) + start.shape) * (end - start) + start
    if sort:
        tensor = t.sort(tensor, dim=0)[0]
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

def average_U(U, s):
    """ U is the U matrix (of dimensions [N, rank])
    s is the singular values (diagonal of the S matrix), a 1d tensor
    of dimensions [rank].
    This function returns a 1d tensor of dimensions [N] representing
    the averaged U vector (averaged over the squares of the singular values).
    """
    assert U.shape[1] == len(s)
    U_weighted = U * (s[None, :])**2
    assert U_weighted.shape == U.shape
    data_sums = t.einsum("nr -> n", U_weighted)
    return data_sums
     