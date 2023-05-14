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

def random_batch(n, shape, start=-1, end=1, move_to_device=True):
    if isinstance(shape, int):
        shape = (shape,)
    tensor = t.rand((n,) + shape)
    # tensor is now distributed [0, 1]; we want [start, end]
    tensor = tensor * (end - start) + start
    if move_to_device == True:
        return tensor.to(get_device())
    elif move_to_device == False:
        return tensor
    else:
        return tensor.to(move_to_device)

def slightly_perturb(tensor, std=0.01):
    return tensor + t.randn(tensor.shape) * std

def perturbed_copies(tensor, n, std=0.01):
    tensors = [slightly_perturb(tensor, std=std) for _ in range(n)]
    return t.stack(tensors)

def with_changed_pixel(tensor, change_amount=0.01):
    new_tensor = tensor.clone()
    random_index = tuple(t.randint(0, dim_size, (1,)).item() for dim_size in tensor.shape)
    new_tensor[random_index] += change_amount
    return new_tensor

def dataset_from_end(dataset, length):
    return tdata.Subset(dataset, range(len(dataset)-length, len(dataset))) 

def tensor_of_dataset(dataset, indices=None):
    if indices == None:
        indices = range(len(dataset))
    subset_ds = tdata.Subset(dataset, indices)
    subset_dl = tdata.DataLoader(subset_ds, batch_size=len(subset_ds))
    return next(iter(subset_dl))[0].to(get_device())

def class_examples(dataset, class_index, n):
    found = []
    while len(found) < n:
        for i, (_, label) in enumerate(dataset):
            if label == class_index:
                found.append(i)
    return tensor_of_dataset(dataset, found[:n])

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

def load_model(name):
    return t.load(
        "../models/" + name + ".pt",
        map_location=get_device()
    )

def accuracy(model, dataloader_or_dataset):
    if isinstance(dataloader_or_dataset, t.utils.data.Dataset):
        dataloader = t.utils.data.DataLoader(dataloader_or_dataset)
    else:
        dataloader = dataloader_or_dataset
    n = 0
    correct = 0
    for batch, label in dataloader:
        batch = batch.to(get_device())
        label = label.to(get_device())
        out = model(batch).argmax(dim=-1)
        correct += (out==label).sum()
        n += batch.shape[0]
    return correct / n