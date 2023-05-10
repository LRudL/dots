import torch as t
import torch.utils.data as tdata
from dots.utils import range_batch

N_DEFAULT = 1000
SEED_DEFAULT = 0

def algorithmic_dataset(fn, start=-1, end=1, N=N_DEFAULT, seed=SEED_DEFAULT):
    """N is the total number of data points.
    start and end can also be tensors of some dimensionality"""
    t.manual_seed(seed)
    X = range_batch(start, end, N)
    Y = fn(X)
    dataset = tdata.TensorDataset(X, Y)
    return dataset

def get_dataset(name):
    match name:
        case "mnist":
            import torchvision
            import torchvision.transforms as transforms
            mnist = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))]
                )
            )
            return mnist
        case "relu":
            return algorithmic_dataset(t.nn.ReLU(), -1, 1, N_DEFAULT)
        case "sin":
            return algorithmic_dataset(t.sin, -1, 1, N_DEFAULT) 
        case _:
            raise ValueError(f"Unknown dataset name: {name}")