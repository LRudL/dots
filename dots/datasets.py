import torch as t
import torch.utils.data as tdata
from dots.utils import range_batch, get_device

N_DEFAULT = 1536
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
            def sin(x):
                return t.sin(t.pi * x)
            return algorithmic_dataset(sin, -1, 1, N_DEFAULT) 
        case "hfsin":
            def sin(x):
                return t.sin(t.pi * 8 * x)
            return algorithmic_dataset(sin, -1, 1, N_DEFAULT)
        case "square":
            def square(x):
                xp = 2 * x
                return t.where(
                        t.floor(xp) % 2 == 0,
                        t.tensor(-1).to(get_device()),
                        t.tensor(1).to(get_device()))
            return algorithmic_dataset(square, -1, 1, N_DEFAULT)
        case "noise":
            def noise(x):
                return t.rand_like(x)
            return algorithmic_dataset(noise, -1, 1, N_DEFAULT)
        case _:
            raise ValueError(f"Unknown dataset name: {name}")
