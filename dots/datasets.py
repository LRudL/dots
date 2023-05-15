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

def gen_cluster(mean, std, n):
    points = t.empty((n, 2))
    for i in range(n):
        points[i] = t.normal(mean=mean, std=std)
    return points


def get_dataset(name, seed=SEED_DEFAULT):
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
                        t.tensor(-1.).to(get_device()),
                        t.tensor(1.).to(get_device()))
            return algorithmic_dataset(square, -1, 1, N_DEFAULT)
        case "noise":
            def noise(x):
                return t.rand_like(x)
            return algorithmic_dataset(noise, -1, 1, N_DEFAULT)
        case "twoclasses":
            t.manual_seed(seed)
            N = 128
            mean1 = t.tensor([0.5, 0.5])
            mean2 = t.tensor([-0.5, -0.5])
            mean3 = t.tensor([0.5, -0.5])
            mean4 = t.tensor([-0.5, 0.5])
            std1 = t.tensor([0.1, 0.1])
            std2 = t.tensor([0.1, 0.1])
            std3 = t.tensor([0.1, 0.1])
            std4 = t.tensor([0.1, 0.1])
            cluster1 = gen_cluster(mean1, std1, N // 4)
            cluster2 = gen_cluster(mean2, std2, N // 4)
            cluster3 = gen_cluster(mean3, std3, N // 4)
            cluster4 = gen_cluster(mean4, std4, N // 4)
            X = t.cat((cluster1, cluster2, cluster3, cluster4))
            Y = t.cat(
                (t.zeros(N // 4).long(),
                 t.zeros(N // 4).long(),
                 t.ones(N // 4).long(),
                 t.ones(N // 4).long()))
            shuffled_idx = t.randperm(N)
            X = X[shuffled_idx]
            Y = Y[shuffled_idx]
            return tdata.TensorDataset(X, Y)
        case "randtwoclasses":
            t.manual_seed(seed)
            N = 20
            d = 1.0
            std = 0.3
            mean1 = t.tensor([d, d])
            mean2 = t.tensor([-d, -d])
            mean3 = t.tensor([d, -d])
            mean4 = t.tensor([-d, d])
            std1 = t.tensor([std, std])
            std2 = t.tensor([std, std])
            std3 = t.tensor([std, std])
            std4 = t.tensor([std, std])
            cluster1 = gen_cluster(mean1, std1, N // 4)
            cluster2 = gen_cluster(mean2, std2, N // 4)
            cluster3 = gen_cluster(mean3, std3, N // 4)
            cluster4 = gen_cluster(mean4, std4, N // 4)
            X = t.cat((cluster1, cluster2, cluster3, cluster4))[t.randperm(N)]
            Y = t.cat(
                (t.zeros(N // 2).long(),
                    t.ones(N // 2).long()))
            shuffled_idx = t.randperm(N)
            X = X[shuffled_idx]
            Y = Y[shuffled_idx]
            return tdata.TensorDataset(X, Y)
        case _:
            raise ValueError(f"Unknown dataset name: {name}")
