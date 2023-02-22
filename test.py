# if this file runs, imports etc. are probably working right

import math
import numpy as np
import torch as t
import torch.utils.data as tdata
import matplotlib.pyplot as plt
from einops import rearrange

from dots.training import *
from dots.models import MLP
from dots.dots import *

import torchvision
import torchvision.transforms as transforms

from dots.models import BasicCNN

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
mnist = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_mnist, test_mnist, valid_mnist = tdata.random_split(
    mnist,
    lengths=[0.8, 0.1, 0.1]
)

mnist_train_loader = tdata.DataLoader(
    train_mnist,
    batch_size=16,
    shuffle=True,
    num_workers=1
)

mnist_test_loader = tdata.DataLoader(
    test_mnist,
    batch_size=16,
    shuffle=True,
    num_workers=1
)

mnist_valid_loader = tdata.DataLoader(
    test_mnist,
    batch_size=16,
    shuffle=True,
    num_workers=1
)

cnn = BasicCNN()

n_eps = 1
trainstate_m = TrainState(
    model=cnn,
    optimiser=t.optim.Adam(cnn.parameters(), lr=1e-3),
    loss_fn=t.nn.CrossEntropyLoss(),
    dataloader=mnist_valid_loader,
    test_loader=mnist_test_loader,
    hooks=[]
)

trainstate_m.train

def mnist_accuracy(model, dl=mnist_test_loader):
    n = 0
    correct = 0
    for batch, label in mnist_test_loader:
        out = model(batch).argmax(dim=-1)
        correct += (out==label).sum()
        n += batch.shape[0]
    return correct / n

print(mnist_accuracy(cnn))

t.save(trainstate_m.model, "test_model_file")