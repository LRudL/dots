from dots.dots import JModule
from dots.utils import flatten
import torch as t


class MLP(JModule):
    def __init__(
        self, in_size, out_size,
        hidden = 0, hidden_size = None,
        nonlinearity=t.nn.ReLU
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        if hidden_size == None:
            hidden_size = in_size
        sizes = [in_size] + [hidden_size for _ in range(hidden)] + [out_size]
        layers = flatten(
            [[t.nn.Linear(sizes[i], sizes[i+1]), nonlinearity()]
             for i in range(len(sizes)-2)]) + [t.nn.Linear(sizes[-2], sizes[-1])]
        self.layers = t.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)