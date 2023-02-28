from dots.dots import JModule
from dots.utils import flatten, get_device
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

class DeepLinear(MLP):
    def __init__(
        self,
        in_size,
        out_size,
        hidden=0,
        hidden_size = None
    ):
        super().__init__(
            in_size,
            out_size,
            hidden,
            hidden_size,
            nonlinearity=t.nn.Identity
        )

class BasicCNN(JModule):
    def __init__(
        self,
        in_size = (28, 28) # MNIST
    ):
        # currently hard_coded to work only with MNIST sizes
        
        super().__init__()
        self.layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2),
            
            t.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc = t.nn.Linear(8 * 5 * 5, 10)
        
    
    def forward(self, x):
        out = self.layers(x)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        
        return out
