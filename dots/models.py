import numpy as np
import einops
import torch.nn.functional as F
from dots.dots import JModule
from dots.utils import flatten, get_device
import torch as t
import torch.nn as nn
from collections.abc import Sequence

class MLP(JModule):
    def __init__(
        self,
        in_size,
        out_size,
        hidden = 0,
        hidden_size = None,
        nonlinearity = t.nn.ReLU,
        bias = True
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        if hidden_size == None and isinstance(hidden, int):
            hidden_size = in_size
        hidden_sizes_vect = [hidden_size for _ in range(hidden)] if isinstance(hidden, int) else hidden
        assert isinstance(hidden_sizes_vect, Sequence), f"hidden must be a list or tuple, was {type(hidden)}: {hidden}"
        sizes = [in_size] + hidden_sizes_vect + [out_size]
        layers = flatten(
            [[t.nn.Linear(sizes[i], sizes[i+1], bias=bias), nonlinearity()]
             for i in range(len(sizes)-2)]) + [t.nn.Linear(sizes[-2], sizes[-1], bias=bias)]
        self.layers = t.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class DeepLinear(MLP):
    def __init__(
        self,
        in_size,
        out_size,
        hidden = 0,
        hidden_size = None,
        bias = True
    ):
        super().__init__(
            in_size,
            out_size,
            hidden,
            hidden_size,
            nonlinearity=t.nn.Identity,
            bias=bias
        )


class MNIST_MLP(MLP):
    def __init__(
        self,
        in_size = 784,
        out_size = 10,
        hidden = 0,
        hidden_size = None,
        nonlinearity = t.nn.ReLU,
        bias = True
    ):
        super().__init__(
            in_size,
            out_size,
            hidden,
            hidden_size,
            nonlinearity,
            bias
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return super().forward(x) 

class BasicCNN(JModule):
    def __init__(
        self,
        in_size = (1, 28, 28), # MNIST
        fc_bias = True,
        **kwargs
    ):
        # currently hard_coded to work only with MNIST sizes
        
        super().__init__()
        self.in_size = in_size
        self.layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2),
            
            t.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc = t.nn.Linear(8 * 5 * 5, 10, bias=fc_bias)
        
    
    def forward(self, x):
        out = self.layers(x)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        
        return out


## transformers:
## (based on code in grokking.ipynb)

class Embed(JModule):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(JModule):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)


class PosEmbed(JModule):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]


class LayerNorm(JModule):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


class Attention(JModule):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.mask = t.tril(t.ones((n_ctx, n_ctx))).to(get_device())
        self.d_head = d_head
        
    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked/np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class TransformerMLP(JModule):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class TransformerBlock(JModule):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = TransformerMLP(d_model, d_mlp, act_type, model=self.model)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp((x))
        return x


class Transformer(JModule):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_ln=True):
        super().__init__()

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x[:, [-1]]
