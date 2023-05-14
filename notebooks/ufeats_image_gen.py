import sys
sys.path.append("..")
import os
import math
import numpy as np
import torch as t
import torch.utils.data as tdata
import matplotlib.pyplot as plt
from einops import rearrange
from dots.training import *
from dots.trainhooks import *
from dots.models import MLP
from dots.dots import *
from dots.utils import *
from dots.plotting import *
from dots.experiment import get_train_state, get_config_dataset

plt.ioff()

ts = get_train_state("../configs/models/mlp-relu-good.yml")

for i in range(31):
    print("\n\n\n\n\n")
    print("Starting: ", i)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for _ in range(10):
        plot_1d_u_feats(ts.model, range_batch(-1, 1, 500, sort=True), ax=axs[0], which_feat=i)
        plot_1d_u_feats(ts.model, range_batch(-1, 1, 5000, sort=True), ax=axs[1], which_feat=i)
    fig.savefig(f"imgs/ufeats_{i}.png")
    print("Finished ", i)
    print("\n\n\n\n\n")