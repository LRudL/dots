import torch as t
import torch.utils.data as tdata
import math
import matplotlib.pyplot as plt
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

def range_batch(start, end, n, move_to_device=True):
    if not isinstance(start, t.Tensor):
        start = t.tensor([start])
        end = t.tensor([end])
    tensor = t.rand((n,) + start.shape) * (end - start) + start
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

def plot_1d_fn(fn, start=-1, end=1, n=100):
    x = t.linspace(start, end, n)
    x = rearrange(x, "n -> n 1")
    y = fn(x).detach().cpu().numpy()
    plt.plot(x, y)
    plt.show()

def plot_1d_u_feats(x, model):
    x_sorted_indices = np.argsort(x.squeeze())  # Sort the indices of x in ascending order
    x_sorted = x.squeeze()[x_sorted_indices]  # Sort the flattened x values
    x_sorted_batch = rearrange(x_sorted, "n -> n 1")
    U_T = model.u_features(x_sorted_batch).detach().cpu().numpy().T
    # U now has size [rank, N] with x_sorted order;
    # we want to plot each row of U as a function of x_sorted
    fig, ax = plt.subplots()
    singular_values = model.jacobian_singular_values(
        x_sorted_batch
    ).detach().cpu().numpy()
    max_singular_value = singular_values.max()
    for i in range(U_T.shape[0]):
        U_T_sorted = U_T[i, x_sorted_indices]  # Sort U_T[i] according to x_sorted order
        ax.plot(x_sorted, U_T_sorted, alpha=singular_values[i] / max_singular_value)
    ax.plot(
        x_sorted,
        model(x_sorted_batch).detach().cpu().numpy(),
        color="black",
        linestyle="dotted"
    )
    ax.set_title("U features, with current function drawn in dotted black")

def plot_dots_stats(model, inputs):
    dots_getters = [
        lambda m, x : m.jacobian_matrix_rank(x),
        lambda m, x : m.singular_value_rank(x, method="entropy"),
        lambda m, x : m.singular_value_rank(x, method="heuristic"),
        lambda m, x : min(
            m.count_params(),
            np.prod(np.array(x.shape))
        )
    ]
    getter_names = [
        "Jacobian rank",
        "SV rank, entropy",
        "SV rank, heuristic threshold",
        "Maximum rank"
    ]
    x = np.arange(len(inputs))
    for i, (getter_name, getter) in enumerate(zip(getter_names, dots_getters)):
        y = []
        for input in inputs:
            new = getter(model, input)
            if is_tensor(new):
                new = new.cpu()
            y.append(new)
        print(f"{getter_name} ranks: {y}")
        plt.bar(x - 0.2 + 0.2 * i, y, 0.2, label=getter_name)
    print(f"Parameters in model: {model.count_params()}")
    plt.legend()
    plt.xticks(x, [input.shape[0] for input in inputs])
    plt.xlabel("Data points")
    plt.show()
    
    # Visualise singular value number vs singular value
    fig, ax = plt.subplots()
    for input in inputs:
        ax.plot(
            model.jacobian_singular_values(input).cpu(),
            label=f"{input.shape[0]} data points"
        )
    ax.set_title("Singular value distribution")
    ax.legend()
    ax.set_yscale("log")
    fig.show()
    
    # Visualise matrices
    matrices = [
        model.matrix_jacobian(input).cpu() for input in inputs
    ]
    fig, axs = plt.subplots(len(matrices))
    if len(matrices) == 1:
        axs = [axs]
    for matrix, ax in zip(matrices, axs):
        ax.imshow(matrix, aspect="auto")
        ax.set_title(f"{matrix.shape[0]} outputs")
    # plot the color map legend:
    # https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    fig.colorbar(plt.cm.ScalarMappable(), ax=axs, orientation="horizontal")
    fig.show()
    
    fig2, axs2 = plt.subplots(len(matrices))
    if len(matrices) == 1:
        axs2 = [axs2]
    # Visualise parameter importances
    for ax2, matrix in zip(axs2, matrices):
        importances = (matrix ** 2).mean(dim=0).numpy()
        ax2.hist(importances, bins=100, log=True)
    fig2.show()
    