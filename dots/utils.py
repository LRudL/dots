import torch as t
import math
import matplotlib.pyplot as plt
import numpy as np

flatten = lambda l : [item for sl in l for item in sl]

def entropy(x, base=math.e):
    if isinstance(x, list) or isinstance(x, tuple):
        x = t.tensor(x)
    x /= x.sum()
    return - (x * t.log(x) / t.log(t.tensor(base))).sum()

def get_device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

def random_batch(n, shape):
    if isinstance(shape, int):
        shape = (shape,)
    return t.rand((n,) + shape)

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
            y.append(getter(model, input))
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
            model.jacobian_singular_values(input),
            label=f"{input.shape[0]} data points"
        )
    ax.set_title("Singular value distribution")
    ax.legend()
    ax.set_yscale("log")
    fig.show()
    
    # Visualise matrices
    matrices = [
        model.matrix_jacobian(input) for input in inputs
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
    