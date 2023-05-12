from dots.utils import *
import matplotlib.pyplot as plt


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
        ax.plot(
            x_sorted,
            U_T_sorted,
            alpha=max(0.05, math.sqrt(singular_values[i] / max_singular_value)))
    ax2 = ax.twinx()
    ax2.plot(
        x_sorted,
        model(x_sorted_batch).detach().cpu().numpy(),
        color="black",
        linestyle="dotted"
    )
    ax.set_title("U features, with current function drawn in dotted black")
    fig.show()


def plot_u_feats_img(start, end, n, model):
    x = range_batch(start, end, n)    
    U_T = model.u_features(x).detach().cpu().numpy().T
    # U now has size [rank, N] with x_sorted order;
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(U_T, cmap="gray", aspect='auto') 
    fig.show()


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
    