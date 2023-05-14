from dots.utils import *
import matplotlib.pyplot as plt

default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_1d_fn(fn, start=-1, end=1, n=1000, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x = t.linspace(start, end, n)
    x = rearrange(x, "n -> n 1")
    y = fn(x).detach().cpu().numpy()
    ax.plot(x, y)
    if given_ax is None:
        fig.show()

def plot_1d_fn_from_data(fn, x, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    sorted_x = t.sort(x.squeeze())[0].unsqueeze(1)
    y = fn(sorted_x).detach().cpu().numpy()
    ax.scatter(sorted_x.squeeze(), y, s=1)
    if given_ax is None:
        fig.show()

def plot_1d_u_feats(
    model, 
    x,
    max_feats=None,
    which_feat=None,
    ax=None
):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x_sorted_indices = np.argsort(x.squeeze())  # Sort the indices of x in ascending order
    x_sorted = x.squeeze()[x_sorted_indices]  # Sort the flattened x values
    x_sorted_batch = rearrange(x_sorted, "n -> n 1")
    U_T = model.u_features(x_sorted_batch).detach().cpu().T
    # U_T now has size [rank, N] with x_sorted order;
    # we want to plot each row of U as a function of x_sorted
    singular_values = model.jacobian_singular_values(
        x_sorted_batch
    ).detach().cpu()
    assert singular_values.shape[0] == U_T.shape[0] # both are the rank
    assert U_T.shape[1] == x_sorted.shape[0] # both are the number of points
    rank = U_T.shape[0]
    N = U_T.shape[1]
    max_singular_value = singular_values.max()
    #U_T_sorted = t.index_select(U_T.T, 0, t.tensor(x_sorted_indices)).T
    U_T_sorted = U_T
    for i in range(U_T.shape[0]):
        if (which_feat is None and (max_feats is None or i < max_feats)) or i == which_feat:
            #U_T_sorted = U_T[i, x_sorted_indices]  # Sort U_T[i] according to x_sorted order
            Ui_sorted = U_T_sorted[i]
            if singular_values[i] / max_singular_value > 0.01:
                ax.plot(
                    x_sorted,
                    Ui_sorted,
                    alpha=math.sqrt(singular_values[i] / max_singular_value)
            )
    ax2 = ax.twinx()
    #ax2.plot(
    #    x_sorted,
    #    model(x_sorted_batch).detach().cpu().numpy(),
    #    color="black",
    #    linestyle="dotted"
    #)
    #ax.set_title("U features, with current function drawn in dotted black")
    #average_change = average_U(U_T_sorted.T, singular_values)
    #assert average_change.shape[0] == x_sorted.shape[0]
    #ax2.plot(
    #    x_sorted,
    #    average_change,
    #    color="black",
    #    linestyle="dotted"
    #)
    #ax.set_title("U features")
    if given_ax is None:
        fig.show()

def plot_u_feats_img(start, end, n, model, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x = range_batch(start, end, n)    
    U_T = model.u_features(x).detach().cpu().numpy().T
    # U now has size [rank, N] with x_sorted order;
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(U_T, cmap="gray", aspect='auto') 
    if given_ax is None:
        fig.show()


def plot_dots_estimates(model, inputs, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
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
        "SV (heuristic)",
        "SV (entropy)",
        "max"
    ]
    # TODO log scale?
    if isinstance(inputs, t.Tensor):
        inputs = [inputs]
    x = np.arange(len(inputs))
    for i, (getter_name, getter) in enumerate(zip(getter_names, dots_getters)):
        y = []
        x_loc = x - 0.2 + 0.2 * i
        for input in inputs:
            new = getter(model, input)
            if is_tensor(new):
                new = new.cpu().item()
            y.append(new)
        ax.bar(x_loc, y, 0.2, label=getter_name)
        for i, value in enumerate(y):
            # display to 2 decimal places:
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            ax.text(x_loc, value, text, ha='center', va='bottom')
    ax.legend()
    ax.set_xticks(x, [input.shape[0] for input in inputs])
    ax.set_xlabel("Data points")
 
def plot_singular_value_distribution(model, inputs, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    if isinstance(inputs, t.Tensor):
        inputs = [inputs]
    for input in inputs:
        ax.plot(
            model.jacobian_singular_values(input).cpu(),
            label=f"{input.shape[0]} data points"
        )
    ax.set_title("Singular value distribution")
    ax.legend()
    ax.set_yscale("log")
    if given_ax is None:
        fig.show()


def plot_jacobian_img(model, x, ax=None, fig=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    matrix = model.matrix_jacobian(x).detach().cpu().numpy()
    ax.imshow(matrix, aspect="auto")
    ax.set_title(f"Jacobian, {matrix.shape[0]} outputs")
    if fig is not None:
        fig.colorbar(plt.cm.ScalarMappable(), ax=ax, orientation="horizontal")
    if given_ax is None:
        fig.show()


def trainplot_1d_regression_over_axes(model, x, ax, fig):
    # plot dots estimates:
    plot_dots_estimates(model, x, ax[0])
    
    # plot singular value distribution:
    plot_singular_value_distribution(model, x, ax[1])
    
    # plot the Jacobian:
    plot_jacobian_img(model, x, ax[2], fig)
    
    # plot the function:
    plot_1d_fn_from_data(model, x, ax[3])
    
    # plot the U features:
    plot_1d_u_feats(model, x, ax[4])
    

def trainplot_1d(trainstate, x1=None, x2=None):
    """Plots relevant information about the model at a particular point
    during training. It is assumed the model is a 1D to 1D model where the
    interesting stuff happens in the range [-1, 1]
    If x1 and x2 are none, will plot statistics for x1 in the first column and
    for a random range in the second.
    If both are supplied, will plot both of them"""
    model = trainstate.model
    figsize = (8, 20)
    if x2 == None and x1 is not None:
        x2 = range_batch(-1, 1, 1000) 
    if x1 == None:
        x1 = range_batch(-1, 1, 1000) 
        fig, axs = plt.subplots(5, 1, figsize=figsize)
        trainplot_1d_regression_over_axes(model, x1, axs, fig)
    fig, axs = plt.subplots(5, 2, figsize=figsize)
    trainplot_1d_regression_over_axes(model, x1, axs[:, 0], fig)
    trainplot_1d_regression_over_axes(model, x2, axs[:, 1], fig)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.95)
    fig.suptitle(f"Epoch {trainstate.epochs} step {trainstate.steps}")
    fig.show()
   
def plot_dots_stats(model, inputs):
    # DEPRECATED
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
    