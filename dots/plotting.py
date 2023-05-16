from dots.utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors

default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def integer_histogram(x, ax=None, xlabel=""):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    n_bins = max(x) - min(x) + 1
    counts, bin_edges = np.histogram(x, bins=n_bins, range=(min(x), max(x)+1))
    ax.bar(bin_edges[:-1], counts, align="center", width=1)
    ax.set_xticks(bin_edges[:-1])
    ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    if given_ax is None:
        plt.show()


def plot_1d_fn(fn, start=-1, end=1, n=1000, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x = t.linspace(start, end, n)
    x = rearrange(x, "n -> n 1")
    y = fn(x).detach().cpu().numpy()
    ax.plot(x.detach().cpu().numpy(), y)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("1d function plot")
    if given_ax is None:
        fig.show()

def plot_1d_fn_from_data(fn, x, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    sorted_x = t.sort(x.squeeze())[0].unsqueeze(1)
    y = fn(sorted_x).detach().cpu().numpy()
    ax.scatter(sorted_x.squeeze().detach().cpu().numpy(), y, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("1d function plot")
    if given_ax is None:
        fig.show()

def plot_1d_u_feats(
    model, 
    x,
    max_feats=None,
    which_feat=None,
    avoid_flip=False,
    ax=None
):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x_sorted_indices = np.argsort(x.squeeze().cpu())  # Sort the indices of x in ascending order
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
            if not avoid_flip:
                if Ui_sorted[-1] < 0:
                    Ui_sorted = -Ui_sorted
            if singular_values[i] / max_singular_value > 0.01:
                ax.plot(
                    x_sorted.detach().cpu().numpy(),
                    Ui_sorted.detach().cpu().numpy(),
                    alpha=math.sqrt(singular_values[i] / max_singular_value) if which_feat is None else 1
            )
    ax.set_xlabel("x")
    ax.set_ylabel("U feature value")
    ax.set_title("U features")
    #ax2 = ax.twinx()
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


def plot_2d_classification_u_feats(
    model, 
    x,
    y,
    max_feats=None,
    which_feat=None,
    avoid_flip=False,
    C=10, # number of classes
    ax=None
):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    x_got_squeezed = False
    if len(x.shape) == 4:
        # then assume we got an MNIST shape of [batch, 1, 28, 28]
        x = x.squeeze(dim=1)
        x_got_squeezed = True
    assert len(x.shape) == 3, "x should be of shape [batch, height, width] or [batch, 1, height, width]"
    batch, height, width = x.shape
    y_sorted, y_sorted_indices = t.sort(y)
    x_sorted = x[y_sorted_indices]
    x_sorted_batch = rearrange(x_sorted, "n h w -> n 1 h w") if x_got_squeezed else x_sorted
    U_T = model.u_features(x_sorted_batch).detach().cpu().T
    # U_T now has size [rank, nC], where n = batch, C = number of classes in output 
    # we want to plot each row of U as a function of x_sorted
    singular_values = model.jacobian_singular_values(
        x_sorted_batch
    ).detach().cpu()
    rank = U_T.shape[0]
    N = U_T.shape[1] # = nC
    assert singular_values.shape[0] == rank, f"shape of singular_values is {singular_values.shape}, but rank is {rank}; should be equal!"
    assert N == x_sorted.shape[0] * C, f"N is {N} and x_sorted.shape[0] * C is {x_sorted.shape[0]}*{C} = {C * x_sorted.shape[0]}; should be equal!"
    max_singular_value = singular_values.max()
    for i in range(U_T.shape[0]):
        if (which_feat is None and (max_feats is None or i < max_feats)) or i == which_feat:
            Ui_sorted = U_T[i]
            if not avoid_flip:
                if Ui_sorted[-1] < 0:
                    Ui_sorted = -Ui_sorted
            # reshape from [nC] to [n, C]:
            Ui_sorted = rearrange(Ui_sorted, "(n C) -> n C", n=batch)
            if singular_values[i] / max_singular_value > 0.01:
                ax.imshow(
                    Ui_sorted.detach().cpu().numpy(),
                    cmap="seismic",
                    aspect=0.1,
                    norm=colors.CenteredNorm(vcenter=0.0)
                )
                c_start_indices = list(change_indices(y_sorted)) + [y_sorted.shape[0]]
                for i, ic in enumerate(c_start_indices[:-1]):
                    c = y_sorted[ic].item()
                    avg_loc = (ic + c_start_indices[i + 1]) // 2
                    ax.text(
                        C, 
                        avg_loc, 
                        str(c)
                    )
                    ax.axhline(y=ic, color="blue", linestyle="dotted")
    ax.set_title("U feature")
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

def plot_2d_points(X, Y):
    unique_labels = t.unique(Y)  # Get unique labels

    # Plot the points with different colors based on labels
    for i, label in enumerate(unique_labels):
        label_points = X[Y == label]  # Filter points for each label
        plt.scatter(
            label_points[:, 0],
            label_points[:, 1],
            label=f'Label {label}',
            s=1
        )

    # Add legend and labels
    plt.legend()

    # Display the plot
    plt.show()


def plot_decision_boundary(model, X, y, resolution=0.02):
    device = next(model.parameters()).device
    
    # Set min and max values for the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a meshgrid of points in the feature space
    xx, yy = t.meshgrid(t.arange(x_min, x_max, resolution),
                        t.arange(y_min, y_max, resolution))
    xx, yy = xx.to(device), yy.to(device)

    # Flatten the grid points and pass them through the model
    grid_points = t.stack((xx.flatten(), yy.flatten()), dim=1)
    with t.no_grad():
        Z = model(grid_points)
    Z = Z.argmax(dim=1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    # Create a contour plot of the decision boundary
    plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), Z, alpha=0.4)

    # Plot the training examples with different colors for each class
    unique_classes = t.unique(y)
    for class_label in unique_classes:
        class_points = X[y == class_label.item()]
        plt.scatter(
            class_points[:, 0],
            class_points[:, 1],
            label=f'Class {class_label.item()}',
            s=4
        )

    # Add labels and legend
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Show the plot
    plt.show()



def plot_dots_estimates(model, inputs, ax=None):
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    dots_getters = [
        lambda m, x : m.jacobian_matrix_rank(x),
        lambda m, x : m.singular_value_rank(x, method="entropy"),
        lambda m, x : m.singular_value_rank(x, method="trim"),
        lambda m, x : min(
            m.count_params(),
            np.prod(np.array(x.shape))
        )
    ]
    getter_names = [
        "Jacobian rank",
        "SV (entropy)",
        "SV (trim)",
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
    num_values = 0
    for input in inputs:
        values = model.jacobian_singular_values(input).cpu()
        num_values = max(num_values, values.shape[0])
        ax.plot(
            values, 
            label=f"{input.shape[0]} data points",
            linestyle="None",
            marker="."
        )
    for i in range(0, num_values, 5):
        ax.axvline(i, color="black", alpha=0.1)
    for i in range(0, num_values, 10):
        ax.axvline(i, color="black", alpha=0.2)
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
    plot_singular_value_distribution(model, x, ax=ax[1])
    
    # plot the Jacobian:
    plot_jacobian_img(model, x, ax=ax[2], fig=fig)
    
    # plot the function:
    plot_1d_fn_from_data(model, x, ax=ax[3])
    
    # plot the U features:
    plot_1d_u_feats(model, x, ax=ax[4])
    

def trainplot_1d(trainstate, x1=None, *x2etc):
    """Plots relevant information about the model at a particular point
    during training. It is assumed the model is a 1D to 1D model where the
    interesting stuff happens in the range [-1, 1]
    If x1 and x2 are none, will plot statistics for x1 in the first column and
    for a random range in the second.
    If both are supplied, will plot both of them"""
    if hasattr(trainstate, "model"):
        # not doing it the proper way (checking isinstance) because
        # this would cause a circular import:
        # training -> plotting -> trainhooks -> training
        model = trainstate.model
        got_model = False
    else:
        print("Assuming trainplot_1d got a model, not a trainstate")
        model = trainstate
        got_model = True
    figsize = (12, 16)
    x2 = None if len(x2etc) == 0 else x2etc[0]
    x3etc = x2etc[1:]
    if x2 == None and x1 is not None:
        x2 = range_batch(-1, 1, 1000) 
    if x1 == None:
        x1 = range_batch(-1, 1, 1000) 
        fig, axs = plt.subplots(5, 1, figsize=figsize)
        trainplot_1d_regression_over_axes(model, x1, axs, fig)
    else:
        fig, axs = plt.subplots(5, 2 + len(x3etc), figsize=figsize)
        trainplot_1d_regression_over_axes(model, x1, axs[:, 0], fig)
        trainplot_1d_regression_over_axes(model, x2, axs[:, 1], fig)
        for i, x in enumerate(x3etc):
            trainplot_1d_regression_over_axes(model, x, axs[:, i+2], fig) 
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.95)
    if got_model:
        fig.suptitle("[run from trainstate not model to see epoch/step]")
    else:
        fig.suptitle(f"Epoch {trainstate.epochs} step {trainstate.steps} (total steps taken: {trainstate.overall_steps})")
    #fig.show()
    return fig


 
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
    