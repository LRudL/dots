
import torch as t
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dots.utils import is_tensor, get_device

def test_trigger(n, step_n, step_range):
    if step_n == -1:
        # this implies that we only want to trigger on the last case
        return n == step_range - 1
    else:
        # otherwise, trigger ever step_n steps
        return n % step_n == 0

class TrainHook:
    """
    Class for representing functions that should trigger at some point during
    the training loop. Arguments:
    - `fn` is the function to run
    - `epochs` sets the number of epochs between triggers;
    - `train_steps` the number of optimisation steps between triggers
    within a single epoch.
    - additional keyword arguments for maintaining state 
    For both arguments, set to -1 to trigger only on the last epoch/step.
    """
    def __init__(self,
                 fn,
                 epochs=1,
                 train_steps=1,
                 storage=None,
                 name="[unnamed hook]",
                 plot_hint=None):
        self.epochs = epochs
        self.train_steps = train_steps
        self.fn = fn
        self.storage = storage
        self.name = name
        self.trigger_steps = []
        self.trigger_epochs = []
        self.trigger_xs = []
        self.prior_epochs = 0
        self.prior_steps = 0
        self.plot_hint = plot_hint
    
    def will_trigger(self, epoch, step, total_epochs, total_steps):
        right_epoch = test_trigger(epoch, self.epochs, total_epochs)
        right_step  = test_trigger(step, self.train_steps, total_steps)
        return right_epoch and (right_step or
                                (self.train_steps == -1 and
                                 step == 0 and
                                 epoch == 0))
    
    def increment_counters(self, epochs, steps):
        self.prior_epochs += epochs
        self.prior_steps += steps
    
    def set_counters(self, epochs, steps):
        self.prior_epochs = epochs
        self.prior_steps = steps
    
    def run(self, obj):
        if self.will_trigger(obj["epoch"], obj["step"],
                             obj["total_epochs"], obj["total_steps"]):
            actual_epochs = self.prior_epochs + obj["epoch"]
            actual_steps = self.prior_steps + obj["step"]
            self.trigger_steps.append(actual_steps)
            self.trigger_epochs.append(actual_epochs)
            self.trigger_xs.append(
                self.prior_steps
                + obj["epoch"] * obj["total_steps"]
                + obj["step"]
            )
            
            self.fn(obj)

def property_storage_hook(
    fn,
    epochs=1,
    train_steps=1,
    name=None,
    plot_hint=None
):
    vals = []
    def val_append(fn_of_obj):
        if is_tensor(fn_of_obj):
            vals.append(fn_of_obj.detach().cpu())
        else:
            vals.append(fn_of_obj)
    return TrainHook(
        lambda obj : val_append(fn(obj)),
        epochs,
        train_steps,
        storage = vals,
        name = name,
        plot_hint = plot_hint
    )

train_loss_hook = lambda epochs=1, train_steps=1 : property_storage_hook(
    lambda obj : obj["train_loss"].item(),
    epochs,
    train_steps,
    name="loss, train"
)

jacobian_rank_hook = lambda x, epochs=1, train_steps=-1 : property_storage_hook(
    lambda obj : obj["model"].jacobian_matrix_rank(x),
    epochs,
    train_steps,
    name=f"DOTS, Jacobian rank w/ n={x.shape[0]}"
)

sv_rank_hook = lambda x, epochs=1, train_steps=-1 : property_storage_hook(
    lambda obj : obj["model"].singular_value_rank(x),
    epochs,
    train_steps,
    name=f"DOTS, sv rank w/ n={x.shape[0]}"
)

def test_loss_hook(test_dataloader, epochs=1, train_steps=-1):
    device = get_device()
    def get_test_loss(obj):
        total_items = 0
        loss_times_items = 0
        for (x, y) in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            predicted_y = obj["model"](x)
            loss = obj["loss_fn"]
            loss_times_items += loss(predicted_y, y).item() * x.shape[0]
            total_items += x.shape[0]
        return loss_times_items / total_items
    return property_storage_hook(
        get_test_loss,
        epochs,
        train_steps,
        name="loss, test"
    )

def accuracy_hook(test_dataloader, epochs=1, train_steps=-1):
    device = get_device()
    def get_acc(obj):
        total_items = 0
        correct_items = 0
        for (x, y) in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            out = obj["model"](x).argmax(dim=-1)
            total_items += out.shape[0]
            correct = (out == y).sum()
            correct_items += correct
        return correct_items / total_items
    return property_storage_hook(
        get_acc,
        epochs,
        train_steps,
        name="accuracy, test"
    )

jacobian_matrix_hook = lambda x, epochs=1, train_steps=-1 : property_storage_hook(
    lambda obj : obj["model"].matrix_jacobian(x),
    epochs,
    train_steps,
    name=f"Jacobians, w/ n={x.shape[0]}",
    plot_hint = "image"
)

parameter_importances_hook = lambda x, epochs=1, train_steps=-1 : property_storage_hook(
    lambda obj : obj["model"].jacobian_parameter_importances(x),
    epochs,
    train_steps,
    name=f"Parameter importances, w/ n={x.shape[0]}",
    plot_hint = "multiline"
)



