import torch as t
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

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
                 name="[unnamed hook]"):
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

def train(
    model,
    optimiser,
    loss_fn,
    dataloader,
    epochs,
    hooks = [],
    progress_bars = True
):
    total_steps = len(dataloader)
    epoch_iterator = tqdm(range(epochs)) if progress_bars else range(epochs)
    for epoch_n in epoch_iterator:
        for i, (x, y) in enumerate(dataloader):
            optimiser.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimiser.step()
            
            obj = {
                "epoch": epoch_n,
                "step": i,
                "total_epochs": epochs,
                "total_steps": total_steps,
                "model": model,
                "optimiser": optimiser,
                "loss_fn": loss_fn,
                "train_loss": loss,
                "dataloader": dataloader,
                "x": x,
                "y": y,
                "predicted_y": yhat
            }
            for hook in hooks:
                hook.run(obj)

def wrap_train_with_hooks(add_hooks):
    return lambda model, optimiser, loss_fn, dataloader, epochs, hooks : train(
        model, optimiser, loss_fn, dataloader, epochs, hooks = add_hooks + hooks 
    )

def property_storage_hook(fn, epochs=1, train_steps=1, name=None):
    vals = []
    return TrainHook(
        lambda obj : vals.append(fn(obj)),
        epochs,
        train_steps,
        storage = vals,
        name = name
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
    def get_test_loss(obj):
        total_items = 0
        loss_times_items = 0
        for (x, y) in test_dataloader:
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

def train_and_return_losses(
    model,
    optimiser,
    loss_fn,
    train_loader,
    test_loader,
    epochs,
    hooks = [],
    steps_per_test_loss = None
):
    train_hook = train_loss_hook(1, 1)
    test_hook = test_loss_hook(
        test_loader,
        train_steps = -1 if steps_per_test_loss is None else steps_per_test_loss
    )
    wrap_train_with_hooks([train_hook, test_hook])(
        model,
        optimiser,
        loss_fn,
        train_loader,
        epochs,
        hooks
    )
    return (
        (np.array(train_hook.trigger_xs), np.array(train_hook.storage)),
        (np.array(test_hook.trigger_xs), np.array(test_hook.storage))
    )

class TrainCheckpoint():
    def __init__(self, epoch, model):
        self.epoch = epoch
        self.model = model

class TrainState():
    def __init__(
        self,
        model,
        optimiser,
        loss_fn,
        dataloader,
        test_loader = None,
        hooks = [],
        add_test_train_hooks = True
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.test_loader = test_loader
        
        default_hooks = []
        if add_test_train_hooks:
            train_hook = train_loss_hook(1, 1)
            test_hook = test_loss_hook(
                test_loader,
                train_steps = -1
            )
            default_hooks += [train_hook, test_hook]
        self.hooks = hooks + default_hooks
        
        self.epochs = 0
        self.steps = 0
        self.epoch_size = len(self.dataloader)
        
        self.checkpoints = []
    
    def checkpoint(self):
        model_copy = deepcopy(self.model)
        self.checkpoints.append(
            TrainCheckpoint(self.epochs, model_copy)
        )
    
    def train(self, epochs=1, checkpoint=False):
        train(model=self.model,
              optimiser=self.optimiser,
              loss_fn=self.loss_fn,
              dataloader=self.dataloader,
              epochs=epochs,
              hooks=self.hooks)
        for hook in self.hooks:
            hook.increment_counters(epochs, epochs*len(self.dataloader))
        if checkpoint:
            self.checkpoint()
    
    def hook_data(self):
        hook_groups = {}
        for hook in self.hooks:
            split = hook.name.split(", ")
            if len(split) == 0:
                assert split[0] not in hook_groups.keys(), "Hook name clash!"
                hook_groups[split[0]] = {"name": "", "data": [
                    {
                        "name": split[0],
                        "x": np.array(hook.trigger_xs) / self.epoch_size,
                        "y": np.array(hook.storage)
                    }
                ]}
            else:
                if split[0] not in hook_groups.keys():
                    hook_groups[split[0]] = {"name": split[0], "data": []}
                hook_groups[split[0]]["data"].append({
                    "name": split[1],
                    "x": np.array(hook.trigger_xs) / self.epoch_size,
                    "y": np.array(hook.storage)
                })
        return hook_groups
    
    def plot(self, fig_size = (9, 5)):
        hook_groups = self.hook_data()
        num_groups = len(list(hook_groups.keys()))
        fig, axs = plt.subplots(
            num_groups,
            figsize=(fig_size[0], fig_size[1] * num_groups)
        )
        if num_groups == 1:
            axs = [axs]
        for i, (gname, group) in enumerate(hook_groups.items()):
            # one plot for each group
            for datum in group["data"]:
                axs[i].plot(datum["x"], datum["y"], label=datum["name"])
            axs[i].set_title(gname)
            axs[i].set_xlabel("epoch")
            axs[i].legend()
            if gname == "loss":
                axs[i].set_yscale("log")