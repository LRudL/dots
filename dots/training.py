import torch as t
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dots.utils import is_tensor, get_device
from dots.trainhooks import train_loss_hook, test_loss_hook

def train(
    model,
    optimiser,
    loss_fn,
    dataloader,
    epochs,
    hooks = [],
    progress_bars = False,
    wandb = None,
    trainstate=None
):
    device = get_device()
    total_steps = len(dataloader)
    epoch_iterator = tqdm(range(epochs)) if progress_bars else range(epochs)
    for epoch_n in epoch_iterator:
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            if epoch_n == 0 and i == 0:
                yhat = model(x)
                loss = loss_fn(yhat, y)
            else:
                optimiser.zero_grad()
                yhat = model(x)
                loss = loss_fn(yhat, y)
                loss.backward()
                optimiser.step()
            
            obj = {
                "epoch": epoch_n,
                "step": i,
                "step_overall": epoch_n * total_steps + i,
                "total_epochs": epochs,
                "total_steps": total_steps,
                "model": model,
                "optimiser": optimiser,
                "loss_fn": loss_fn,
                "train_loss": loss,
                "dataloader": dataloader,
                "x": x,
                "y": y,
                "predicted_y": yhat,
                "trainstate": trainstate
            }
            for hook in hooks:
                hook.run(obj)
            
            trainstate.epochs = epoch_n
            trainstate.steps = i
            #if wandb is not None:
            #    wandb.log(
            #        {
            #            "train_loss": loss.item(),
            #            "epoch": epoch_n
            #        },
            #        step = epoch_n * total_steps + i
            #    )

def wrap_train_with_hooks(add_hooks):
    return lambda model, optimiser, loss_fn, dataloader, epochs, hooks : train(
        model, optimiser, loss_fn, dataloader, epochs, hooks = add_hooks + hooks 
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
        train_dataloader,
        test_dataloader = None,
        val_dataloader = None,
        hooks = [],
        add_test_train_hooks = True,
        wandb = None
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.dataloader = train_dataloader
        self.test_loader = test_dataloader
        self.val_loader = val_dataloader
        
        default_hooks = []
        if add_test_train_hooks:
            train_hook = train_loss_hook(1, 1, wandb=wandb)
            default_hooks += [train_hook]
            if self.test_loader is not None:
                test_hook = test_loss_hook(
                    self.test_loader,
                    train_steps = -1,
                    wandb=wandb
                )
                default_hooks += [test_hook]
            else:
                print("Warning: no test loader provided, omitting test loss hook.")
        self.hooks = hooks + default_hooks
        
        self.epochs = 0
        self.steps = 0
        self.epoch_size = len(self.dataloader)
        
        self.checkpoints = []
        
        self.wandb = wandb
    
    def checkpoint(self):
        model_copy = deepcopy(self.model)
        self.checkpoints.append(
            TrainCheckpoint(self.epochs, model_copy)
        )
    
    def save_model(self, path):
        t.save(self.model, path)
    
    def train(self, epochs=1, checkpoint=False):
        train(model=self.model,
              optimiser=self.optimiser,
              loss_fn=self.loss_fn,
              dataloader=self.dataloader,
              epochs=epochs,
              hooks=self.hooks,
              wandb=self.wandb,
              trainstate=self)
        for hook in self.hooks:
            hook.increment_counters(epochs, epochs*len(self.dataloader))
        if checkpoint:
            self.checkpoint()
    
    def validation_loss(self):
        if self.val_loader is None:
            raise ValueError("No validation loader provided.") 
        total_items = 0
        loss_times_items = 0
        for (x, y) in self.val_loader:
            x = x.to(get_device())
            y = y.to(get_device())
            predicted_y = self.model(x)
            loss_times_items += self.loss_fn(predicted_y, y).item() * x.shape[0]
            total_items += x.shape[0]
        val_loss = loss_times_items / total_items
        if self.wandb is not None:
            self.wandb.log(
                {"val_loss" : val_loss},
                step=self.steps
            )
        return val_loss
    
    def hook_data(self):
        hook_groups = {}
        for hook in self.hooks:
            split = hook.name.split(", ")
            if hook.plot_hint == None:
                # Then we expect hook.storage to contain a straightforward
                # list of numbers
                if len(split) == 0:
                    assert split[0] not in hook_groups.keys(), "Hook name clash!"
                    hook_groups[split[0]] = {
                        "name": "",
                        "plot": "default",
                        "data": [
                            {
                                "name": split[0],
                                "x": np.array(hook.trigger_xs) / self.epoch_size,
                                "y": np.array(hook.storage)
                            }
                        ]}
                else:
                    if split[0] not in hook_groups.keys():
                        hook_groups[split[0]] = {
                            "name": split[0],
                            "plot": "default",
                            "data": []
                        }
                    hook_groups[split[0]]["data"].append({
                        "name": split[1],
                        "x": np.array(hook.trigger_xs) / self.epoch_size,
                        "y": np.array(hook.storage)
                    })
            elif hook.plot_hint == "image":
                # Then we expect hook.storage to contain 2D arrays / tensors
                hook_groups[hook.name] = {
                    "name": hook.name,
                    "plot": "image",
                    "data": hook.storage
                }
            elif hook.plot_hint == "multiline":
                # Then we expect hook.storage to contain, for each x,
                # a vector of values that at index i contains the next timestep
                # in some line that we want to plot across timesteps.
                lines = t.stack(hook.storage, dim=0)
                hook_groups[hook.name] = {
                    "name": hook.name,
                    "plot": "multiline",
                    "data": {
                        "lines" : lines,
                        "x": np.array(hook.trigger_xs) / self.epoch_size
                    }
                }
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
            if group["plot"] == "default":
                # one plot for each group
                for datum in group["data"]:
                    axs[i].plot(datum["x"], datum["y"], label=datum["name"])
                axs[i].set_title(gname)
                axs[i].set_xlabel("epoch")
                axs[i].legend()
                if gname == "loss":
                    axs[i].set_yscale("log")
            elif group["plot"] == "multiline":
                axs[i].set_title(group["name"])
                axs[i].set_xlabel("epoch")
                axs[i].set_yscale("log")
                axs[i].plot(group["data"]["x"], group["data"]["lines"], color="black", alpha=0.2)
        fig.show()
