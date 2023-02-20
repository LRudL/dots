import torch as t
import numpy as np

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
    def __init__(self, fn, epochs=1, train_steps=1, storage=None):
        self.epochs = epochs
        self.train_steps = train_steps
        self.fn = fn
        self.storage = storage
        self.trigger_steps = []
        self.trigger_epochs = []
        self.trigger_xs = []
    
    def will_trigger(self, epoch, step, total_epochs, total_steps):
        right_epoch = test_trigger(epoch, self.epochs, total_epochs)
        right_step  = test_trigger(step, self.train_steps, total_steps)
        return right_epoch and right_step
    
    def run(self, obj):
        if self.will_trigger(obj["epoch"], obj["step"],
                             obj["total_epochs"], obj["total_steps"]):
            self.trigger_steps.append(obj["step"])
            self.trigger_epochs.append(obj["epoch"])
            self.trigger_xs.append(
                obj["epoch"] * obj["total_steps"] + obj["step"]
            )
            self.fn(obj)

def train(
    model,
    optimiser,
    loss_fn,
    dataloader,
    epochs,
    hooks = []
):
    total_steps = len(dataloader)
    for epoch_n in range(epochs):
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

def property_storage_hook(fn, epochs=1, train_steps=1):
    vals = []
    return TrainHook(
        lambda obj : vals.append(fn(obj)),
        epochs,
        train_steps,
        storage = vals 
    )

train_loss_hook = lambda epochs=1, train_steps=1 : property_storage_hook(
    lambda obj : obj["train_loss"].item(),
    epochs,
    train_steps
)

jacobian_rank_hook = lambda x, ep=1, steps=1 : property_storage_hook(
    lambda obj : obj["model"].jacobian_matrix_rank(x),
    ep,
    steps
)

sv_rank_hook = lambda x, epochs=1, train_steps=1 : property_storage_hook(
    lambda obj : obj["model"].singular_value_rank(x),
    epochs,
    train_steps
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
        train_steps
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
