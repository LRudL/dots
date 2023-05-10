import wandb
import torch as t
import torch.utils.data as tdata
from dots.training import train, TrainState
from dots.utils import range_batch, get_device
from dots.datasets import get_dataset
from dots.models import *


def get_model(name):
    match name:
        case "MLP":
            return MLP
        case "DeepLinear":
            return DeepLinear
        case "BasicCNN":
            return BasicCNN
        case _:
            raise ValueError(f"Unknown model name: {name}")

def get_optimiser(name):
    match name:
        case "SGD":
            return t.optim.SGD
        case "Adam":
            return t.optim.Adam
        case _:
            raise ValueError(f"Unknown optimiser name: {name}")

def get_loss_fn(name):
    match name:
        case "MSELoss":
            return t.nn.MSELoss
        case "CrossEntropy":
            return t.nn.CrossEntropyLoss
        case _:
            raise ValueError(f"Unknown loss function name: {name}")

def run_experiment(
    given_config
):
    with wandb.init(project="DOTS", config=given_config):
        config = wandb.config
        
        # datasets have their own manual seeding, so do this first:
        dataset = get_dataset(config["dataset"]["name"])
        dataset_split_sizes = [
            int(frac * len(dataset)) 
            for frac in config["dataset"]["train_test_val_split"]
        ]
        
        if config.get("seed") is not None:
            t.manual_seed(config.seed)
        
        # want dataset shuffle to be done after seeding:
        train_ds, test_ds, val_ds = tdata.random_split(
            dataset, 
            lengths = dataset_split_sizes
        )
        train_dataloader = tdata.DataLoader(
            train_ds,
            batch_size=config["hp"]["batch_size"],
            shuffle=True
        )
        test_dataloader = tdata.DataLoader(
            test_ds,
            batch_size=config["hp"]["batch_size"],
            shuffle=True
        )
        val_dataloader = tdata.DataLoader(
            val_ds,
            batch_size=config["hp"]["batch_size"],
            shuffle=True
        )
        
        device = get_device()
        model = get_model(config["model_class"])(**config["model"])
        model.to(device)
        
        optimiser = get_optimiser(config["hp"]["optimiser"])(
            model.parameters(),
            **config["hp"]["optimiser_args"]
        )
        loss_fn = get_loss_fn(config["hp"]["loss_fn"])()
        
        train_state = TrainState(
            model,
            optimiser,
            loss_fn,
            train_dataloader,
            test_dataloader,
            val_dataloader,
            hooks = [],
            add_test_train_hooks = True,
            wandb = wandb
        )
        
        train_state.train(epochs=config["hp"]["epochs"])
        
        train_state.validation_loss() 
        # ^this will also log the validation loss in wandb 
        
        return train_state

def run_sweep(config, sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="DOTS")
    wandb.agent(sweep_id, function = lambda  : run_experiment(wandb.config))
