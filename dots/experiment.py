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

def get_subargs_config(argprefix, config):
    subargs = {}
    for key in config.keys():
        if key.startswith(argprefix):
            subargs[key[len(argprefix):]] = config[key]
    return subargs

def process_config(config):
    def value_of_key(key):
        val = config[key]
        if isinstance(val, str):
            if val[0] == "[" and val[-1] == "]":
                print(f"Warning: coerced string '{val}' to list")
                val = eval(val) 
            else:
                try:
                    val = int(val)
                except:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
        return val
    return {
        key : value_of_key(key)
        for key in config
    }

def run_experiment(
    given_config
):
    with wandb.init(project="DOTS", config=given_config):
        config = process_config(wandb.config.as_dict())
        
        # datasets have their own manual seeding, so do this first:
        dataset = get_dataset(config["dataset_name"])
        split_fracs = [
            config["dataset_train_frac"],
            config["dataset_test_frac"],
            config["dataset_val_frac"]
        ]
        dataset_split_sizes = [
            int(frac * len(dataset)) 
            for frac in split_fracs
        ]
        
        if config.get("seed") is not None:
            t.manual_seed(config["seed"])
        
        # want dataset shuffle to be done after seeding:
        train_ds, test_ds, val_ds = tdata.random_split(
            dataset, 
            lengths = dataset_split_sizes
        )
        train_dataloader = tdata.DataLoader(
            train_ds,
            batch_size=config["hp_batch_size"],
            shuffle=True
        )
        test_dataloader = tdata.DataLoader(
            test_ds,
            batch_size=config["hp_batch_size"],
            shuffle=True
        )
        val_dataloader = tdata.DataLoader(
            val_ds,
            batch_size=config["hp_batch_size"],
            shuffle=True
        )
        
        device = get_device()
        model_config = get_subargs_config("modelarg_", config)
        model = get_model(config["model_class"])(**model_config)
        model.to(device)
        
        opt_config = get_subargs_config("hp_optarg_", config)
        optimiser = get_optimiser(config["hp_opt_name"])(
            model.parameters(),
            **opt_config
        )
        loss_fn = get_loss_fn(config["loss_fn"])()
        
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
        
        train_state.train(epochs=config["hp_epochs"])
        
        train_state.validation_loss() 
        # ^this will also log the validation loss in wandb 
        
        return train_state

def run_sweep(config, sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="DOTS")
    wandb.agent(sweep_id, function = lambda  : run_experiment(wandb.config))
