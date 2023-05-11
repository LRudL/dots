import os
import yaml
import wandb
import torch as t
import torch.utils.data as tdata
from dots.training import train, TrainState
from dots.utils import range_batch, get_device, tensor_of_dataset
from dots.datasets import get_dataset
from dots.models import *
from dots.trainhooks import jacobian_rank_hook


def get_model(name):
    match name:
        case "MLP":
            return MLP
        case "DeepLinear":
            return DeepLinear
        case "MNIST_MLP":
            return MNIST_MLP
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
    # make sure this remains idempotent; in some cases it's called twice
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

def parse_config(config, wandb=None):
    config = process_config(config)
    
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
    
    hooks = []
    
    if config.get("log_accuracy") is not None:
        hooks.append(accuracy_hook(test_dataloader, wandb=wandb))
    
    real_in_size = model.in_size#config.get("modelarg_in_size", model.in_size)
    
    if config.get("log_dots") is not None:
        rand_data = [
            range_batch(
                start=-t.ones(real_in_size),
                end=t.ones(real_in_size),
                n=rand_dots_size
            )
            for rand_dots_size in config["log_dots"]
        ] 
        for x in rand_data:
            hooks.append(
                jacobian_rank_hook(
                    x,
                    epochs=1,
                    name_extra="(rand)",
                    wandb=wandb
                )
            )
    if config.get("log_datadots") is not None:
        datas = [
            tensor_of_dataset(dataset, range(0, data_dots_size)) 
            for data_dots_size in config["log_datadots"]
        ] 
        if datas[-1].shape[-1] > dataset_split_sizes[0]:
            print("WARNING: using test data for data-based DOTS")
        for x in rand_data:
            hooks.append(
                jacobian_rank_hook(
                    x, 
                    epochs=1, 
                    name_extra="(data)", 
                    wandb=wandb
                )
            )
    
    return {
        "model" : model,
        "optimiser" : optimiser,
        "loss_fn" : loss_fn,
        "train_dataloader" : train_dataloader,
        "test_dataloader" : test_dataloader,
        "val_dataloader" : val_dataloader,
        "hooks" : hooks
    }

def get_train_state_from_config(config, **kwargs):
    config = parse_config(config)
    config.update(kwargs)
    return TrainState(**config)

def get_train_state(file_or_config, **kwargs):
    if isinstance(file_or_config, str):
        with open(file_or_config, "r") as f:
            config = yaml.safe_load(f)
            return get_train_state_from_config(config, **kwargs)
    return get_train_state_from_config(file_or_config, **kwargs)

def run_experiment(
    given_config,
    save_loc = "models/"
):
    if save_loc is not None:
        os.makedirs(save_loc, exist_ok=True)
        
    with wandb.init(project="DOTS", config=given_config):
        config = process_config(wandb.config.as_dict())
        train_state = get_train_state_from_config(
            config,
            wandb=wandb,
            add_test_train_hooks=True
        )
        
        train_state.train(epochs=config["hp_epochs"])
        
        train_state.validation_loss() 
        # ^this will also log the validation loss in wandb 
        
        train_state.save_model(save_loc + wandb.run.name + ".pt")
                
        return train_state

def run_sweep(config, sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="DOTS")
    wandb.agent(sweep_id, function = lambda  : run_experiment(wandb.config))
