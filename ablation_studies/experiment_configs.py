import torch.nn as nn
import torch.optim as optim

## Define all parameters we can tweak ##
p_lr_exp_configs = {
    "parameter": "p_lr",
    # Model parameters
    "model": {
        "input_size": 28 * 28,  # 28x28 images
        "hidden_size": 256,
        "output_size": 10,  # 10 classes
        "activation_fn": nn.ReLU,
    },
    
    # Trainer parameters
    "trainer": {
        "T": 20,
        "optimizer_x_fn": optim.SGD,
        "x_lr": [0.1],
        "optimizer_p_fn": optim.Adam,
        "p_lrs": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    },
    
    # Training parameters
    "training": {
        "epochs": {
            0.5: 10,
            0.1: 10,
            0.05: 10,
            0.01: 10,
            0.005: 10,
            0.001: 10,
            0.0005: 10,
            0.0001: 10
        },
        "max_epochs": 10  # maximum of epoch values
    }
}

x_lr_exp_configs = {
    "parameter": "x_lr",
    # Model parameters
    "model": {
        "input_size": 28 * 28,  # 28x28 images
        "hidden_size": 256,
        "output_size": 10,  # 10 classes
        "activation_fn": nn.ReLU,
    },
    
    # Trainer parameters
    "trainer": {
        "T": 20,
        "optimizer_x_fn": optim.SGD,
        "x_lr": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "optimizer_p_fn": optim.Adam,
        "p_lrs": [0.001],
    },
    
    # Training parameters
    "training": {
        "epochs": {
            0.5: 10,
            0.1: 10,
            0.05: 10,
            0.01: 10,
            0.005: 10,
            0.001: 10,
            0.0005: 10,
            0.0001: 10
        },
        "max_epochs": 10  # maximum of epoch values
    }
}