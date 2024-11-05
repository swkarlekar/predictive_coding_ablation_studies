from config import device
import torch.nn as nn
import sys
sys.path.append("..")
import predictive_coding as pc

class PCModel: 
    def __init__(self, input_size, hidden_size, output_size, activation_fn):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            pc.PCLayer(),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            pc.PCLayer(),
            activation_fn(),
            nn.Linear(hidden_size, output_size)
        )
        self.model.train()
        self.model.to(device)
