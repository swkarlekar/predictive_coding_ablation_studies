import sys
sys.path.append("..")
import predictive_coding as pc
from config import device 

class PCTrainer: 
    def __init__(self, T, model, optimizer_x_fn, x_lr, optimizer_p_fn, p_lr):
        self.T = T
        self.model = model
        self.optimizer_x = optimizer_x_fn
        self.optimizer_p = optimizer_p_fn
        self.x_lr = x_lr
        self.p_lr = p_lr
        self.trainer = pc.PCTrainer(model, 
            T = self.T, 
            optimizer_x_fn = self.optimizer_x,
            optimizer_x_kwargs = {'lr': self.x_lr},
            update_p_at = 'last',   
            optimizer_p_fn = self.optimizer_p,
            optimizer_p_kwargs = {'lr': self.p_lr},
        )