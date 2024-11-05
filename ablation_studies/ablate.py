from pc_model import PCModel
from config import device
from pc_trainer import PCTrainer
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from data_loader import load_data
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

## Define all parameters we can tweak ##

# Model parameters
input_size = 28*28  # 28x28 images
hidden_size = 256
output_size = 10    # 10 classes
activation_fn = nn.ReLU
loss_fn = lambda output, _target: 0.5 * (output - _target).pow(2).sum() # this loss function holds to the error of the output layer of the model

# Trainer parameters 
T = 20 
optimizer_x_fn = optim.SGD
x_lr = 0.01
optimizer_p_fn = optim.Adam  
p_lr = 0.001

# Training parameters
epochs = 10 

## Initialize the model and trainer ##

train_loader, test_loader = load_data()
model = PCModel(input_size, hidden_size, output_size, activation_fn).model
trainer = PCTrainer(T, model, optimizer_x_fn, x_lr, optimizer_p_fn, p_lr).trainer

def test(): 
    model.eval()
    correct = 0
    total = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, predicted = torch.max(pred, -1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    model.train()
    return round(correct / total, 4)


## Train the model ##
test_acc = np.zeros(epochs + 1)
test_acc[0] = test()
for epoch in range(epochs):
    # Initialize the tqdm progress bar
    print(train_loader)
    with tqdm(train_loader, desc=f'Epoch {epoch+1} - Test accuracy: {test_acc[epoch]:.3f}') as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            # convert labels to one-hot encoding
            label = F.one_hot(label, num_classes=output_size).float()
            trainer.train_on_batch(
                inputs=data,
                loss_fn=loss_fn,
                loss_fn_kwargs={'_target': label}
            )
    test_acc[epoch + 1] = test()
    pbar.set_description(f'Epoch {epoch + 1} - Test accuracy: {test_acc[epoch + 1]:.3f}')

plt.plot(test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.savefig("test_acc.png")
