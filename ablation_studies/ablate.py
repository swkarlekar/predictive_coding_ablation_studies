from pc_model import PCModel
from config import device
from pc_trainer import PCTrainer

import numpy as np 
from data_loader import load_data
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from experiment_configs import p_lr_exp_configs, x_lr_exp_configs



exp_configs = p_lr_exp_configs
# max_epochs = max(epochs.values())

## Initialize the model and trainer ##

train_loader, test_loader = load_data()

def test(model): 
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, predicted = torch.max(pred, -1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    model.train()
    accuracy = round(correct / total, 4)
    return accuracy

# Initialize dictionaries to store accuracy and final loss
accuracy_per_lr = {}
final_losses = []
for x_lr in exp_configs["trainer"]["x_lr"]:
    for p_lr in exp_configs["trainer"]["p_lr"]: 
        model = PCModel(**exp_configs['model']).model
        trainer = PCTrainer(exp_configs['trainer']['T'], model, exp_configs['trainer']['optimizer_x_fn'], x_lr, exp_configs['trainer']['optimizer_p_fn'], p_lr).trainer
        ## Train the model ##
        eps = exp_configs['training']['epochs'][p_lr]
        test_acc = np.zeros(eps + 1)
        test_acc[0] = test(model)
        for epoch in range(eps):
            # Initialize the tqdm progress bar
            print(train_loader)
            with tqdm(train_loader, desc=f'Epoch {epoch+1} - Test accuracy: {test_acc[epoch]:.3f}') as pbar:
                for data, label in pbar:
                    data, label = data.to(device), label.to(device)
                    # convert labels to one-hot encoding
                    label = F.one_hot(label, num_classes=output_size).float()
                    trainer.train_on_batch(
                        inputs=data,
                        loss_fn=lambda output, _target: 0.5 * (output - _target).pow(2).sum(),
                        loss_fn_kwargs={'_target': label}
                    )
            test_acc[epoch + 1] = test(model)
            pbar.set_description(f'Epoch {epoch + 1} - Test accuracy: {test_acc[epoch + 1]:.3f}')
        accuracy_per_lr[p_lr] = test_acc
        final_losses.append((p_lr, accuracy_per_lr[p_lr][-1]))
    

# Plot (1): Test accuracy across epochs for each learning rate
plt.figure(figsize=(12, 8))
for p_lr, accuracies in accuracy_per_lr.items():
    # Pad the accuracies to max_epochs
    accuracies = np.append(accuracies, [accuracies[-1]] * (max_epochs + 1 - len(accuracies)))
    plt.plot(range(max_epochs + 1), accuracies, label=f'LR={p_lr}')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Test Accuracy Across Epochs for Different Learning Rates')
plt.grid(True)
plt.savefig("test_accuracy_epochs_p_lr.png")
plt.show()

# Plot (2): Final test loss for each learning rate
x_lrs, final_test_losses = zip(*final_losses)
plt.figure(figsize=(8, 6))
plt.plot(x_lrs, final_test_losses, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Final Test Accuracy')
plt.xscale('log')
plt.title('Final Test Accuracy for Each Learning Rate')
plt.grid(True)
plt.savefig("final_test_acc_p_lr.png")
plt.show()
