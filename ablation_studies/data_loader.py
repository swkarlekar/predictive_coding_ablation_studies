import torch
from torchvision import datasets, transforms

def load_data(): 
    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)

    batch_size = 500
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'# train images: {len(train_dataset)} and # test images: {len(test_dataset)}')
    return train_loader, test_loader

