import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from src.config import DATA_DIR

def get_dataloaders(batch_size=256):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True)
    ])
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    mnist_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=False, transform=test_transform)
    mnist_test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=False, transform=test_transform)
    
    cifar_train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=train_transform)
    cifar_test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=test_transform)

    loaders = {
        'mnist_train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'mnist_test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        'cifar_train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'cifar_test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }
    return loaders
