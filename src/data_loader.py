import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from src.config import DATA_DIR

def get_test_dataloaders():
    base_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    mnist_test_ds = torchvision.datasets.MNIST(DATA_DIR, train=False, download=False, transform=base_transform)
    cifar_test_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=base_transform)
    
    mnist_test_loader = DataLoader(mnist_test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    cifar_test_loader = DataLoader(cifar_test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    return mnist_test_loader, cifar_test_loader
