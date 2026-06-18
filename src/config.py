import os
import torch

# Chemins dynamiques (fonctionne sur n'importe quelle machine)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')

MNIST_CKPT = os.path.join(SAVE_DIR, "best_mnist_model.pth")
CIFAR_CKPT = os.path.join(SAVE_DIR, "best_cifar10_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tenseurs de normalisation statiques
MNIST_MEAN = torch.tensor([0.1307], device=DEVICE).view(1, 1, 1, 1)
MNIST_STD  = torch.tensor([0.3081], device=DEVICE).view(1, 1, 1, 1)

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1, 3, 1, 1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE).view(1, 3, 1, 1)

def normalize_mnist(images):
    return (images - MNIST_MEAN) / MNIST_STD

def normalize_cifar(images):
    return (images - CIFAR_MEAN) / CIFAR_STD
