import os
import torch
from src.config import DEVICE, SAVE_DIR, normalize_mnist, normalize_cifar
from src.models import MNISTNet, CIFAR10Net
from src.data_loader import get_dataloaders
from src.train import train_baseline, evaluate_model

def main():
    print(f"[INFO] Démarrage sur : {DEVICE}")
    loaders = get_dataloaders(batch_size=256)
    
    # === Exemple : Évaluation ou Entraînement de MNIST ===
    print("\n--- Modèle Baseline MNIST ---")
    mnist_model = MNISTNet().to(DEVICE)
    mnist_save_path = os.path.join(SAVE_DIR, "mnist_baseline.pth")
    
    # Décommente cette ligne si tu veux relancer l'entraînement
    # train_baseline(mnist_model, loaders['mnist_train'], loaders['mnist_test'], epochs=5, lr=1e-3, normalize_fn=normalize_mnist, save_path=mnist_save_path)
    
    if os.path.exists(mnist_save_path):
        mnist_model.load_state_dict(torch.load(mnist_save_path, map_location=DEVICE))
        print("[INFO] Poids MNIST chargés avec succès.")
    else:
        print("[AVERTISSEMENT] Aucun poids trouvé. Pense à lancer l'entraînement.")
    
    acc = evaluate_model(mnist_model, loaders['mnist_test'], normalize_mnist)
    print(f"-> Accuracy sur Test MNIST Clean : {acc:.2f}%\n")
    print("[SUCCÈS] L'architecture est en place. Importe `fgsm_attack` et `pgd_attack` depuis `src.attacks` pour tester la robustesse !")

if __name__ == "__main__":
    main()
